#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <limits>
#include <unistd.h>

#include <dragon/dictionary.hpp>
#include <dragon/serializable.hpp>
#include "cpp_serializers.hpp"
#include <mpi.h>

// Default DDict operation timeout
static timespec_t TIMEOUT = {600, 0};

// DDict type used by this proxy simulation:
//   keys  : strings (e.g. "check-run", "y.<rank>")
//   values: 1D vectors of doubles
using SimDDict = dragon::DDict<dragon::SerializableString,
                               custom::SerializableDoubleVector>;

// Logger
static FILE *g_log_fp = nullptr;
static int   g_log_rank = 0;
static int   g_log_size = 0;
static bool  g_debug_enabled = false;

static void log_init(MPI_Comm comm, const char *path)
{
    MPI_Comm_rank(comm, &g_log_rank);
    MPI_Comm_size(comm, &g_log_size);

    if (g_log_rank == 0) {
        FILE *f = std::fopen(path, "w");
        if (f) std::fclose(f);
    }
    MPI_Barrier(comm);
    g_log_fp = std::fopen(path, "a");
}

static void log_close()
{
    if (g_log_fp) {
        std::fflush(g_log_fp);
        std::fclose(g_log_fp);
        g_log_fp = nullptr;
    }
}

static void log_line(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void log_line(const char *fmt, ...)
{
    if (!g_log_fp) return;

    char ts[32];
    std::time_t t = std::time(nullptr);
    std::tm tm_val;
    localtime_r(&t, &tm_val);
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_val);

    std::fprintf(g_log_fp, "%s [rank %3d/%d] INFO ",
                 ts, g_log_rank, g_log_size);

    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_log_fp, fmt, ap);
    va_end(ap);

    std::fputc('\n', g_log_fp);
    std::fflush(g_log_fp);
}

#define log_debug(...) do { if (g_debug_enabled) log_line(__VA_ARGS__); } while (0)


int check_run(MPI_Comm comm, SimDDict *dd)
{
    int exit_val = 1;
    dragon::SerializableString run_key("check-run");
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        if (dd->contains(run_key)) {
            custom::SerializableDoubleVector run_val = (*dd)[run_key];
            const std::vector<double> &run_val_vec = run_val.getVal();
            if (!run_val_vec.empty()) {
                exit_val = static_cast<int>(run_val_vec[0]);
            }
        }
    }
    MPI_Bcast(&exit_val, 1, MPI_INT, 0, comm);

    if (exit_val == 0 && rank == 0) {
        log_line("[Sim] Consumer says time to quit");
    }
    return exit_val;
}


int main(int argc, char *argv[])
{
    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Log path: use $PRODUCER_LOG if set (driver injects this), else producer.out
    const char *log_path_env = std::getenv("PRODUCER_LOG");
    log_init(comm, log_path_env ? log_path_env : "producer.out");

    // Parse positional args, then optional flags
    if (argc < 4) {
        if (rank == 0) {
            log_line("[Sim] Usage: %s <deployment> <bytes_per_rank> <serialized_ddict> [--verbose]",
                     argv[0]);
        }
        log_close();
        MPI_Finalize();
        return -1;
    }
    std::string deployment = argv[1];
    long long bytes_per_rank = std::stoll(argv[2]);
    const char *ddict_ser = argv[3];
    for (int i = 4; i < argc; i++) {
        if (std::strcmp(argv[i], "--verbose") == 0) {
            g_debug_enabled = true;
        } else if (rank == 0) {
            log_line("[Sim] Unknown flag '%s'; ignoring", argv[i]);
        }
    }

    if (bytes_per_rank % sizeof(double) != 0) {
        if (rank == 0) {
            log_line("[Sim] bytes_per_rank (%lld) must be a multiple of sizeof(double) = %zu",
                     bytes_per_rank, sizeof(double));
        }
        log_close();
        MPI_Abort(comm, 1);
    }
    long long N = bytes_per_rank / sizeof(double);

    if (rank == 0) {
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        log_line("[Sim] Running on %s with %d MPI ranks and %g GB per rank (deployment=%s)",
                 hostname, size, static_cast<double>(bytes_per_rank) / 1e9, deployment.c_str());
        if (g_debug_enabled) log_line("[Sim] Debug logging enabled");
    }

    // Attach to the Distributed Dictionary created on the Python side
    if (rank == 0) log_line("[Sim] Attaching to Dragon DDict...");
    SimDDict dd(ddict_ser, &TIMEOUT);
    MPI_Barrier(comm);

    // For colocated deployments, use the local manager to access local data only
    if (deployment == "colocated") {
        dd = dd.manager(dd.local_manager());
    }

    // Setup iteration loop
    int iters = 1000;
    int sleep_time = 2000;
    std::vector<double> U(N, 0.0);
    dragon::SerializableString U_key("y." + std::to_string(rank));
    double put_time = 0.0, transfer_time = 0.0;
    int completed_iters = 0;

    for (int iter = 0; iter < iters; iter++) {
        int exit_val = check_run(comm, &dd);
        if (exit_val == 0) break;

        // Emulate compute time then fill buffer
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        double frac = (iter != 0) ? (1.0 / iter) : 0.0;
        for (long long n = 0; n < N; n++) {
            U[n] = static_cast<double>(n + frac);
        }

        MPI_Barrier(comm);
        double tic = MPI_Wtime();

        double put_start = MPI_Wtime();
        custom::SerializableDoubleVector U_value(U);
        dd[U_key] = U_value;
        double put_end = MPI_Wtime();
        if (iter > 0) put_time += put_end - put_start;

        MPI_Barrier(comm);
        double toc = MPI_Wtime();
        if (iter > 0) transfer_time += toc - tic;

        if (rank == 0) {
            log_line("[Sim] Iter %d: %.6f s", iter, toc - tic);
        }
        completed_iters = iter + 1;
    }

    // Metrics
    if (completed_iters > 1) {
        put_time /= (completed_iters - 1);
        transfer_time /= (completed_iters - 1);

        double avg_put_time = 0.0, max_put_time = 0.0, min_put_time = 0.0;
        MPI_Allreduce(&put_time, &avg_put_time, 1, MPI_DOUBLE, MPI_SUM, comm);
        avg_put_time /= size;
        MPI_Allreduce(&put_time, &max_put_time, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&put_time, &min_put_time, 1, MPI_DOUBLE, MPI_MIN, comm);

        double local_rank_bw = (static_cast<double>(bytes_per_rank) / 1e9) / put_time;
        double sum_of_rates = 0.0;
        MPI_Allreduce(&local_rank_bw, &sum_of_rates, 1, MPI_DOUBLE, MPI_SUM, comm);

        if (rank == 0) {
            double gb_per_rank = static_cast<double>(bytes_per_rank) / 1e9;
            double gb_per_iter = gb_per_rank * size;
            log_line("=== Producer Performance Summary ===");
            log_line("Producer ranks: %d", size);
            log_line("Data per rank per iter: %g GB", gb_per_rank);
            log_line("Total data per iter: %g GB", gb_per_iter);
            log_line("Iterations timed: %d (iter 0 = warmup)", completed_iters - 1);
            log_line("Avg per-rank put time: %g s", avg_put_time);
            log_line("Min per-rank put time: %g s (fastest rank)", min_put_time);
            log_line("Max per-rank put time: %g s (slowest rank)", max_put_time);
            log_line("Wall-clock transfer time (barrier-to-barrier): %g s", transfer_time);
            log_line("Avg per-rank bandwidth (from put time): %g GB/s", gb_per_rank / avg_put_time);
            log_line("Peak per-rank bandwidth (from min put time): %g GB/s", gb_per_rank / min_put_time);
            log_line("Aggregate bandwidth (sum of per-rank rates):    %g GB/s", sum_of_rates);
            log_line("Aggregate bandwidth (from wall-clock barriers): %g GB/s", gb_per_iter / transfer_time);
        }
    }

    log_close();
    MPI_Finalize();
    return 0;
}
