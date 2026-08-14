#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <limits>
#include <unistd.h>

#include "client.h"
#include <mpi.h>


int check_run(MPI_Comm comm, SmartRedis::Client *client)
{
    int exit_val = 1;
    std::string run_key = "check-run";
    int check_run_val = 0;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Check if check-run tensor exists in DB from head rank
    if (rank == 0) {
        if (client->tensor_exists(run_key)) {
            client->unpack_tensor(run_key, &check_run_val, {1},
                SRTensorTypeInt32, SRMemLayoutContiguous);
            exit_val = check_run_val;
        }
    }
    MPI_Bcast(&exit_val, 1, MPI_INT, 0, comm);

    if (exit_val == 0 && rank == 0) {
        std::cout << "[Sim] Consumer says time to quit" << std::endl;
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

    // Parse args
    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "[Sim] Usage: " << argv[0]
                      << " <bytes_per_rank> <db_nodes>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
    long long bytes_per_rank = std::stoll(argv[1]);
    int db_nodes = std::stoi(argv[2]);

    if (bytes_per_rank % sizeof(double) != 0) {
        if (rank == 0) {
            std::cerr << "[Sim] bytes_per_rank (" << bytes_per_rank
                      << ") must be a multiple of sizeof(double) = " << sizeof(double) << std::endl;
        }
        MPI_Abort(comm, 1);
    }
    long long N = bytes_per_rank / sizeof(double);

    if (rank == 0) {
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        std::cout << "[Sim] Running on " << hostname << " with " << size << " MPI ranks and "
                  << static_cast<double>(bytes_per_rank) / 1e9 << " GB per rank"
                  << " (db_nodes=" << db_nodes << ")" << std::endl;
    }

    // Initialize SmartRedis client
    bool cluster_mode = (db_nodes > 1);
    std::string logger_name("Client");
    if (rank == 0) std::cout << "[Sim] Initializing SmartRedis client..." << std::endl;
    SmartRedis::Client client(cluster_mode, logger_name);
    MPI_Barrier(comm);

    // Setup iteration loop
    int iters = 1000;
    int sleep_time = 2000;
    std::vector<double> U(N, 0.0);
    std::string key = "y." + std::to_string(rank);
    double put_time = 0.0, transfer_time = 0.0;
    int completed_iters = 0;

    for (int iter = 0; iter < iters; iter++) {
        // Check for quit signal from consumer
        int exit_val = check_run(comm, &client);
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
        client.put_tensor(key, U.data(),
                          {static_cast<size_t>(N)},
                          SRTensorTypeDouble, SRMemLayoutContiguous);
        double put_end = MPI_Wtime();
        if (iter > 0) put_time += put_end - put_start;

        MPI_Barrier(comm);
        double toc = MPI_Wtime();
        if (iter > 0) transfer_time += toc - tic;

        if (rank == 0) {
            std::cout << "[Sim] Iter " << iter << ": " << toc - tic << " s" << std::endl;
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

        // Sum of per-rank rates
        double local_rank_bw = (static_cast<double>(bytes_per_rank) / 1e9) / put_time;
        double sum_of_rates = 0.0;
        MPI_Allreduce(&local_rank_bw, &sum_of_rates, 1, MPI_DOUBLE, MPI_SUM, comm);

        if (rank == 0) {
            double gb_per_rank = static_cast<double>(bytes_per_rank) / 1e9;
            double gb_per_iter = gb_per_rank * size;
            std::cout << "\n=== Producer Performance Summary ===" << std::endl;
            std::cout << "Producer ranks: " << size << std::endl;
            std::cout << "Data per rank per iter: " << gb_per_rank << " GB" << std::endl;
            std::cout << "Total data per iter: " << gb_per_iter << " GB" << std::endl;
            std::cout << "Iterations timed: " << (completed_iters - 1) << " (iter 0 = warmup)" << std::endl;
            std::cout << "Avg per-rank put time: " << avg_put_time << " s" << std::endl;
            std::cout << "Min per-rank put time: " << min_put_time << " s (fastest rank)" << std::endl;
            std::cout << "Max per-rank put time: " << max_put_time << " s (slowest rank)" << std::endl;
            std::cout << "Wall-clock transfer time (barrier-to-barrier): " << transfer_time << " s" << std::endl;
            std::cout << "Avg per-rank bandwidth (from put time): " << gb_per_rank / avg_put_time << " GB/s" << std::endl;
            std::cout << "Peak per-rank bandwidth (from min put time): " << gb_per_rank / min_put_time << " GB/s" << std::endl;
            std::cout << "Aggregate bandwidth (sum of per-rank rates):    " << sum_of_rates << " GB/s" << std::endl;
            std::cout << "Aggregate bandwidth (from wall-clock barriers): " << gb_per_iter / transfer_time << " GB/s" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
