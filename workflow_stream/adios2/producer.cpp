#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <limits>
#include <unistd.h>

#include "adios2.h"
#include <mpi.h>


int check_run(MPI_Comm comm, adios2::IO checkIO, const std::string& check_path)
{
    int exit_val = 1;
    int exists;
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        exists = std::filesystem::exists(check_path) ? 1 : 0;
        if (exists) {
            printf("[Sim] Found check-run file!\n");
            fflush(stdout);
        }
    }
    MPI_Bcast(&exists, 1, MPI_INT, 0, comm);

    if (exists) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        adios2::Engine reader = checkIO.Open(check_path, adios2::Mode::Read);
        reader.BeginStep();
        adios2::Variable<int> var = checkIO.InquireVariable<int>("check-run");
        if (rank == 0 && var) {
            reader.Get(var, &exit_val);
        }
        reader.EndStep();
        reader.Close();
        MPI_Bcast(&exit_val, 1, MPI_INT, 0, comm);
    }

    if (exit_val == 0 && rank == 0) {
        printf("[Sim] Consumer says time to quit\n");
        fflush(stdout);
    }
    return exit_val;
}


int main(int argc, char *argv[])
{
    int global_rank, rank;
    int global_size, size;
    int provide;

    // MPI_THREAD_MULTIPLE is required if you enable the SST MPI_DP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provide);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

    int color = 5678;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, global_rank, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Parse args
    if (argc != 6) {
        if (rank == 0) {
            std::cerr << "[Sim] Usage: " << argv[0]
                      << " <bytes_per_rank> <engine:bp5|sst> <sst_mode:sync|async>"
                      << " <data_plane:WAN|MPI|UCX|RDMA|fabric> <io_mode:posix|daos>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
    long long bytes_per_rank = std::stoll(argv[1]);
    std::string engine = argv[2];
    std::string sst_mode = argv[3];
    std::string data_plane = argv[4];
    std::string io_mode = argv[5];

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
                  << " (engine=" << engine;
        if (engine == "sst") {
            std::cout << " sst_mode=" << sst_mode << " data_plane=" << data_plane;
        }
        std::cout << " io_mode=" << io_mode << ")" << std::endl;
    }

    std::string path_prefix = (io_mode == "daos") ? "/tmp/datascience/balin/" : "./";
    std::string check_path = path_prefix + "check-run.bp";
    std::string solution_path = path_prefix + "solution.bp";
    std::string ready_path = path_prefix + "solution.ready";

    try
    {
        adios2::ADIOS adios(comm);
        adios2::IO streamIO = adios.DeclareIO("solutionStream");
        adios2::IO checkIO = adios.DeclareIO("checkIO");

        std::string open_path;
        if (engine == "bp5") {
            streamIO.SetEngine("BP5");
            open_path = solution_path;
        } else if (engine == "sst") {
            streamIO.SetEngine("SST");
            adios2::Params params;
            if (sst_mode == "sync") {
                params["RendezvousReaderCount"] = "1";
                params["QueueFullPolicy"] = "Block";
                params["QueueLimit"] = "1";
            } else if (sst_mode == "async") {
                params["RendezvousReaderCount"] = "1";
                params["QueueFullPolicy"] = "Discard";
                params["QueueLimit"] = "3";
                params["ReserveQueueLimit"] = "0";
            }
            params["DataTransport"] = data_plane;
            params["OpenTimeoutSecs"] = "600";
            streamIO.SetParameters(params);
            open_path = "solutionStream";
        } else {
            if (rank == 0) std::cerr << "[Sim] Unknown engine: " << engine << std::endl;
            MPI_Abort(comm, 1);
        }

        // Define variables (uniform N across ranks)
        unsigned long long _N = static_cast<unsigned long long>(N);
        unsigned long long _global_N = _N * static_cast<unsigned long long>(size);
        unsigned long long _offset_N = _N * static_cast<unsigned long long>(rank);
        auto UVar = streamIO.DefineVariable<double>("U", {_global_N}, {_offset_N}, {_N});
        auto bprVar = streamIO.DefineVariable<long long>("bytes_per_rank");

        // Setup iteration loop
        int iters = 1000;
        int sleep_time = 2000;
        std::vector<double> U(N, 0.0);
        double put_time = 0.0, transfer_time = 0.0;
        int completed_iters = 0;

        if (rank == 0) std::cout << "[Sim] Opening stream..." << std::endl;
        adios2::Engine solWriter = streamIO.Open(open_path, adios2::Mode::Write);

        for (int iter = 0; iter < iters; iter++) {
            // Check for quit signal from consumer
            int exit_val = check_run(comm, checkIO, check_path);
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
            solWriter.BeginStep();
            solWriter.Put<double>(UVar, U.data());
            if (iter == 0 && rank == 0) {
                solWriter.Put<long long>(bprVar, bytes_per_rank);
            }
            solWriter.EndStep();
            double put_end = MPI_Wtime();
            if (iter > 0) put_time += put_end - put_start;

            MPI_Barrier(comm);
            double toc = MPI_Wtime();
            if (iter > 0) transfer_time += toc - tic;

            // After first step, drop a sentinel so the consumer knows the BP file is safe to open
            if (engine == "bp5" && iter == 0) {
                MPI_Barrier(comm);
                if (rank == 0) {
                    std::ofstream(ready_path).close();
                    std::cout << "[Sim] Wrote sentinel " << ready_path << std::endl;
                }
                MPI_Barrier(comm);
            }

            if (rank == 0) {
                std::cout << "[Sim] Iter " << iter << ": " << toc - tic << " s" << std::endl;
            }
            completed_iters = iter + 1;
        }
        solWriter.Close();

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
    }
    catch (std::invalid_argument &e)
    {
        std::cout << "[Sim] Invalid argument exception, STOPPING from rank " << rank << ": " << e.what() << std::endl;
    }
    catch (std::ios_base::failure &e)
    {
        std::cout << "[Sim] IO base failure exception, STOPPING from rank " << rank << ": " << e.what() << std::endl;
    }
    catch (std::exception &e)
    {
        std::cout << "[Sim] Exception, STOPPING from rank " << rank << ": " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
