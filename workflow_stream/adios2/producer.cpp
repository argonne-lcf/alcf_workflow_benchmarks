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
#include <sycl/sycl.hpp>


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
    if (argc < 6 || argc > 7) {
        if (rank == 0) {
            std::cerr << "[Sim] Usage: " << argv[0]
                      << " <bytes_per_rank> <engine:bp5|sst> <sst_mode:sync|async>"
                      << " <data_plane:WAN|MPI|UCX|RDMA|fabric> <io_mode:posix|daos>"
                      << " [device:gpu|cpu]" << std::endl;
            std::cerr << "[Sim]   device defaults to gpu" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
    long long bytes_per_rank = std::stoll(argv[1]);
    std::string engine = argv[2];
    std::string sst_mode = argv[3];
    std::string data_plane = argv[4];
    std::string io_mode = argv[5];
    std::string device = (argc == 7) ? std::string(argv[6]) : std::string("gpu");
    if (device != "gpu" && device != "cpu") {
        if (rank == 0) {
            std::cerr << "[Sim] device must be 'gpu' or 'cpu', got '" << device << "'" << std::endl;
        }
        MPI_Abort(comm, 1);
    }
    bool on_gpu = (device == "gpu");

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
        std::cout << " io_mode=" << io_mode << " device=" << device << ")" << std::endl;
    }

    // SYCL queue is only used when buffers live on the GPU
    // Round-robin across the local node's GPUs
    sycl::queue Q;
    if (on_gpu) {
        std::vector<sycl::device> gpu_devices;
        for (const auto& plat : sycl::platform::get_platforms()) {
            if (plat.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
            for (const auto& dev : plat.get_devices()) {
                if (dev.is_gpu()) {
                    gpu_devices.push_back(dev);
                }
            }
        }
        if (gpu_devices.empty()) {
            std::cerr << "[Sim] [rank " << rank << "] No Level Zero GPU devices found!" << std::endl;
            MPI_Abort(comm, 1);
        }
        int local_idx = rank % static_cast<int>(gpu_devices.size());
        Q = sycl::queue(gpu_devices[local_idx]);
        std::cout << "[Sim] [rank " << rank << "] SYCL device (" << local_idx
                  << "/" << gpu_devices.size() << "): "
                  << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
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
        int sleep_time;
        if (engine == "bp5") {
            streamIO.SetEngine("BP5");
            open_path = solution_path;
            sleep_time = 2000;
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
            sleep_time = 500;
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

        // Tell ADIOS2 the memory space of the buffer we'll hand to Put
        if (on_gpu) {
            UVar.SetMemorySpace(adios2::MemorySpace::GPU);
        }

        // Setup iteration loop. 
        int iters = 1000;
        std::vector<double> U_host;
        double *U_gpu = nullptr;
        double *U = nullptr;
        if (on_gpu) {
            U_gpu = sycl::malloc_device<double>(N, Q);
            Q.memset(U_gpu, 0, N * sizeof(double)).wait();
            U = U_gpu;
        } else {
            U_host.assign(N, 0.0);
            U = U_host.data();
        }
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
            if (on_gpu) {
                double *U_ptr = U;
                long long N_local = N;
                Q.parallel_for(sycl::range<1>(N_local), [=](sycl::id<1> idx) {
                    long long n = static_cast<long long>(idx[0]);
                    U_ptr[n] = static_cast<double>(n) + frac;
                }).wait();
            } else {
                for (long long n = 0; n < N; n++) {
                    U[n] = static_cast<double>(n) + frac;
                }
            }

            MPI_Barrier(comm);
            double tic = MPI_Wtime();

            double put_start = MPI_Wtime();
            solWriter.BeginStep();
            // Sync mode forces ADIOS2 to consume U before Put returns, so the next
            // iteration is free to overwrite the USM buffer. With the default Deferred
            // mode + SST async, the transport thread can still be reading U while our
            // kernel rewrites it, which shows up as GPU page faults ("NotPresent").
            solWriter.Put<double>(UVar, U, adios2::Mode::Sync);
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
        std::cout << "[Sim][DBG] rank " << rank << ": loop exited, completed_iters=" << completed_iters << std::endl;
        std::cout.flush();

        MPI_Barrier(comm);
        if (rank == 0) std::cout << "[Sim][DBG] before solWriter.Close()" << std::endl;
        solWriter.Close();
        MPI_Barrier(comm);
        if (rank == 0) std::cout << "[Sim][DBG] after solWriter.Close()" << std::endl;

        // Cleanup
        if (on_gpu) {
            if (rank == 0) std::cout << "[Sim][DBG] before sycl::free(U_gpu)" << std::endl;
            sycl::free(U_gpu, Q);
            if (rank == 0) std::cout << "[Sim][DBG] after sycl::free(U_gpu)" << std::endl;
        }
        MPI_Barrier(comm);
        if (rank == 0) std::cout << "[Sim][DBG] entering metrics section (completed_iters=" << completed_iters << ")" << std::endl;

        // Metrics
        if (completed_iters > 1) {
            put_time /= (completed_iters - 1);
            transfer_time /= (completed_iters - 1);
            std::cout << "[Sim][DBG] rank " << rank << ": local put_time=" << put_time
                      << " transfer_time=" << transfer_time << std::endl;
            std::cout.flush();
            MPI_Barrier(comm);

            double avg_put_time = 0.0, max_put_time = 0.0, min_put_time = 0.0;
            if (rank == 0) std::cout << "[Sim][DBG] before Allreduce(SUM, put_time)" << std::endl;
            MPI_Allreduce(&put_time, &avg_put_time, 1, MPI_DOUBLE, MPI_SUM, comm);
            if (rank == 0) std::cout << "[Sim][DBG] after  Allreduce(SUM, put_time)" << std::endl;
            avg_put_time /= size;
            if (rank == 0) std::cout << "[Sim][DBG] before Allreduce(MAX, put_time)" << std::endl;
            MPI_Allreduce(&put_time, &max_put_time, 1, MPI_DOUBLE, MPI_MAX, comm);
            if (rank == 0) std::cout << "[Sim][DBG] after  Allreduce(MAX, put_time)" << std::endl;
            if (rank == 0) std::cout << "[Sim][DBG] before Allreduce(MIN, put_time)" << std::endl;
            MPI_Allreduce(&put_time, &min_put_time, 1, MPI_DOUBLE, MPI_MIN, comm);
            if (rank == 0) std::cout << "[Sim][DBG] after  Allreduce(MIN, put_time)" << std::endl;

            // Sum of per-rank rates
            double local_rank_bw = (static_cast<double>(bytes_per_rank) / 1e9) / put_time;
            double sum_of_rates = 0.0;
            if (rank == 0) std::cout << "[Sim][DBG] before Allreduce(SUM, local_rank_bw)" << std::endl;
            MPI_Allreduce(&local_rank_bw, &sum_of_rates, 1, MPI_DOUBLE, MPI_SUM, comm);
            if (rank == 0) std::cout << "[Sim][DBG] after  Allreduce(SUM, local_rank_bw)" << std::endl;

            if (rank == 0) {
                double gb_per_rank = static_cast<double>(bytes_per_rank) / 1e9;
                double gb_per_iter = gb_per_rank * size;
                // For SST, the Put call only stages metadata -- bytes move at the consumer's
                // Get. Reporting producer-side times/BW would be misleading, so zero them out.
                // Only BP5 (where EndStep actually writes to disk) gets meaningful numbers.
                bool sst = (engine == "sst");
                double r_avg_put   = sst ? 0.0 : avg_put_time;
                double r_min_put   = sst ? 0.0 : min_put_time;
                double r_max_put   = sst ? 0.0 : max_put_time;
                double r_transfer  = sst ? 0.0 : transfer_time;
                double r_avg_bw    = sst ? 0.0 : gb_per_rank / avg_put_time;
                double r_peak_bw   = sst ? 0.0 : gb_per_rank / min_put_time;
                double r_sum_rates = sst ? 0.0 : sum_of_rates;
                double r_wall_bw   = sst ? 0.0 : gb_per_iter / transfer_time;

                std::cout << "\n=== Producer Performance Summary ===" << std::endl;
                std::cout << "Producer ranks: " << size << std::endl;
                std::cout << "Data per rank per iter: " << gb_per_rank << " GB" << std::endl;
                std::cout << "Total data per iter: " << gb_per_iter << " GB" << std::endl;
                std::cout << "Iterations timed: " << (completed_iters - 1) << " (iter 0 = warmup)" << std::endl;
                if (sst) {
                    std::cout << "(SST engine: bytes move on consumer Get, not producer Put; "
                              << "producer-side numbers reported as 0.)" << std::endl;
                }
                std::cout << "Avg per-rank put time: " << r_avg_put << " s" << std::endl;
                std::cout << "Min per-rank put time: " << r_min_put << " s (fastest rank)" << std::endl;
                std::cout << "Max per-rank put time: " << r_max_put << " s (slowest rank)" << std::endl;
                std::cout << "Wall-clock transfer time (barrier-to-barrier): " << r_transfer << " s" << std::endl;
                std::cout << "Avg per-rank bandwidth (from put time): " << r_avg_bw << " GB/s" << std::endl;
                std::cout << "Peak per-rank bandwidth (from min put time): " << r_peak_bw << " GB/s" << std::endl;
                std::cout << "Aggregate bandwidth (sum of per-rank rates):    " << r_sum_rates << " GB/s" << std::endl;
                std::cout << "Aggregate bandwidth (from wall-clock barriers): " << r_wall_bw << " GB/s" << std::endl;
            }
        }
        MPI_Barrier(comm);
        if (rank == 0) std::cout << "[Sim][DBG] finished metrics section, exiting try block" << std::endl;
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

    if (rank == 0) std::cout << "[Sim][DBG] before MPI_Finalize" << std::endl;
    MPI_Finalize();
    if (rank == 0) std::cout << "[Sim][DBG] after MPI_Finalize (returning)" << std::endl;
    return 0;
}
