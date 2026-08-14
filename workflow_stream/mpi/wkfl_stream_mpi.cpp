#include <mpi.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <limits>

int main(int argc, char *argv[])
{
    int rank;
    int size;
    int provide;

    // MPI_THREAD_MULTIPLE is only required if you enable the SST MPI_DP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provide);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm comm = MPI_COMM_WORLD;
    if (size % 2 != 0) {
        if (rank == 0) {
            std::cerr << "This communication pattern requires an even MPI size";
        }
        MPI_Abort(comm, 1);
    }

    // Read input
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <bytes_per_rank>" << std::endl;
        std::cerr << "Expected 1 argument, got " << (argc - 1) << std::endl;
        MPI_Finalize();
        return -1;
    }
    long long int bytes_per_rank = std::stoll(argv[1]);
    if (bytes_per_rank % sizeof(double) != 0) {
        if (rank == 0) {
            std::cerr << "bytes_per_rank (" << bytes_per_rank << ") must be a multiple of sizeof(double) = "
                      << sizeof(double) << std::endl;
        }
        MPI_Abort(comm, 1);
    }
    long long int N = bytes_per_rank / sizeof(double);

    if (rank == 0) {
        std::cout << "Running with " << size << " MPI ranks and "
                  << static_cast<double>(bytes_per_rank) / 1e9 << " GB "
                  << "per rank\n" << std::endl;
    }

    // Define size of data
    int half_size = size / 2;

    // Setup iteration loop
    int iters = 20;
    int sleep_time = 2000; // in ms
    std::vector<double> U(N, 0.0);
    double send_time = 0.0, recv_time = 0.0, transfer_time = 0.0;
    for (int iter=0; iter<iters; iter++) {
        // Update solution vector from producer ranks and sleep to emulate compute time
        if (rank < half_size) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
            double frac = (iter != 0) ? (1.0 / iter) : 0.0;
            for (int n=0; n<N; n++) {
                U[n] = static_cast<double>(n+frac);
            }
        }
        MPI_Barrier(comm);

        // Emulate producer/consumer pattern with ranks in lower half sending data to upper half
        MPI_Status status;
        double tic = MPI_Wtime();
        if (rank < half_size) {
            int dest_rank = rank + half_size;
            double start_time = MPI_Wtime();
            MPI_Send(U.data(), N, MPI_DOUBLE, dest_rank, 0, comm);
            double end_time = MPI_Wtime();
            if (iter > 0) {
                send_time += end_time - start_time;
            }
        } else {
            int src_rank = rank - half_size;
            double start_time = MPI_Wtime();
            MPI_Recv(U.data(), N, MPI_DOUBLE, src_rank, 0, comm, &status);
            double end_time = MPI_Wtime();
            if (iter > 0) {
                recv_time += end_time - start_time;
            }
        }
        MPI_Barrier(comm);
        double toc = MPI_Wtime();
        if (iter > 0) {
            transfer_time += toc - tic;
        }
        if (rank == 0) {
            std::cout << "Iteration " << iter << ": " << toc - tic << " seconds" << std::endl;
        }
    }

    // Per-rank per-iteration averages (iter 0 is warmup)
    send_time /= (iters - 1);
    recv_time /= (iters - 1);
    transfer_time /= (iters - 1);

    // Average send and receive times across the active half of ranks
    double avg_send_time = 0.0, avg_recv_time = 0.0;
    MPI_Allreduce(&send_time, &avg_send_time, 1, MPI_DOUBLE, MPI_SUM, comm);
    avg_send_time /= half_size;
    MPI_Allreduce(&recv_time, &avg_recv_time, 1, MPI_DOUBLE, MPI_SUM, comm);
    avg_recv_time /= half_size;

    // Max and min recv times across pairs (senders contribute 0, so use only receiver values)
    double max_recv_time = 0.0;
    MPI_Allreduce(&recv_time, &max_recv_time, 1, MPI_DOUBLE, MPI_MAX, comm);
    double local_recv_for_min = (rank >= half_size) ? recv_time : std::numeric_limits<double>::infinity();
    double min_recv_time = 0.0;
    MPI_Allreduce(&local_recv_for_min, &min_recv_time, 1, MPI_DOUBLE, MPI_MIN, comm);

    // Sum of per-pair rates: sum over consumers of (bytes / local recv_time)
    double local_pair_bw = (rank >= half_size)
        ? (static_cast<double>(bytes_per_rank) / 1e9) / recv_time
        : 0.0;
    double sum_of_rates = 0.0;
    MPI_Allreduce(&local_pair_bw, &sum_of_rates, 1, MPI_DOUBLE, MPI_SUM, comm);

    // Print results
    if (rank == 0) {
        double gb_per_pair = static_cast<double>(bytes_per_rank) / 1e9; 
        double gb_per_iter = gb_per_pair * half_size;
        std::cout << "\n=== Communication Performance Summary ===" << std::endl;
        std::cout << "Producer/consumer pairs: " << half_size << std::endl;
        std::cout << "Data per message (per pair): " << gb_per_pair << " GB" << std::endl;
        std::cout << "Total data per iteration (all pairs): " << gb_per_iter << " GB" << std::endl;
        std::cout << "Iterations timed: " << (iters - 1) << " (iter 0 = warmup)" << std::endl;
        std::cout << "Avg per-pair send time:  " << avg_send_time << " s" << std::endl;
        std::cout << "Avg per-pair recv time:  " << avg_recv_time << " s" << std::endl;
        std::cout << "Min per-pair recv time:  " << min_recv_time << " s (fastest pair)" << std::endl;
        std::cout << "Max per-pair recv time:  " << max_recv_time << " s (slowest pair)" << std::endl;
        std::cout << "Wall-clock transfer time (barrier-to-barrier): " << transfer_time << " s" << std::endl;
        std::cout << "Avg per-pair bandwidth (from send time): " << gb_per_pair / avg_send_time << " GB/s" << std::endl;
        std::cout << "Avg per-pair bandwidth (from recv time): " << gb_per_pair / avg_recv_time << " GB/s" << std::endl;
        std::cout << "Peak per-pair bandwidth (from min recv time):   " << gb_per_pair / min_recv_time << " GB/s" << std::endl;
        std::cout << "Aggregate bandwidth (sum of per-pair rates):    " << sum_of_rates << " GB/s" << std::endl;
        std::cout << "Aggregate bandwidth (from wall-clock barriers): " << gb_per_iter / transfer_time << " GB/s" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
