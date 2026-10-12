// mpi_engine.hpp
// MPI distributed-memory driver, shared by the pure-MPI (dna_mpi) and
// hybrid MPI+OpenMP (dna_hybrid) executables. The two binaries are built
// from the SAME source (src/mpi/main.cpp and src/hybrid/main.cpp are
// thin wrappers around run_mpi_cli): dna_mpi is compiled with mpic++
// only, dna_hybrid is compiled with `mpic++ -fopenmp`. When built
// without -fopenmp, --threads is ignored inside each rank (the OpenMP
// pragmas in the shared algorithm headers become no-ops), giving pure
// process-level parallelism; with -fopenmp, each rank additionally
// spawns an OpenMP thread team, giving the hybrid model described in
// Spec section 18.
//
// Two distinct MPI parallelization strategies are used, matching the
// two different parallelism structures identified in Spec section 19:
//
//   1. Exact / approximate matching -> INTRA-SEQUENCE parallelism.
//      The genome is one long string; it is partitioned into
//      contiguous, overlapping (haloed) chunks and distributed with
//      MPI_Scatterv, matched locally, then results are combined with
//      MPI_Gatherv. See Spec section 11 for the halo/boundary logic
//      (reused directly from exact_match.hpp / approximate_match.hpp).
//
//   2. Smith-Waterman alignment -> INTER-SEQUENCE (batch) parallelism.
//      Wavefront-parallel single-pair alignment does not distribute
//      cleanly across nodes without per-diagonal halo exchange on every
//      step (very high message-count overhead for the gain). Instead,
//      matching the literature note in Spec section 15 ("parallelizing
//      multiple alignments and computation within individual
//      alignments"), MPI distributes a *batch* of database sequences to
//      align against one query across ranks (inter-sequence
//      parallelism), while OpenMP still parallelizes the wavefront
//      *within* each individual alignment (intra-sequence parallelism)
//      when running as the hybrid binary. This is a deliberate,
//      documented scope decision (see report/design_decisions.md).
#pragma once

#include <mpi.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>

#include "dna_common.hpp"
#include "exact_match.hpp"
#include "approximate_match.hpp"
#include "smith_waterman.hpp"

namespace dna {

// Computes contiguous start-position partition boundaries across `size`
// workers, identical formula to the OpenMP partitioning in
// exact_match.hpp / approximate_match.hpp so both levels of parallelism
// use one consistent decomposition rule.
inline std::vector<size_t> partition_bounds(size_t total, int size) {
    std::vector<size_t> bounds(size + 1);
    for (int t = 0; t <= size; ++t) {
        bounds[t] = (total * static_cast<size_t>(t)) / static_cast<size_t>(size);
    }
    return bounds;
}

// ---------------------------------------------------------------------
// Strategy 1: exact / approximate matching via genome Scatterv/Gatherv
// ---------------------------------------------------------------------
inline int run_mpi_search_cli(int rank, int size, ArgParser& args) {
    std::string input = args.get("--input");
    std::string query = args.get("--query");
    std::string algorithm = args.get("--algorithm", "exact");
    int threads = args.get_int("--threads", 1);
    int mismatches = args.get_int("--mismatches", 1);
#ifndef _OPENMP
    threads = 1;
#endif

    std::string text;
    size_t total_bases = 0;

    if (rank == 0) {
        if (input.empty() || query.empty()) {
            std::cerr << "Usage: mpirun -np N ./dna_mpi --input <fasta> --query <DNA> "
                      << "--algorithm exact|approximate [--threads N] [--mismatches K]\n";
        }
        try {
            auto records = parse_fasta(input);
            for (auto& r : records) text += r.sequence;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing FASTA: " << e.what() << "\n";
        }
        total_bases = text.size();
    }

    // Broadcast basic parameters + failure/empty signal so all ranks exit
    // together on error instead of hanging.
    long long n_signed = rank == 0 ? static_cast<long long>(total_bases) : 0;
    MPI_Bcast(&n_signed, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (n_signed <= 0) {
        if (rank == 0) std::cerr << "No data to process; aborting.\n";
        return 1;
    }
    total_bases = static_cast<size_t>(n_signed);

    int qlen_i = static_cast<int>(query.size());
    MPI_Bcast(&qlen_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) query.resize(qlen_i);
    MPI_Bcast(&query[0], qlen_i, MPI_CHAR, 0, MPI_COMM_WORLD);
    const size_t m = query.size();

    // ---- Partition + Scatterv the genome (with halo for boundary safety) ----
    auto bounds = partition_bounds(total_bases - m + 1, size); // over start positions
    std::vector<int> sendcounts(size), displs(size);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            size_t owned_len = bounds[r + 1] - bounds[r];
            size_t local_len = std::min(owned_len + (m - 1), total_bases - bounds[r]);
            sendcounts[r] = static_cast<int>(local_len);
            displs[r] = static_cast<int>(bounds[r]);
        }
    }
    // every rank needs its own owned_len/local_len regardless of role
    size_t owned_len = bounds[rank + 1] - bounds[rank];
    size_t local_len = std::min(owned_len + (m - 1), total_bases - bounds[rank]);

    std::string local_text(local_len, '\0');
    MPI_Scatterv(rank == 0 ? text.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_CHAR,
                 &local_text[0], static_cast<int>(local_len), MPI_CHAR,
                 0, MPI_COMM_WORLD);

    Timer timer;
    MPI_Barrier(MPI_COMM_WORLD); // align start line for fair timing
    timer.reset();

    std::vector<long long> local_positions; // as long long for clean MPI typing
    std::vector<int> local_mismatch_counts;  // only used for approximate mode

    if (algorithm == "exact") {
        auto matches = exact_match_search(local_text, query, threads, ExactAlgo::KMP);
        for (size_t p : matches) {
            if (p < owned_len) local_positions.push_back(static_cast<long long>(bounds[rank] + p));
        }
    } else if (algorithm == "approximate") {
        auto matches = approximate_match_search(local_text, query, mismatches, threads);
        for (auto& mtch : matches) {
            if (mtch.position < owned_len) {
                local_positions.push_back(static_cast<long long>(bounds[rank] + mtch.position));
                local_mismatch_counts.push_back(mtch.mismatches);
            }
        }
    } else {
        if (rank == 0) std::cerr << "Unknown search algorithm for MPI mode: " << algorithm << "\n";
        return 1;
    }

    double local_elapsed = timer.elapsed_seconds();
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Gatherv results back to rank 0 ----
    int local_count = static_cast<int>(local_positions.size());
    std::vector<int> counts(size), rdispls(size);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<long long> all_positions;
    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < size; ++r) { rdispls[r] = total; total += counts[r]; }
        all_positions.resize(total);
    }
    MPI_Gatherv(local_positions.data(), local_count, MPI_LONG_LONG,
                rank == 0 ? all_positions.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? rdispls.data() : nullptr,
                MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::sort(all_positions.begin(), all_positions.end());
        std::cout << "=================================================\n";
        std::cout << "       PARALLEL DNA SEQUENCE ANALYSIS ENGINE\n";
        std::cout << "=================================================\n\n";
        std::cout << "Input:\n" << input << "\n\nDataset Size:\n"
                  << format_number(static_cast<long long>(total_bases)) << " bases\n\n";
        std::cout << "Query:\n" << query << "\n\n";
#ifdef _OPENMP
        std::cout << "Execution Mode:\nHybrid MPI + OpenMP\nMPI Processes: " << size
                  << "\nOpenMP Threads per Process: " << threads << "\n\n";
#else
        std::cout << "Execution Mode:\nMPI (" << size << " processes)\n\n";
#endif
        std::cout << "-------------------------------------------------\n\n";
        std::cout << "Matches found: " << all_positions.size() << "\n\n";
        size_t show = std::min<size_t>(all_positions.size(), 20);
        for (size_t i = 0; i < show; ++i) std::cout << "Position " << all_positions[i] << "\n";
        if (all_positions.size() > show) std::cout << "... (" << all_positions.size() - show << " more)\n";
        std::cout << "\n-------------------------------------------------\n\n";
        std::cout << "Execution Time (max across ranks): " << std::fixed
                  << std::setprecision(4) << max_elapsed << " s\n";
        if (max_elapsed > 0)
            std::cout << "Throughput: " << std::setprecision(2)
                      << (total_bases / 1e6) / max_elapsed << " million bases/s\n";
        std::cout << "\n-------------------------------------------------\n";
    }
    return 0;
}

// ---------------------------------------------------------------------
// Strategy 2: Smith-Waterman batch alignment distributed across ranks
// ---------------------------------------------------------------------
inline int run_mpi_align_cli(int rank, int size, ArgParser& args) {
    std::string input = args.get("--input");
    std::string query = args.get("--query");
    int threads = args.get_int("--threads", 1);
#ifndef _OPENMP
    threads = 1;
#endif
    AlignmentScoring scoring;
    scoring.match = args.get_int("--match", 2);
    scoring.mismatch = args.get_int("--mismatch", -1);
    scoring.gap = args.get_int("--gap", -2);

    // Rank 0 parses the database FASTA (each record = one sequence to
    // align the query against) and serializes it for broadcast.
    std::vector<std::string> db;
    std::string blob;      // all sequences concatenated with '\n' separators
    std::vector<int> lens; // length of each sequence

    if (rank == 0) {
        try {
            auto records = parse_fasta(input);
            for (auto& r : records) {
                db.push_back(r.sequence);
                lens.push_back(static_cast<int>(r.sequence.size()));
                blob += r.sequence;
                blob.push_back('\n');
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing FASTA: " << e.what() << "\n";
        }
    }

    int nrecords = static_cast<int>(db.size());
    MPI_Bcast(&nrecords, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (nrecords == 0) {
        if (rank == 0) std::cerr << "No database sequences found; aborting.\n";
        return 1;
    }
    if (rank != 0) lens.resize(nrecords);
    MPI_Bcast(lens.data(), nrecords, MPI_INT, 0, MPI_COMM_WORLD);

    int blob_len = static_cast<int>(blob.size());
    MPI_Bcast(&blob_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) blob.resize(blob_len);
    MPI_Bcast(&blob[0], blob_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    int qlen = static_cast<int>(query.size());
    MPI_Bcast(&qlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) query.resize(qlen);
    MPI_Bcast(&query[0], qlen, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Every rank re-splits the blob into individual sequences locally
    // (cheap, avoids sending variable-length data structures).
    if (rank != 0) {
        db.resize(nrecords);
        size_t pos = 0;
        for (int i = 0; i < nrecords; ++i) {
            db[i] = blob.substr(pos, lens[i]);
            pos += lens[i] + 1; // +1 for '\n' separator
        }
    }

    // Block-partition record indices across ranks (inter-sequence parallelism).
    auto bounds = partition_bounds(static_cast<size_t>(nrecords), size);

    MPI_Barrier(MPI_COMM_WORLD);
    Timer timer;

    int local_best_score = -1;
    int local_best_idx = -1;
    std::vector<std::pair<int, int>> local_scores; // (record_index, score)

    for (size_t i = bounds[rank]; i < bounds[rank + 1]; ++i) {
        auto result = smith_waterman(query, db[i], scoring, threads);
        local_scores.push_back({static_cast<int>(i), result.score});
        if (result.score > local_best_score) {
            local_best_score = result.score;
            local_best_idx = static_cast<int>(i);
        }
    }

    double local_elapsed = timer.elapsed_seconds();
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // MPI_MAXLOC to find which rank holds the single best-scoring record.
    struct { double score; int rank; } local_ml, global_ml;
    local_ml.score = local_best_score;
    local_ml.rank = rank;
    MPI_Reduce(&local_ml, &global_ml, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    // MPI_Reduce only populates the recvbuf on the root; every rank needs
    // to know who the winner is (so that rank, and only that rank, sends
    // its full traceback below), so broadcast the winning rank id out.
    int winner_rank = (rank == 0) ? global_ml.rank : -1;
    MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather every rank's per-record scores back to rank 0 for a summary table.
    int local_n = static_cast<int>(local_scores.size());
    std::vector<int> counts(size), rdispls(size);
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> flat_local; // (idx, score) pairs flattened
    for (auto& p : local_scores) { flat_local.push_back(p.first); flat_local.push_back(p.second); }

    std::vector<int> gcounts(size), gdispls(size);
    std::vector<int> flat_all;
    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < size; ++r) { gcounts[r] = counts[r] * 2; gdispls[r] = total; total += gcounts[r]; }
        flat_all.resize(total);
    }
    MPI_Gatherv(flat_local.data(), static_cast<int>(flat_local.size()), MPI_INT,
                rank == 0 ? flat_all.data() : nullptr,
                rank == 0 ? gcounts.data() : nullptr,
                rank == 0 ? gdispls.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 requests the full traceback of the global-best alignment
    // from whichever rank owns it (Send/Recv point-to-point).
    if (rank == 0) {
        std::string best_a, best_b, best_m;
        int best_idx = -1;
        if (winner_rank == 0) {
            auto result = smith_waterman(query, db[local_best_idx], scoring, threads);
            best_a = result.aligned_a; best_b = result.aligned_b; best_m = result.match_line;
            best_idx = local_best_idx;
        } else {
            MPI_Recv(&best_idx, 1, MPI_INT, winner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int lens3[3];
            MPI_Recv(lens3, 3, MPI_INT, winner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            best_a.resize(lens3[0]); best_b.resize(lens3[1]); best_m.resize(lens3[2]);
            MPI_Recv(&best_a[0], lens3[0], MPI_CHAR, winner_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_b[0], lens3[1], MPI_CHAR, winner_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_m[0], lens3[2], MPI_CHAR, winner_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::cout << "=================================================\n";
        std::cout << "       PARALLEL DNA SEQUENCE ANALYSIS ENGINE\n";
        std::cout << "=================================================\n\n";
        std::cout << "Database:\n" << input << " (" << nrecords << " sequences)\n\n";
        std::cout << "Query:\n" << query << "\n\n";
#ifdef _OPENMP
        std::cout << "Execution Mode:\nHybrid MPI + OpenMP (batch alignment)\nMPI Processes: "
                  << size << "\nOpenMP Threads per Process: " << threads << "\n\n";
#else
        std::cout << "Execution Mode:\nMPI batch alignment (" << size << " processes)\n\n";
#endif
        std::cout << "-------------------------------------------------\n\n";
        std::cout << "Best match: sequence #" << best_idx
                  << "  (score = " << global_ml.score << ")\n\n";
        std::cout << best_a << "\n" << best_m << "\n" << best_b << "\n\n";
        std::cout << "-------------------------------------------------\n";
        std::cout << "Execution Time (max across ranks): " << std::fixed
                  << std::setprecision(4) << max_elapsed << " s\n";
        std::cout << "-------------------------------------------------\n";
    } else if (rank == winner_rank) {
        // Recompute this rank's winning alignment fully (with traceback
        // strings) and send it to rank 0 on request.
        auto result = smith_waterman(query, db[local_best_idx], scoring, threads);
        MPI_Send(&local_best_idx, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        int lens3[3] = {static_cast<int>(result.aligned_a.size()),
                         static_cast<int>(result.aligned_b.size()),
                         static_cast<int>(result.match_line.size())};
        MPI_Send(lens3, 3, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(result.aligned_a.data(), lens3[0], MPI_CHAR, 0, 2, MPI_COMM_WORLD);
        MPI_Send(result.aligned_b.data(), lens3[1], MPI_CHAR, 0, 3, MPI_COMM_WORLD);
        MPI_Send(result.match_line.data(), lens3[2], MPI_CHAR, 0, 4, MPI_COMM_WORLD);
    }
    return 0;
}

inline int run_mpi_cli(int argc, char** argv) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ArgParser args(argc, argv);
    std::string algorithm = args.get("--algorithm", "exact");

    int rc;
    if (algorithm == "align") {
        rc = run_mpi_align_cli(rank, size, args);
    } else {
        rc = run_mpi_search_cli(rank, size, args);
    }
    return rc;
}

} // namespace dna
