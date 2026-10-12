// engine_main.hpp
// Shared command-line driver for the sequential and OpenMP executables.
// Both src/sequential/main.cpp and src/openmp/main.cpp just call
// dna::run_engine_cli(argc, argv). The ONLY difference between the two
// binaries is the compiler flag -fopenmp: without it, all #pragma omp
// directives in the headers are ignored and everything below runs with
// nthreads effectively forced to 1 (the sequential baseline), which is
// exactly the "same source proves parallel correctness" design called
// for by the spec's validation requirement (Section 21).
#pragma once

#include <iostream>
#include <iomanip>
#include "dna_common.hpp"
#include "exact_match.hpp"
#include "approximate_match.hpp"
#include "smith_waterman.hpp"

namespace dna {

inline int run_engine_cli(int argc, char** argv) {
    ArgParser args(argc, argv);

    std::string input = args.get("--input");
    std::string query = args.get("--query");
    std::string algorithm = args.get("--algorithm", "exact");
    int threads = args.get_int("--threads", 1);
    int mismatches = args.get_int("--mismatches", 1);

#ifndef _OPENMP
    threads = 1; // sequential binary: force single-threaded regardless of flag
#endif

    if (input.empty() || query.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --input <fasta> --query <DNA> --algorithm exact|approximate|align"
                  << " [--threads N] [--mismatches K]\n";
        return 1;
    }

    std::vector<FastaRecord> records;
    try {
        records = parse_fasta(input);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing FASTA: " << e.what() << "\n";
        return 1;
    }
    if (records.empty()) {
        std::cerr << "No sequences found in " << input << "\n";
        return 1;
    }

    // Concatenate all records into one search space (a simple, documented
    // choice: cross-record boundaries are irrelevant for pattern search).
    std::string text;
    for (auto& r : records) text += r.sequence;

    size_t total_bases = text.size();

    std::cout << "=================================================\n";
    std::cout << "       PARALLEL DNA SEQUENCE ANALYSIS ENGINE\n";
    std::cout << "=================================================\n\n";
    std::cout << "Input:\n" << input << "\n\n";
    std::cout << "Dataset Size:\n" << format_number(static_cast<long long>(total_bases)) << " bases\n\n";
    std::cout << "Query:\n" << query << "\n\n";
#ifdef _OPENMP
    std::cout << "Execution Mode:\nOpenMP (" << threads << " threads)\n\n";
#else
    std::cout << "Execution Mode:\nSequential\n\n";
#endif
    std::cout << "-------------------------------------------------\n\n";

    Timer timer;

    if (algorithm == "exact") {
        auto matches = exact_match_search(text, query, threads, ExactAlgo::KMP);
        double elapsed = timer.elapsed_seconds();
        std::cout << "Matches found: " << matches.size() << "\n\n";
        size_t show = std::min<size_t>(matches.size(), 20);
        for (size_t i = 0; i < show; ++i) std::cout << "Position " << matches[i] << "\n";
        if (matches.size() > show) std::cout << "... (" << matches.size() - show << " more)\n";
        std::cout << "\n-------------------------------------------------\n\n";
        std::cout << "Execution Time: " << std::fixed << std::setprecision(4) << elapsed << " s\n";
        if (elapsed > 0)
            std::cout << "Throughput: " << std::setprecision(2)
                      << (total_bases / 1e6) / elapsed << " million bases/s\n";
    } else if (algorithm == "approximate") {
        auto matches = approximate_match_search(text, query, mismatches, threads);
        double elapsed = timer.elapsed_seconds();
        std::cout << "Maximum mismatches: " << mismatches << "\n\n";
        int exact_count = 0;
        for (auto& m : matches) if (m.mismatches == 0) ++exact_count;
        std::cout << "Exact Matches: " << exact_count << "\n";
        std::cout << "Total Matches (<=" << mismatches << " mismatches): " << matches.size() << "\n";
        std::cout << "\n-------------------------------------------------\n\n";
        std::cout << "Execution Time: " << std::fixed << std::setprecision(4) << elapsed << " s\n";
        if (elapsed > 0)
            std::cout << "Throughput: " << std::setprecision(2)
                      << (total_bases / 1e6) / elapsed << " million bases/s\n";
    } else if (algorithm == "align") {
        // For alignment mode, EVERY record in --input is aligned against
        // --query, and the best-scoring record is reported. This matches
        // the MPI/hybrid batch-alignment behavior (mpi_engine.hpp) so
        // that sequential/OpenMP timings are a fair, directly-comparable
        // baseline for batch alignment speedup -- not just a single-pair
        // alignment, which would make cross-mode speedup meaningless.
        AlignmentScoring scoring; // defaults: match +2, mismatch -1, gap -2
        scoring.match = args.get_int("--match", 2);
        scoring.mismatch = args.get_int("--mismatch", -1);
        scoring.gap = args.get_int("--gap", -2);

        int best_score = -1;
        size_t best_idx = 0;
        AlignmentResult best_result;
        for (size_t i = 0; i < records.size(); ++i) {
            auto result = smith_waterman(records[i].sequence, query, scoring, threads);
            if (result.score > best_score) {
                best_score = result.score;
                best_idx = i;
                best_result = result;
            }
        }
        double elapsed = timer.elapsed_seconds();
        std::cout << "Database: " << records.size() << " sequences\n\n";
        std::cout << "Best match: sequence #" << best_idx
                  << " (" << records[best_idx].id << ")  score = " << best_score << "\n\n";
        std::cout << best_result.aligned_a << "\n" << best_result.match_line
                  << "\n" << best_result.aligned_b << "\n";
        std::cout << "\n-------------------------------------------------\n\n";
        std::cout << "Execution Time: " << std::fixed << std::setprecision(4) << elapsed << " s\n";
    } else {
        std::cerr << "Unknown algorithm: " << algorithm << " (expected exact|approximate|align)\n";
        return 1;
    }

    std::cout << "\n-------------------------------------------------\n";
    return 0;
}

} // namespace dna
