// approximate_match.hpp
// Mutation-tolerant matching using Hamming distance over equal-length
// windows (Spec section 12), parallelized the same way as exact_match.hpp
// (Spec section 13): the genome's window-start positions are partitioned
// across threads, each independently scored. This is embarrassingly
// parallel -- no halo/overlap is needed here because each window is
// evaluated in full by a single thread (unlike exact matching, there is
// no risk of a match spanning a partition boundary since the window
// itself, not the partition, defines the compared region).
#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dna {

struct ApproxMatch {
    size_t position;
    int mismatches;
};

inline int hamming_distance(const std::string& text, size_t offset,
                             const std::string& pattern, int max_allowed) {
    int dist = 0;
    const size_t m = pattern.size();
    for (size_t i = 0; i < m; ++i) {
        if (text[offset + i] != pattern[i]) {
            if (++dist > max_allowed) return dist; // early exit, still > max
        }
    }
    return dist;
}

// Returns every window whose Hamming distance to `pattern` is
// <= max_mismatches, parallelized over `nthreads` (partitioned window
// start positions, same partitioning scheme as exact_match_search but
// without a halo, per the note above).
inline std::vector<ApproxMatch> approximate_match_search(const std::string& text,
                                                           const std::string& pattern,
                                                           int max_mismatches,
                                                           int nthreads) {
    std::vector<ApproxMatch> all_matches;
    if (pattern.empty() || text.size() < pattern.size() || nthreads < 1) {
        return all_matches;
    }

    const size_t m = pattern.size();
    const size_t num_windows = text.size() - m + 1;

    std::vector<size_t> bounds(nthreads + 1);
    for (int t = 0; t <= nthreads; ++t) {
        bounds[t] = (num_windows * static_cast<size_t>(t)) / static_cast<size_t>(nthreads);
    }

    std::vector<std::vector<ApproxMatch>> partial(nthreads);

    // NOTE: chunk size here must be small relative to nthreads (the loop
    // below iterates over PARTITIONS, not raw window positions -- there
    // are only `nthreads` iterations total). A chunk size larger than
    // the iteration count (e.g. the 4096 originally used here, intended
    // for the *window*-level granularity implied by the comment above)
    // causes OpenMP's dynamic scheduler to hand the entire loop to a
    // single thread as one chunk, leaving every other thread with zero
    // work -- silently defeating the parallelism while still paying
    // full thread-team spawn overhead. Real benchmarking on Babbage
    // caught this: pure-OpenMP approximate matching showed ~1.0x
    // "speedup" at every thread count while MPI (unaffected, since it
    // parallelizes via independent processes rather than this pragma)
    // scaled to ~22x at 32 ranks on the same workload. schedule(dynamic,
    // 1) still gives the intended data-dependent load balancing across
    // partitions (some partitions early-exit more than others), just
    // with a chunk size that can't exceed the real iteration count.
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int t = 0; t < nthreads; ++t) {
        for (size_t i = bounds[t]; i < bounds[t + 1]; ++i) {
            int d = hamming_distance(text, i, pattern, max_mismatches);
            if (d <= max_mismatches) partial[t].push_back({i, d});
        }
    }

    for (auto& v : partial) all_matches.insert(all_matches.end(), v.begin(), v.end());
    std::sort(all_matches.begin(), all_matches.end(),
              [](const ApproxMatch& a, const ApproxMatch& b) { return a.position < b.position; });
    return all_matches;
}

} // namespace dna
