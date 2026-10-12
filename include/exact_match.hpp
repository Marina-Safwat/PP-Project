// exact_match.hpp
// Exact DNA pattern matching (Spec section 10) with an OpenMP-parallel
// partitioned implementation (Spec section 11).
//
// Design: the genome is logically split into `nthreads` contiguous
// partitions. Each partition is extended by (pattern_length - 1) bases
// on its right edge ("halo") so that matches straddling a partition
// boundary are not missed. Each thread reports only matches whose start
// position falls inside its *own* (non-overlapping) partition range, so
// no match is double-counted.
//
// Compiled without -fopenmp, the #pragma omp lines are no-ops and the
// function runs as a plain sequential scan -- this is how the sequential
// and OpenMP binaries share exactly the same source.
#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dna {

// Naive O(n*m) sliding-window search over [begin, end) of text, returning
// absolute match start positions.
inline void naive_scan(const std::string& text, const std::string& pattern,
                        size_t begin, size_t end, std::vector<size_t>& out) {
    const size_t m = pattern.size();
    if (m == 0 || text.size() < m) return;
    if (end > text.size() - m + 1) end = text.size() - m + 1;
    for (size_t i = begin; i < end; ++i) {
        if (text.compare(i, m, pattern) == 0) {
            out.push_back(i);
        }
    }
}

// Builds the KMP failure/partial-match table for `pattern`.
inline std::vector<int> kmp_failure_table(const std::string& pattern) {
    std::vector<int> fail(pattern.size(), 0);
    int k = 0;
    for (size_t i = 1; i < pattern.size(); ++i) {
        while (k > 0 && pattern[i] != pattern[k]) k = fail[k - 1];
        if (pattern[i] == pattern[k]) ++k;
        fail[i] = k;
    }
    return fail;
}

// KMP search restricted to the window [begin, end_inclusive_search_start).
// `end` bounds where a match may *start*; scanning of characters may run
// past `end` up to pattern.size()-1 extra characters (that's the halo).
inline void kmp_scan(const std::string& text, const std::string& pattern,
                      const std::vector<int>& fail,
                      size_t begin, size_t end, std::vector<size_t>& out) {
    const size_t m = pattern.size();
    if (m == 0) return;
    size_t scan_limit = std::min(text.size(), end + m - 1);
    int k = 0;
    for (size_t i = begin; i < scan_limit; ++i) {
        while (k > 0 && text[i] != pattern[k]) k = fail[k - 1];
        if (text[i] == pattern[k]) ++k;
        if (k == static_cast<int>(m)) {
            size_t start = i - m + 1;
            if (start >= begin && start < end) out.push_back(start);
            k = fail[k - 1];
        }
    }
}

enum class ExactAlgo { NAIVE, KMP };

// Partitioned exact match. `nthreads` controls both the OpenMP team size
// (when compiled with OpenMP) AND the number of logical partitions, so
// the partitioning strategy is identical whether run sequentially or in
// parallel -- only the number of workers touching it differs.
inline std::vector<size_t> exact_match_search(const std::string& text,
                                               const std::string& pattern,
                                               int nthreads,
                                               ExactAlgo algo = ExactAlgo::KMP) {
    std::vector<size_t> all_matches;
    if (pattern.empty() || text.size() < pattern.size() || nthreads < 1) {
        return all_matches;
    }

    const size_t n = text.size();
    const size_t m = pattern.size();
    const size_t num_start_positions = n - m + 1;
    const size_t halo = m - 1; // Spec section 11: overlap = pattern_length - 1

    // partition boundaries over *start positions* [0, num_start_positions)
    std::vector<size_t> bounds(nthreads + 1);
    for (int t = 0; t <= nthreads; ++t) {
        bounds[t] = (num_start_positions * static_cast<size_t>(t)) / static_cast<size_t>(nthreads);
    }

    std::vector<std::vector<size_t>> partial(nthreads);
    std::vector<int> fail;
    if (algo == ExactAlgo::KMP) fail = kmp_failure_table(pattern);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int t = 0; t < nthreads; ++t) {
        size_t begin = bounds[t];
        size_t end = bounds[t + 1]; // exclusive: valid match-start range owned by this thread
        if (begin == end) continue;
        if (algo == ExactAlgo::NAIVE) {
            // extend scan window by halo so cross-boundary matches are seen,
            // but naive_scan already only reports starts in [begin,end)
            size_t scan_end = std::min(end + halo, num_start_positions);
            // clamp: naive_scan takes [begin,end) as *start* range directly,
            // so just widen its own end and let it filter via caller below
            std::vector<size_t> local;
            naive_scan(text, pattern, begin, scan_end, local);
            for (size_t p : local) if (p >= begin && p < end) partial[t].push_back(p);
        } else {
            kmp_scan(text, pattern, fail, begin, end, partial[t]);
        }
    }

    for (auto& v : partial) {
        all_matches.insert(all_matches.end(), v.begin(), v.end());
    }
    std::sort(all_matches.begin(), all_matches.end());
    return all_matches;
}

} // namespace dna
