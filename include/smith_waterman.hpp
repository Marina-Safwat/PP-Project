// smith_waterman.hpp
// Smith-Waterman local sequence alignment (Spec section 14) with an
// OpenMP wavefront/anti-diagonal parallelization (Spec sections 15-16).
//
// H(i,j) depends on H(i-1,j-1), H(i-1,j), H(i,j-1): cells on diagonal
// d = i+j all depend only on cells on diagonals d-1 and d-2, so every
// cell *within* a diagonal is independent and can be computed
// concurrently. We iterate diagonal by diagonal; within a diagonal an
// OpenMP `parallel for` fans the (i,j) pairs on that diagonal out across
// threads, with an implicit barrier at the end of each diagonal's loop
// enforcing the wavefront dependency before the next diagonal starts.
//
// The full (M+1) x (N+1) score matrix is kept (Spec section 34, memory
// section) so a traceback can reconstruct the alignment; a
// memory-optimized score-only variant is a documented optional
// extension (Spec section 35), not implemented here to keep the core
// system's scope bounded per Spec section 37.
#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dna {

struct AlignmentScoring {
    int match = 2;
    int mismatch = -1;
    int gap = -2;
};

struct AlignmentResult {
    int score = 0;
    size_t end_row = 0, end_col = 0;     // bottom-right corner of best local alignment
    size_t start_row = 0, start_col = 0; // traced-back start
    std::string aligned_a, aligned_b, match_line;
};

// Direction codes for traceback.
enum class Dir : uint8_t { NONE = 0, DIAG = 1, UP = 2, LEFT = 3 };

// Computes the Smith-Waterman score matrix using wavefront parallelism
// and returns the best-scoring alignment (with traceback). `nthreads`
// controls the OpenMP team size used per diagonal.
inline AlignmentResult smith_waterman(const std::string& a, const std::string& b,
                                       const AlignmentScoring& scoring,
                                       int nthreads) {
    const size_t M = a.size();
    const size_t N = b.size();

    // H[i][j], flattened row-major, size (M+1) x (N+1)
    std::vector<int> H(static_cast<size_t>(M + 1) * (N + 1), 0);
    std::vector<Dir> T(static_cast<size_t>(M + 1) * (N + 1), Dir::NONE);
    auto idx = [&](size_t i, size_t j) { return i * (N + 1) + j; };

    int best_score = 0;
    size_t best_i = 0, best_j = 0;

    // Diagonals d = i + j range from 2 (first interior cell, i=1,j=1) to M+N.
    // Row/col 0 are boundary (already zero-initialized, per SW definition).
    for (size_t d = 2; d <= M + N; ++d) {
        // On diagonal d, valid i ranges over max(1, d-N) .. min(M, d-1)
        size_t i_lo = (d > N) ? (d - N) : 1;
        size_t i_hi = std::min(M, d - 1);
        if (i_lo > i_hi) continue;

        size_t count = i_hi - i_lo + 1;
        int local_best_score = 0;

        #pragma omp parallel for num_threads(nthreads) schedule(static) \
            reduction(max:local_best_score)
        for (size_t k = 0; k < count; ++k) {
            size_t i = i_lo + k;
            size_t j = d - i;

            int diag = H[idx(i - 1, j - 1)] + (a[i - 1] == b[j - 1] ? scoring.match : scoring.mismatch);
            int up   = H[idx(i - 1, j)] + scoring.gap;
            int left = H[idx(i, j - 1)] + scoring.gap;

            int best = 0;
            Dir dir = Dir::NONE;
            if (diag > best) { best = diag; dir = Dir::DIAG; }
            if (up   > best) { best = up;   dir = Dir::UP;   }
            if (left > best) { best = left; dir = Dir::LEFT; }

            H[idx(i, j)] = best;
            T[idx(i, j)] = dir;

            if (best > local_best_score) {
                local_best_score = best;
                // note: not thread-safe to also reduce i/j positions via OpenMP
                // 'max' reduction (it only tracks the scalar), so recovered below
            }
        }

        // Recover the (i,j) achieving local_best_score for this diagonal
        // with a small sequential scan (cheap: at most one diagonal's
        // worth of cells, and only runs when this diagonal beats the
        // global best so far).
        if (local_best_score > best_score) {
            for (size_t k = 0; k < count; ++k) {
                size_t i = i_lo + k;
                size_t j = d - i;
                if (H[idx(i, j)] == local_best_score) {
                    best_score = local_best_score;
                    best_i = i;
                    best_j = j;
                    break;
                }
            }
        }
    }

    AlignmentResult result;
    result.score = best_score;
    result.end_row = best_i;
    result.end_col = best_j;

    // Traceback from (best_i, best_j) until we hit a 0-score cell or NONE.
    size_t i = best_i, j = best_j;
    std::string ra, rb, rm;
    while (i > 0 && j > 0 && H[idx(i, j)] > 0) {
        Dir dir = T[idx(i, j)];
        if (dir == Dir::DIAG) {
            ra.push_back(a[i - 1]);
            rb.push_back(b[j - 1]);
            rm.push_back(a[i - 1] == b[j - 1] ? '|' : ' ');
            --i; --j;
        } else if (dir == Dir::UP) {
            ra.push_back(a[i - 1]);
            rb.push_back('-');
            rm.push_back(' ');
            --i;
        } else if (dir == Dir::LEFT) {
            ra.push_back('-');
            rb.push_back(b[j - 1]);
            rm.push_back(' ');
            --j;
        } else {
            break;
        }
    }
    result.start_row = i;
    result.start_col = j;
    std::reverse(ra.begin(), ra.end());
    std::reverse(rb.begin(), rb.end());
    std::reverse(rm.begin(), rm.end());
    result.aligned_a = ra;
    result.aligned_b = rb;
    result.match_line = rm;
    return result;
}

} // namespace dna
