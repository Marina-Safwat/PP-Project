// test_all.cpp
// Automated correctness tests (Spec section 21). Run via `make test`.
// Compiled WITH -fopenmp so these tests exercise the real parallel code
// paths (varying --threads-equivalent nthreads), not just the inert
// sequential fallback.
//
// Strategy: for each algorithm, compute a trusted result with a brute
// force / independent method, then assert that exact_match_search /
// approximate_match_search / smith_waterman agree across nthreads =
// 1, 2, 4, 8 -- this is exactly the invariant Section 21 requires
// (Sequential == OpenMP == MPI == Hybrid), tested at the algorithm
// level here; test_mpi_consistency.sh (scripts/) covers the same
// invariant end-to-end across the actual MPI/hybrid binaries.
#include <cassert>
#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "dna_common.hpp"
#include "exact_match.hpp"
#include "approximate_match.hpp"
#include "smith_waterman.hpp"

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
    ++g_checks; \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_failures; \
    } \
} while (0)

// Independent brute-force reference for exact matching.
static std::vector<size_t> brute_force_exact(const std::string& text, const std::string& pattern) {
    std::vector<size_t> out;
    if (pattern.empty() || text.size() < pattern.size()) return out;
    for (size_t i = 0; i + pattern.size() <= text.size(); ++i) {
        if (text.compare(i, pattern.size(), pattern) == 0) out.push_back(i);
    }
    return out;
}

// Independent brute-force reference for Hamming-distance approximate matching.
static std::vector<dna::ApproxMatch> brute_force_approx(const std::string& text,
                                                          const std::string& pattern,
                                                          int max_mismatches) {
    std::vector<dna::ApproxMatch> out;
    if (pattern.empty() || text.size() < pattern.size()) return out;
    for (size_t i = 0; i + pattern.size() <= text.size(); ++i) {
        int d = 0;
        for (size_t k = 0; k < pattern.size(); ++k) if (text[i + k] != pattern[k]) ++d;
        if (d <= max_mismatches) out.push_back({i, d});
    }
    return out;
}

static void test_exact_matching() {
    std::cout << "[test] exact matching ... ";
    int __before = g_failures;
    std::string genome = dna::random_dna_sequence(50000, /*seed=*/42);
    std::string pattern = "ACGTAGCTAG";
    dna::inject_known_matches(genome, pattern, 30, /*seed=*/7);

    auto expected = brute_force_exact(genome, pattern);
    CHECK(expected.size() >= 30, "expected at least the injected matches");

    for (int nt : {1, 2, 3, 4, 8, 16}) {
        for (auto algo : {dna::ExactAlgo::NAIVE, dna::ExactAlgo::KMP}) {
            auto got = dna::exact_match_search(genome, pattern, nt, algo);
            CHECK(got == expected, "exact match mismatch at nthreads=" + std::to_string(nt) +
                                     " algo=" + (algo == dna::ExactAlgo::NAIVE ? "naive" : "kmp"));
        }
    }
    // Edge cases
    CHECK(dna::exact_match_search("", "ACGT", 4).empty(), "empty text -> no matches");
    CHECK(dna::exact_match_search("ACGT", "", 4).empty(), "empty pattern -> no matches");
    CHECK(dna::exact_match_search("ACGT", "ACGTACGT", 4).empty(), "pattern longer than text -> no matches");
    std::cout << (g_failures == __before ? "ok" : "FAILED") << "\n";
}

static void test_boundary_overlap() {
    std::cout << "[test] partition boundary overlap ... ";
    int __before = g_failures;
    // Construct a genome where matches deliberately straddle likely
    // partition boundaries for small thread counts, to specifically
    // exercise the halo logic from Spec section 11.
    std::string genome = std::string(100, 'A') + "TAGCTA" + std::string(100, 'C') +
                          "TAGCTA" + std::string(100, 'G');
    std::string pattern = "TAGCTA";
    auto expected = brute_force_exact(genome, pattern);
    for (int nt : {2, 3, 4, 5, 7}) {
        auto got = dna::exact_match_search(genome, pattern, nt);
        CHECK(got == expected, "boundary overlap failed at nthreads=" + std::to_string(nt));
    }
    std::cout << (g_failures == __before ? "ok" : "FAILED") << "\n";
}

static void test_approximate_matching() {
    std::cout << "[test] approximate matching ... ";
    int __before = g_failures;
    std::string genome = dna::random_dna_sequence(20000, 99);
    std::string pattern = "ACGTTGCA";
    for (int max_mm : {0, 1, 2, 3}) {
        auto expected = brute_force_approx(genome, pattern, max_mm);
        for (int nt : {1, 2, 4, 8}) {
            auto got = dna::approximate_match_search(genome, pattern, max_mm, nt);
            CHECK(got.size() == expected.size(),
                  "approx match count mismatch at mm=" + std::to_string(max_mm) +
                  " nthreads=" + std::to_string(nt) + " (" + std::to_string(got.size()) +
                  " vs " + std::to_string(expected.size()) + ")");
            for (size_t i = 0; i < got.size() && i < expected.size(); ++i) {
                CHECK(got[i].position == expected[i].position && got[i].mismatches == expected[i].mismatches,
                      "approx match content mismatch");
            }
        }
    }
    std::cout << (g_failures == __before ? "ok" : "FAILED") << "\n";
}

static void test_smith_waterman() {
    std::cout << "[test] Smith-Waterman alignment ... ";
    int __before = g_failures;
    // Classic textbook example (Wikipedia's SW worked example uses
    // similar sequences): verify the parallel wavefront implementation
    // agrees with itself across thread counts, and that it correctly
    // finds a known perfect local alignment.
    std::string a = "ACACACTA";
    std::string b = "AGCACACA";
    dna::AlignmentScoring scoring; // match=2, mismatch=-1, gap=-2

    int ref_score = -1;
    for (int nt : {1, 2, 4, 8}) {
        auto result = dna::smith_waterman(a, b, scoring, nt);
        if (ref_score == -1) ref_score = result.score;
        CHECK(result.score == ref_score, "SW score differs across nthreads=" + std::to_string(nt));
        // alignment must not exceed either sequence's length
        CHECK(result.aligned_a.size() == result.aligned_b.size(), "aligned strings must be equal length");
    }
    CHECK(ref_score > 0, "expected a positive-scoring local alignment for similar sequences");

    // Self-alignment must score length * match_score exactly.
    std::string s = "ACGTACGTACGT";
    auto self = dna::smith_waterman(s, s, scoring, 4);
    CHECK(self.score == static_cast<int>(s.size()) * scoring.match,
          "self-alignment should score length*match exactly");

    // Completely dissimilar sequences under harsh penalties should score 0
    // (no local alignment is worth taking).
    dna::AlignmentScoring harsh; harsh.match = 1; harsh.mismatch = -10; harsh.gap = -10;
    auto none = dna::smith_waterman("AAAA", "TTTT", harsh, 2);
    CHECK(none.score == 0, "fully dissimilar sequences under harsh penalties should score 0");

    std::cout << (g_failures == __before ? "ok" : "FAILED") << "\n";
}

static void test_fasta_parser() {
    std::cout << "[test] FASTA parser ... ";
    int __before = g_failures;
    std::string path = "/tmp/dna_engine_test.fasta";
    {
        std::ofstream out(path);
        out << ">seq1 description here\n";
        out << "ACGT\nACGT\n";
        out << ">seq2\n";
        out << "TTTT\n";
    }
    auto records = dna::parse_fasta(path);
    CHECK(records.size() == 2, "expected 2 records");
    CHECK(records[0].sequence == "ACGTACGT", "multiline sequence should be concatenated");
    CHECK(records[1].sequence == "TTTT", "second record sequence mismatch");
    CHECK(records[0].id.substr(0, 4) == "seq1", "record id parsed");

    bool threw = false;
    {
        std::ofstream out("/tmp/dna_engine_test_bad.fasta");
        out << ">bad\nACGTX\n"; // X is invalid
    } // <-- out must be closed/flushed before parse_fasta reads the file
    try {
        dna::parse_fasta("/tmp/dna_engine_test_bad.fasta");
    } catch (const std::exception&) { threw = true; }
    CHECK(threw, "invalid nucleotide symbol should throw");

    std::cout << (g_failures == __before ? "ok" : "FAILED") << "\n";
}

int main() {
    test_fasta_parser();
    test_exact_matching();
    test_boundary_overlap();
    test_approximate_matching();
    test_smith_waterman();

    std::cout << "\n" << g_checks << " checks run, " << g_failures << " failed.\n";
    return g_failures == 0 ? 0 : 1;
}
