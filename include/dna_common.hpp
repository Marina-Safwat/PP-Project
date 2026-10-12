// dna_common.hpp
// FASTA parsing, sequence validation, synthetic dataset generation, and
// small timing/reporting utilities shared by every executable in the
// project (sequential, OpenMP, MPI, hybrid).
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <chrono>
#include <cctype>
#include <algorithm>

namespace dna {

// ---------------------------------------------------------------------
// FASTA record + parser (Spec section 8)
// ---------------------------------------------------------------------
struct FastaRecord {
    std::string id;
    std::string sequence; // upper-case, whitespace-stripped
};

inline bool is_valid_base(char c) {
    switch (c) {
        case 'A': case 'C': case 'G': case 'T':
        case 'N': // N = unknown base, common in real genomic data
            return true;
        default:
            return false;
    }
}

// Parses a FASTA file into a vector of records. Multiline sequences are
// concatenated; whitespace is stripped; bases are upper-cased. Throws
// std::runtime_error on missing file or an invalid symbol.
inline std::vector<FastaRecord> parse_fasta(const std::string& path, bool validate = true) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open FASTA file: " + path);
    }

    std::vector<FastaRecord> records;
    std::string line;
    FastaRecord current;
    bool have_current = false;

    auto flush = [&]() {
        if (have_current) {
            records.push_back(current);
        }
    };

    while (std::getline(in, line)) {
        // strip trailing \r (Windows line endings) and surrounding whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                  line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) continue;

        if (line[0] == '>') {
            flush();
            current = FastaRecord{};
            current.id = line.substr(1);
            have_current = true;
        } else {
            if (!have_current) {
                throw std::runtime_error("FASTA file does not start with a '>' header: " + path);
            }
            for (char c : line) {
                if (std::isspace(static_cast<unsigned char>(c))) continue;
                char up = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                if (validate && !is_valid_base(up)) {
                    throw std::runtime_error(
                        std::string("Invalid nucleotide symbol '") + c +
                        "' in record " + current.id);
                }
                current.sequence.push_back(up);
            }
        }
    }
    flush();
    return records;
}

// ---------------------------------------------------------------------
// Synthetic dataset generation with known-match injection
// (Spec section 20A) -- used mainly by scripts/generate_dataset.py, but
// provided here too so C++ test code can build small in-memory fixtures
// without shelling out to Python.
// ---------------------------------------------------------------------
inline std::string random_dna_sequence(size_t length, unsigned seed) {
    static const char bases[4] = {'A', 'C', 'G', 'T'};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 3);
    std::string s;
    s.resize(length);
    for (size_t i = 0; i < length; ++i) s[i] = bases[dist(rng)];
    return s;
}

// Inserts `pattern` at `count` random non-overlapping positions in `seq`
// (in place) and returns the positions used, so correctness can be
// verified automatically against a ground truth.
inline std::vector<size_t> inject_known_matches(std::string& seq,
                                                  const std::string& pattern,
                                                  size_t count,
                                                  unsigned seed) {
    std::vector<size_t> positions;
    if (pattern.empty() || pattern.size() > seq.size()) return positions;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, seq.size() - pattern.size());
    for (size_t i = 0; i < count; ++i) {
        size_t pos = dist(rng);
        seq.replace(pos, pattern.size(), pattern);
        positions.push_back(pos);
    }
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    return positions;
}

// ---------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------
class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    void reset() { start_ = std::chrono::steady_clock::now(); }
    double elapsed_seconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------------
// Simple CLI argument parsing helper (shared across all four mains)
// ---------------------------------------------------------------------
class ArgParser {
public:
    ArgParser(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) args_.push_back(argv[i]);
    }

    std::string get(const std::string& flag, const std::string& def = "") const {
        for (size_t i = 0; i < args_.size(); ++i) {
            if (args_[i] == flag && i + 1 < args_.size()) return args_[i + 1];
        }
        return def;
    }

    int get_int(const std::string& flag, int def) const {
        std::string v = get(flag);
        if (v.empty()) return def;
        return std::stoi(v);
    }

    bool has(const std::string& flag) const {
        return std::find(args_.begin(), args_.end(), flag) != args_.end();
    }

private:
    std::vector<std::string> args_;
};

inline std::string format_number(long long n) {
    std::string s = std::to_string(n);
    std::string out;
    int cnt = 0;
    for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
        out.push_back(s[i]);
        if (++cnt == 3 && i != 0) { out.push_back(','); cnt = 0; }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

} // namespace dna
