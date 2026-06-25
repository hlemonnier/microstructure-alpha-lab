#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Row {
    std::string event_type;
    std::string side;
    double price = 0.0;
    double size = 0.0;
    long long exchange_timestamp = 0;
    long long sequence = -1;
    long long update_id = -1;
    std::string venue;
};

static std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> cells;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
        cells.push_back(cell);
    }
    return cells;
}

static long long to_int(const std::string &value, long long missing = -1) {
    if (value.empty()) {
        return missing;
    }
    return std::atoll(value.c_str());
}

static double to_double(const std::string &value) {
    if (value.empty()) {
        return 0.0;
    }
    return std::atof(value.c_str());
}

static std::string get_cell(
    const std::vector<std::string> &cells,
    const std::unordered_map<std::string, size_t> &columns,
    const std::string &name
) {
    auto it = columns.find(name);
    if (it == columns.end() || it->second >= cells.size()) {
        return "";
    }
    return cells[it->second];
}

static Row parse_row(const std::vector<std::string> &cells, const std::unordered_map<std::string, size_t> &columns) {
    Row row;
    row.event_type = get_cell(cells, columns, "event_type");
    row.side = get_cell(cells, columns, "side");
    row.price = to_double(get_cell(cells, columns, "price"));
    row.size = to_double(get_cell(cells, columns, "size"));
    row.exchange_timestamp = to_int(get_cell(cells, columns, "exchange_timestamp"), 0);
    row.sequence = to_int(get_cell(cells, columns, "sequence"));
    row.update_id = to_int(get_cell(cells, columns, "update_id"));
    row.venue = get_cell(cells, columns, "venue");
    return row;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: l2_replay <normalized_l2_csv> [max_sequence_step]\n";
        return 2;
    }
    const std::string path = argv[1];
    const long long max_step = argc == 3 ? std::atoll(argv[2]) : 1;
    if (max_step <= 0) {
        std::cerr << "max_sequence_step must be positive\n";
        return 2;
    }

    std::ifstream input(path);
    if (!input) {
        std::cerr << "cannot open " << path << "\n";
        return 1;
    }

    std::string header_line;
    if (!std::getline(input, header_line)) {
        std::cerr << "empty csv\n";
        return 1;
    }
    const auto headers = split_csv_line(header_line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t i = 0; i < headers.size(); ++i) {
        columns[headers[i]] = i;
    }

    std::map<double, double, std::greater<double>> bids;
    std::map<double, double> asks;
    long long last_sequence = -1;
    long long last_update_id = -1;
    bool needs_resnapshot = false;
    std::string active_snapshot_key;
    std::unordered_set<std::string> seen_delta_keys;

    std::cout << "row_index,sequence_gap,reset,crossed,best_bid,best_ask,needs_resnapshot\n";
    std::string line;
    long long row_index = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        ++row_index;
        Row row = parse_row(split_csv_line(line), columns);
        if ((row.event_type != "snapshot" && row.event_type != "delta") ||
            (row.side != "bid" && row.side != "ask") ||
            row.price <= 0.0 || row.size < 0.0) {
            std::cerr << "invalid row at " << row_index << "\n";
            return 1;
        }

        bool sequence_gap = false;
        const bool update_id_is_primary = row.venue == "bybit" && row.update_id >= 0;
        if (!update_id_is_primary && row.sequence >= 0 && last_sequence >= 0) {
            sequence_gap = row.sequence < last_sequence || row.sequence > last_sequence + max_step;
        }
        if (row.update_id >= 0 && last_update_id >= 0) {
            sequence_gap = sequence_gap || row.update_id < last_update_id || row.update_id > last_update_id + max_step;
        }
        const std::string delta_key =
            std::to_string(row.exchange_timestamp) + "|" + std::to_string(row.sequence) + "|" + std::to_string(row.update_id) + "|" + row.side + "|" + std::to_string(row.price);
        const bool duplicate_update = row.event_type == "delta" && seen_delta_keys.find(delta_key) != seen_delta_keys.end();
        if (row.event_type == "delta") {
            seen_delta_keys.insert(delta_key);
        }

        bool reset = false;
        if (row.event_type == "snapshot") {
            const std::string key = std::to_string(row.exchange_timestamp) + "|" + std::to_string(row.sequence) + "|" + std::to_string(row.update_id);
            if (key != active_snapshot_key) {
                bids.clear();
                asks.clear();
                seen_delta_keys.clear();
                active_snapshot_key = key;
                reset = true;
            }
            needs_resnapshot = false;
        } else {
            active_snapshot_key.clear();
            if (sequence_gap || duplicate_update) {
                bids.clear();
                asks.clear();
                reset = true;
                needs_resnapshot = true;
            }
        }

        if (row.side == "bid") {
            if (row.size <= 0.0) {
                bids.erase(row.price);
            } else {
                bids[row.price] = row.size;
            }
        } else {
            if (row.size <= 0.0) {
                asks.erase(row.price);
            } else {
                asks[row.price] = row.size;
            }
        }
        if (row.sequence >= 0) {
            last_sequence = std::max(last_sequence, row.sequence);
        }
        if (row.update_id >= 0) {
            last_update_id = std::max(last_update_id, row.update_id);
        }

        const bool has_bid = !bids.empty();
        const bool has_ask = !asks.empty();
        const bool crossed = has_bid && has_ask && bids.begin()->first >= asks.begin()->first;
        std::cout << row_index << ","
                  << (sequence_gap ? 1 : 0) << ","
                  << (reset ? 1 : 0) << ","
                  << (crossed ? 1 : 0) << ",";
        if (has_bid) {
            std::cout << bids.begin()->first;
        }
        std::cout << ",";
        if (has_ask) {
            std::cout << asks.begin()->first;
        }
        std::cout << "," << (needs_resnapshot ? 1 : 0) << "\n";
    }
    return 0;
}
