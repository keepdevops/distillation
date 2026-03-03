/**
 * Training watchdog in C++ — deterministic rules, no agent.
 * Watches trainer_state.json for loss plateau; writes watchdog_suggestions.json.
 * LaunchAgent-ready. No LibTorch dependency.
 *
 * Build: cmake -B build && cmake --build build
 * Run:   ./build/watchdog ./distilled-minillm --interval 60
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

#define WATCHDOG_LOG(level, msg) \
  do { std::cerr << "[watchdog][" level "] " << msg << std::endl; } while (0)

// Default rules (overridable via --config)
struct Rules {
  int plateau_window = 3;
  double plateau_max_delta = 0.001;
  int plateau_min_points = 5;
  double plateau_lr_scale = 0.8;
  bool thermal_enabled = false;
  double thermal_pause_if_over = 95.0;
  bool validator_backup = true;
  double validator_max_lr_scale = 0.5;

  void load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    try {
      json j = json::parse(f);
      if (j.contains("plateau")) {
        if (j["plateau"].contains("window")) plateau_window = j["plateau"]["window"];
        if (j["plateau"].contains("max_delta")) plateau_max_delta = j["plateau"]["max_delta"];
        if (j["plateau"].contains("min_points")) plateau_min_points = j["plateau"]["min_points"];
        if (j["plateau"].contains("lr_scale")) plateau_lr_scale = j["plateau"]["lr_scale"];
      }
      if (j.contains("thermal")) {
        if (j["thermal"].contains("enabled")) thermal_enabled = j["thermal"]["enabled"];
        if (j["thermal"].contains("pause_if_over")) thermal_pause_if_over = j["thermal"]["pause_if_over"];
      }
      if (j.contains("validator")) {
        if (j["validator"].contains("backup_before_write")) validator_backup = j["validator"]["backup_before_write"];
        if (j["validator"].contains("max_lr_scale")) validator_max_lr_scale = j["validator"]["max_lr_scale"];
      }
      WATCHDOG_LOG("INFO", "Loaded config from " << path);
    } catch (const json::exception& e) {
      WATCHDOG_LOG("WARN", "Config parse failed: " << e.what() << " — using defaults");
    } catch (const std::exception& e) {
      WATCHDOG_LOG("WARN", "Config load failed: " << e.what() << " — using defaults");
    }
  }
};

static std::vector<double> getRecentLosses(const json& state, int n) {
  std::vector<double> losses;
  if (!state.contains("log_history")) return losses;
  const auto& log = state["log_history"];
  for (auto it = log.rbegin(); it != log.rend() && static_cast<int>(losses.size()) < n; ++it) {
    if (it->contains("loss")) {
      losses.push_back((*it)["loss"].get<double>());
    }
  }
  std::reverse(losses.begin(), losses.end());
  return losses;
}

static int getLastStep(const json& state) {
  if (!state.contains("log_history")) return 0;
  const auto& log = state["log_history"];
  for (auto it = log.rbegin(); it != log.rend(); ++it) {
    if (it->contains("step")) return (*it)["step"].get<int>();
  }
  return 0;
}

static bool detectPlateau(const std::vector<double>& losses, const Rules& r) {
  const int w = r.plateau_window;
  const double maxD = r.plateau_max_delta;
  const int minP = r.plateau_min_points;
  if (static_cast<int>(losses.size()) < minP || static_cast<int>(losses.size()) < w + 1)
    return false;
  std::vector<double> deltas;
  const size_t start = losses.size() - (w + 1);
  for (size_t i = start; i + 1 < losses.size(); ++i) {
    deltas.push_back(std::fabs(losses[i + 1] - losses[i]));
  }
  return std::all_of(deltas.begin(), deltas.end(), [maxD](double d) { return d < maxD; });
}

static json loadState(const fs::path& outputDir) {
  fs::path p = outputDir / "trainer_state.json";
  if (!fs::exists(p)) return json();
  std::ifstream f(p);
  if (!f) return json();
  try {
    json j = json::parse(f);
    if (!j.contains("log_history") || !j["log_history"].is_array()) {
      WATCHDOG_LOG("WARN", "trainer_state.json missing/invalid log_history: " << p);
      return json();
    }
    return j;
  } catch (const json::exception& e) {
    WATCHDOG_LOG("WARN", "trainer_state.json parse error: " << e.what());
    return json();
  } catch (const std::exception& e) {
    WATCHDOG_LOG("WARN", "trainer_state.json read error: " << e.what());
    return json();
  }
}

static json loadSuggestions(const fs::path& outputDir) {
  fs::path p = outputDir / "watchdog_suggestions.json";
  if (!fs::exists(p)) return json::object();
  std::ifstream f(p);
  if (!f) return json::object();
  try {
    return json::parse(f);
  } catch (const json::exception& e) {
    WATCHDOG_LOG("WARN", "watchdog_suggestions.json parse error: " << e.what() << " — using empty");
    return json::object();
  } catch (const std::exception& e) {
    WATCHDOG_LOG("WARN", "watchdog_suggestions.json read error: " << e.what());
    return json::object();
  }
}

static void writeSuggestions(const fs::path& outputDir, const json& suggestions,
                             const Rules& rules, bool dryRun) {
  if (dryRun) return;
  fs::path target = outputDir / "watchdog_suggestions.json";
  fs::path backup = outputDir / "watchdog_suggestions.json.bak";
  fs::path tmp = outputDir / "watchdog_suggestions.json.tmp";

  json out = suggestions;
  if (out.contains("next_lr_scale")) {
    double v = out["next_lr_scale"].get<double>();
    out["next_lr_scale"] = std::max(v, rules.validator_max_lr_scale);
  }

  try {
    if (rules.validator_backup && fs::exists(target)) {
      fs::copy(target, backup, fs::copy_options::overwrite_existing);
    }
    std::ofstream tf(tmp);
    if (!tf) {
      WATCHDOG_LOG("ERROR", "Cannot open temp file: " << tmp);
      return;
    }
    tf << out.dump(2);
    tf.close();
    if (!tf.good()) {
      WATCHDOG_LOG("ERROR", "Write failed for " << tmp);
      if (fs::exists(tmp)) fs::remove(tmp);
      return;
    }
    fs::rename(tmp, target);
    WATCHDOG_LOG("INFO", "Wrote watchdog_suggestions.json");
  } catch (const std::exception& e) {
    WATCHDOG_LOG("ERROR", "writeSuggestions failed: " << e.what());
    if (fs::exists(tmp)) {
      try { fs::remove(tmp); } catch (...) {}
    }
  }
}

static void runTick(const fs::path& outputDir, Rules& rules, bool dryRun) {
  json state = loadState(outputDir);
  if (state.is_null() || state.empty()) return;
  if (!state.contains("log_history") || state["log_history"].empty()) return;

  std::vector<double> losses = getRecentLosses(state, 20);
  if (static_cast<int>(losses.size()) < rules.plateau_min_points) return;

  int step = getLastStep(state);
  json current = loadSuggestions(outputDir);
  bool updated = false;

  if (detectPlateau(losses, rules)) {
    double prevScale = current.value("next_lr_scale", 1.0);
    double newScale = prevScale * rules.plateau_lr_scale;
    if (newScale != prevScale) {
      current["action"] = "lr_scale";
      current["next_lr_scale"] = newScale;
      current["reason"] = "plateau";
      current["at_step"] = step;
      json lastLosses = json::array();
      const int n = std::min(5, static_cast<int>(losses.size()));
      for (int i = static_cast<int>(losses.size()) - n; i < static_cast<int>(losses.size()); ++i)
        lastLosses.push_back(losses[i]);
      current["last_losses"] = lastLosses;
      updated = true;
    }
  }

  if (updated && !dryRun) {
    writeSuggestions(outputDir, current, rules, dryRun);
    std::cout << "[watchdog] step=" << step << " action=" << current.value("action", "")
              << " reason=" << current.value("reason", "") << std::endl;
  }
}

static void printUsage(const char* prog) {
  std::cerr << "Usage: " << prog << " <output_dir> [--interval SEC] [--config PATH]\n"
            << "       [--once] [--dry-run]\n"
            << "  output_dir  Directory containing trainer_state.json\n"
            << "  --interval  Poll interval in seconds (default: 60)\n"
            << "  --config    JSON rules file (optional)\n"
            << "  --once      Run one tick and exit\n"
            << "  --dry-run   Log only, do not write\n";
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return 1;
  }

  std::string outputDir = argv[1];
  int interval = 60;
  std::string configPath;
  bool once = false;
  bool dryRun = false;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--interval" && i + 1 < argc) {
      try {
        interval = std::stoi(argv[++i]);
        if (interval <= 0) {
          WATCHDOG_LOG("WARN", "interval must be > 0, using 60");
          interval = 60;
        }
      } catch (const std::exception&) {
        WATCHDOG_LOG("WARN", "Invalid --interval, using 60");
        ++i;
      }
    } else if (arg == "--config" && i + 1 < argc) {
      configPath = argv[++i];
    } else if (arg == "--once") {
      once = true;
    } else if (arg == "--dry-run") {
      dryRun = true;
    }
  }

  fs::path outPath = fs::absolute(outputDir);
  if (!fs::exists(outPath)) {
    try {
      fs::create_directories(outPath);
    } catch (const std::filesystem::filesystem_error& e) {
      WATCHDOG_LOG("ERROR", "output_dir create failed: " << outPath << " — " << e.what());
      std::cerr << "  Use an existing path (e.g. ./distilled-minillm)" << std::endl;
      return 1;
    } catch (const std::exception& e) {
      WATCHDOG_LOG("ERROR", "output_dir create failed: " << outPath << " — " << e.what());
      std::cerr << "  Use an existing path (e.g. ./distilled-minillm)" << std::endl;
      return 1;
    }
  }

  Rules rules;
  if (!configPath.empty() && fs::exists(configPath)) {
    rules.load(configPath);
  }

  if (once) {
    runTick(outPath, rules, dryRun);
    return 0;
  }

  std::cout << "[watchdog] Monitoring " << outPath << " every " << interval << "s" << std::endl;
  while (true) {
    runTick(outPath, rules, dryRun);
    std::this_thread::sleep_for(std::chrono::seconds(interval));
  }
  return 0;
}
