/**
 * metric_streamer.cpp — TrainingStepMetrics struct for live telemetry.
 *
 * Designed for zero-copy streaming: the training loop writes metrics into
 * a shared TrainingStepMetrics object; the UI polls it via the Python bridge.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>

namespace py = pybind11;

// ── TrainingStepMetrics ───────────────────────────────────────────────────────

struct TrainingStepMetrics {
    int         step           = 0;
    int         total_steps    = 0;
    double      loss           = 0.0;
    double      learning_rate  = 0.0;
    double      grad_norm      = 0.0;
    double      tokens_per_sec = 0.0;
    double      gpu_mem_gb     = 0.0;
    double      elapsed_sec    = 0.0;
    std::string phase;         // "warmup", "sft", "minillm", "eval"
    std::string backend;

    double progress_pct() const {
        if (total_steps <= 0) return 0.0;
        return 100.0 * static_cast<double>(step) / static_cast<double>(total_steps);
    }

    double eta_sec() const {
        if (step <= 0 || total_steps <= 0) return 0.0;
        double rate = elapsed_sec / static_cast<double>(step);
        return rate * static_cast<double>(total_steps - step);
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "TrainingStepMetrics(step=" << step << "/" << total_steps
           << ", loss=" << loss
           << ", lr=" << learning_rate
           << ", " << static_cast<int>(progress_pct()) << "%)";
        return ss.str();
    }
};

// ── MetricsHistory — rolling window ──────────────────────────────────────────

struct MetricsHistory {
    std::vector<TrainingStepMetrics> steps;
    size_t max_size = 1000;

    void push(const TrainingStepMetrics& m) {
        steps.push_back(m);
        if (steps.size() > max_size) {
            steps.erase(steps.begin());
        }
    }

    std::vector<double> loss_series() const {
        std::vector<double> out;
        out.reserve(steps.size());
        for (const auto& s : steps) out.push_back(s.loss);
        return out;
    }

    std::vector<int> step_series() const {
        std::vector<int> out;
        out.reserve(steps.size());
        for (const auto& s : steps) out.push_back(s.step);
        return out;
    }

    double smoothed_loss(int window = 10) const {
        if (steps.empty()) return 0.0;
        int n = static_cast<int>(steps.size());
        int start = std::max(0, n - window);
        double sum = 0.0;
        for (int i = start; i < n; ++i) sum += steps[i].loss;
        return sum / static_cast<double>(n - start);
    }

    void clear() { steps.clear(); }
};

// ── Bindings ──────────────────────────────────────────────────────────────────

void bind_metrics(py::module_& m) {
    py::class_<TrainingStepMetrics>(m, "TrainingStepMetrics")
        .def(py::init<>())
        .def_readwrite("step",           &TrainingStepMetrics::step)
        .def_readwrite("total_steps",    &TrainingStepMetrics::total_steps)
        .def_readwrite("loss",           &TrainingStepMetrics::loss)
        .def_readwrite("learning_rate",  &TrainingStepMetrics::learning_rate)
        .def_readwrite("grad_norm",      &TrainingStepMetrics::grad_norm)
        .def_readwrite("tokens_per_sec", &TrainingStepMetrics::tokens_per_sec)
        .def_readwrite("gpu_mem_gb",     &TrainingStepMetrics::gpu_mem_gb)
        .def_readwrite("elapsed_sec",    &TrainingStepMetrics::elapsed_sec)
        .def_readwrite("phase",          &TrainingStepMetrics::phase)
        .def_readwrite("backend",        &TrainingStepMetrics::backend)
        .def("progress_pct", &TrainingStepMetrics::progress_pct)
        .def("eta_sec",      &TrainingStepMetrics::eta_sec)
        .def("to_dict", [](const TrainingStepMetrics& t) {
            return py::dict(
                py::arg("step")           = t.step,
                py::arg("total_steps")    = t.total_steps,
                py::arg("loss")           = t.loss,
                py::arg("learning_rate")  = t.learning_rate,
                py::arg("grad_norm")      = t.grad_norm,
                py::arg("tokens_per_sec") = t.tokens_per_sec,
                py::arg("gpu_mem_gb")     = t.gpu_mem_gb,
                py::arg("elapsed_sec")    = t.elapsed_sec,
                py::arg("phase")          = t.phase,
                py::arg("backend")        = t.backend,
                py::arg("progress_pct")   = t.progress_pct()
            );
        })
        .def("__repr__", &TrainingStepMetrics::repr);

    py::class_<MetricsHistory>(m, "MetricsHistory")
        .def(py::init<>())
        .def_readwrite("max_size",    &MetricsHistory::max_size)
        .def_readwrite("steps",       &MetricsHistory::steps)
        .def("push",          &MetricsHistory::push)
        .def("loss_series",   &MetricsHistory::loss_series)
        .def("step_series",   &MetricsHistory::step_series)
        .def("smoothed_loss", &MetricsHistory::smoothed_loss, py::arg("window") = 10)
        .def("clear",         &MetricsHistory::clear)
        .def("__len__", [](const MetricsHistory& h) { return h.steps.size(); });
}
