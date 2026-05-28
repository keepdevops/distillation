/**
 * lora_bindings.cpp — LoRAConfig and LoRATrainingMetrics structs.
 *
 * LoRAConfig describes a LoRA / QLoRA adapter configuration.
 * LoRATrainingMetrics tracks live adapter health during training:
 *   adapter norms, update ratios, and per-layer gradient flow.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace py = pybind11;

// ── LoRAConfig ─────────────────────────────────────────────────────────────────

struct LoRAConfig {
    int    rank          = 16;
    int    alpha         = 32;
    double dropout       = 0.05;
    bool   use_qlora     = false;
    int    qlora_bits    = 4;
    std::string bias     = "none";
    std::string task_type = "CAUSAL_LM";
    std::vector<std::string> target_modules = {"q_proj", "v_proj"};

    double scaling() const {
        return static_cast<double>(alpha) / static_cast<double>(rank);
    }

    // Estimated trainable params given model dimensions
    long long estimated_params(int hidden_size = 2048, int num_layers = 24) const {
        long long per_layer = 2LL * hidden_size * rank
                              * static_cast<long long>(target_modules.size()) / 2;
        return per_layer * num_layers;
    }

    // Estimated VRAM for adapters in MB (float16)
    double estimated_vram_mb(int hidden_size = 2048, int num_layers = 24,
                              int dtype_bytes = 2) const {
        double params = static_cast<double>(estimated_params(hidden_size, num_layers));
        return (params * dtype_bytes) / (1024.0 * 1024.0);
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "LoRAConfig(rank=" << rank
           << ", alpha=" << alpha
           << ", scaling=" << scaling()
           << ", qlora=" << (use_qlora ? "true" : "false")
           << ", targets=" << target_modules.size() << ")";
        return ss.str();
    }
};

// ── LoRATrainingMetrics ────────────────────────────────────────────────────────

struct LoRALayerMetrics {
    std::string layer_name;
    double      adapter_norm   = 0.0;   // ||W_A||_F * ||W_B||_F
    double      update_ratio   = 0.0;   // ||ΔW||/||W_0|| (relative update magnitude)
    double      grad_norm      = 0.0;   // gradient norm for this adapter
    bool        is_active      = true;  // flag dead/saturated layers
};

struct LoRATrainingMetrics {
    int    step           = 0;
    double mean_adapter_norm  = 0.0;
    double mean_update_ratio  = 0.0;
    double mean_grad_norm     = 0.0;
    double rank_utilization   = 1.0;   // effective rank / nominal rank (1.0 = full use)
    int    dead_layers        = 0;
    std::vector<LoRALayerMetrics> layers;

    void push_layer(const LoRALayerMetrics& m) {
        layers.push_back(m);
    }

    void compute_aggregates() {
        if (layers.empty()) return;
        double sum_norm = 0, sum_ratio = 0, sum_grad = 0;
        int dead = 0;
        for (const auto& l : layers) {
            sum_norm  += l.adapter_norm;
            sum_ratio += l.update_ratio;
            sum_grad  += l.grad_norm;
            if (!l.is_active) ++dead;
        }
        double n = static_cast<double>(layers.size());
        mean_adapter_norm = sum_norm  / n;
        mean_update_ratio = sum_ratio / n;
        mean_grad_norm    = sum_grad  / n;
        dead_layers       = dead;
    }

    bool health_ok() const {
        // Adapters are healthy when norms are non-trivial and updates are proportionate
        return mean_adapter_norm > 1e-6
            && mean_update_ratio > 1e-5
            && dead_layers == 0;
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "LoRATrainingMetrics(step=" << step
           << ", norm=" << mean_adapter_norm
           << ", ratio=" << mean_update_ratio
           << ", dead=" << dead_layers
           << ", health=" << (health_ok() ? "ok" : "WARN") << ")";
        return ss.str();
    }
};

// ── Bindings ───────────────────────────────────────────────────────────────────

void bind_lora(py::module_& m) {
    py::class_<LoRAConfig>(m, "LoRAConfig")
        .def(py::init<>())
        .def_readwrite("rank",           &LoRAConfig::rank)
        .def_readwrite("alpha",          &LoRAConfig::alpha)
        .def_readwrite("dropout",        &LoRAConfig::dropout)
        .def_readwrite("use_qlora",      &LoRAConfig::use_qlora)
        .def_readwrite("qlora_bits",     &LoRAConfig::qlora_bits)
        .def_readwrite("bias",           &LoRAConfig::bias)
        .def_readwrite("task_type",      &LoRAConfig::task_type)
        .def_readwrite("target_modules", &LoRAConfig::target_modules)
        .def("scaling",           &LoRAConfig::scaling)
        .def("estimated_params",  &LoRAConfig::estimated_params,
             py::arg("hidden_size") = 2048, py::arg("num_layers") = 24)
        .def("estimated_vram_mb", &LoRAConfig::estimated_vram_mb,
             py::arg("hidden_size") = 2048, py::arg("num_layers") = 24,
             py::arg("dtype_bytes") = 2)
        .def("to_dict", [](const LoRAConfig& c) {
            return py::dict(
                py::arg("rank")           = c.rank,
                py::arg("alpha")          = c.alpha,
                py::arg("dropout")        = c.dropout,
                py::arg("use_qlora")      = c.use_qlora,
                py::arg("qlora_bits")     = c.qlora_bits,
                py::arg("bias")           = c.bias,
                py::arg("target_modules") = c.target_modules,
                py::arg("scaling")        = c.scaling()
            );
        })
        .def("__repr__", &LoRAConfig::repr);

    py::class_<LoRALayerMetrics>(m, "LoRALayerMetrics")
        .def(py::init<>())
        .def_readwrite("layer_name",   &LoRALayerMetrics::layer_name)
        .def_readwrite("adapter_norm", &LoRALayerMetrics::adapter_norm)
        .def_readwrite("update_ratio", &LoRALayerMetrics::update_ratio)
        .def_readwrite("grad_norm",    &LoRALayerMetrics::grad_norm)
        .def_readwrite("is_active",    &LoRALayerMetrics::is_active);

    py::class_<LoRATrainingMetrics>(m, "LoRATrainingMetrics")
        .def(py::init<>())
        .def_readwrite("step",              &LoRATrainingMetrics::step)
        .def_readwrite("mean_adapter_norm", &LoRATrainingMetrics::mean_adapter_norm)
        .def_readwrite("mean_update_ratio", &LoRATrainingMetrics::mean_update_ratio)
        .def_readwrite("mean_grad_norm",    &LoRATrainingMetrics::mean_grad_norm)
        .def_readwrite("rank_utilization",  &LoRATrainingMetrics::rank_utilization)
        .def_readwrite("dead_layers",       &LoRATrainingMetrics::dead_layers)
        .def_readwrite("layers",            &LoRATrainingMetrics::layers)
        .def("push_layer",          &LoRATrainingMetrics::push_layer)
        .def("compute_aggregates",  &LoRATrainingMetrics::compute_aggregates)
        .def("health_ok",           &LoRATrainingMetrics::health_ok)
        .def("to_dict", [](const LoRATrainingMetrics& m) {
            return py::dict(
                py::arg("step")              = m.step,
                py::arg("mean_adapter_norm") = m.mean_adapter_norm,
                py::arg("mean_update_ratio") = m.mean_update_ratio,
                py::arg("mean_grad_norm")    = m.mean_grad_norm,
                py::arg("rank_utilization")  = m.rank_utilization,
                py::arg("dead_layers")       = m.dead_layers,
                py::arg("health_ok")         = m.health_ok()
            );
        })
        .def("__repr__", &LoRATrainingMetrics::repr);
}
