/**
 * llama_bindings.cpp — QuantConfig and ModelMetrics structs.
 *
 * QuantConfig describes a quantization job (bits, method, thresholds).
 * ModelMetrics captures inference performance (tokens/sec, peak memory, etc.).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <sstream>

namespace py = pybind11;

// ── QuantConfig ───────────────────────────────────────────────────────────────

struct QuantConfig {
    std::string method;        // "q4_k_m", "q5_k_m", "awq", "gptq", "exl2"
    int         bits         = 4;
    int         group_size   = 128;
    bool        use_k_quant  = true;   // K-quant variants (GGUF)
    double      perplexity_threshold = 0.0;  // 0 = no gate
    std::string output_format;  // "gguf", "safetensors", "mlx", "onnx"
    std::string output_path;

    std::string key() const {
        return method + "_b" + std::to_string(bits) + "_g" + std::to_string(group_size);
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "QuantConfig(method=" << method
           << ", bits=" << bits
           << ", group=" << group_size
           << ", fmt=" << output_format << ")";
        return ss.str();
    }
};

// ── ModelMetrics ──────────────────────────────────────────────────────────────

struct ModelMetrics {
    std::string model_id;
    std::string backend;       // "gguf", "mlx", "pytorch", "vllm"
    double      tokens_per_sec  = 0.0;
    double      ttft_ms         = 0.0;   // time-to-first-token
    double      peak_memory_gb  = 0.0;
    long long   param_count     = 0;
    double      perplexity      = 0.0;
    double      quality_score   = 0.0;
    std::string quant_config_key;

    bool is_better_than(const ModelMetrics& other) const {
        // Simple heuristic: lower perplexity + higher quality + faster
        if (perplexity > 0 && other.perplexity > 0) {
            return perplexity < other.perplexity;
        }
        return quality_score > other.quality_score;
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "ModelMetrics(id=" << model_id
           << ", tps=" << tokens_per_sec
           << ", ppl=" << perplexity
           << ", mem=" << peak_memory_gb << "GB)";
        return ss.str();
    }
};

// ── Bindings ──────────────────────────────────────────────────────────────────

void bind_llama(py::module_& m) {
    py::class_<QuantConfig>(m, "QuantConfig")
        .def(py::init<>())
        .def_readwrite("method",               &QuantConfig::method)
        .def_readwrite("bits",                 &QuantConfig::bits)
        .def_readwrite("group_size",           &QuantConfig::group_size)
        .def_readwrite("use_k_quant",          &QuantConfig::use_k_quant)
        .def_readwrite("perplexity_threshold", &QuantConfig::perplexity_threshold)
        .def_readwrite("output_format",        &QuantConfig::output_format)
        .def_readwrite("output_path",          &QuantConfig::output_path)
        .def("key",  &QuantConfig::key)
        .def("to_dict", [](const QuantConfig& q) {
            return py::dict(
                py::arg("method")               = q.method,
                py::arg("bits")                 = q.bits,
                py::arg("group_size")           = q.group_size,
                py::arg("use_k_quant")          = q.use_k_quant,
                py::arg("perplexity_threshold") = q.perplexity_threshold,
                py::arg("output_format")        = q.output_format,
                py::arg("output_path")          = q.output_path
            );
        })
        .def("__repr__", &QuantConfig::repr);

    py::class_<ModelMetrics>(m, "ModelMetrics")
        .def(py::init<>())
        .def_readwrite("model_id",         &ModelMetrics::model_id)
        .def_readwrite("backend",          &ModelMetrics::backend)
        .def_readwrite("tokens_per_sec",   &ModelMetrics::tokens_per_sec)
        .def_readwrite("ttft_ms",          &ModelMetrics::ttft_ms)
        .def_readwrite("peak_memory_gb",   &ModelMetrics::peak_memory_gb)
        .def_readwrite("param_count",      &ModelMetrics::param_count)
        .def_readwrite("perplexity",       &ModelMetrics::perplexity)
        .def_readwrite("quality_score",    &ModelMetrics::quality_score)
        .def_readwrite("quant_config_key", &ModelMetrics::quant_config_key)
        .def("is_better_than", &ModelMetrics::is_better_than)
        .def("to_dict", [](const ModelMetrics& m) {
            return py::dict(
                py::arg("model_id")       = m.model_id,
                py::arg("backend")        = m.backend,
                py::arg("tokens_per_sec") = m.tokens_per_sec,
                py::arg("ttft_ms")        = m.ttft_ms,
                py::arg("peak_memory_gb") = m.peak_memory_gb,
                py::arg("param_count")    = m.param_count,
                py::arg("perplexity")     = m.perplexity,
                py::arg("quality_score")  = m.quality_score
            );
        })
        .def("__repr__", &ModelMetrics::repr);
}
