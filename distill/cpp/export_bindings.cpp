/**
 * export_bindings.cpp — ExportFormatSpec and ExportResult structs.
 *
 * ExportFormatSpec describes one target format with all its parameters.
 * ExportResult captures the outcome: output path, size, timing, errors.
 * ExportManifest bundles multiple results for a "production pack".
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <sstream>

namespace py = pybind11;

// ── ExportFormatSpec ───────────────────────────────────────────────────────────

struct ExportFormatSpec {
    std::string format_key;       // "gguf", "awq", "safetensors", etc.
    std::string quant_method;     // "q4_k_m", "awq-4bit", etc.
    int         bits           = 4;
    int         group_size     = 128;
    bool        merge_lora     = true;
    bool        push_to_hub    = false;
    std::string output_dir;
    std::string hub_repo_id;
    std::string optimize_for   = "balanced"; // "speed","quality","size","balanced"

    std::string key() const {
        return format_key + "_" + quant_method + "_b" + std::to_string(bits);
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "ExportFormatSpec(format=" << format_key
           << ", quant=" << quant_method
           << ", bits=" << bits
           << ", merge_lora=" << (merge_lora ? "true" : "false") << ")";
        return ss.str();
    }
};

// ── ExportResult ──────────────────────────────────────────────────────────────

struct ExportResult {
    std::string format_key;
    std::string output_path;
    bool        success        = false;
    std::string error;
    double      size_mb        = 0.0;
    double      elapsed_sec    = 0.0;
    double      perplexity_delta = 0.0;  // vs FP16 baseline (negative = better)
    std::string checksum;               // SHA256 of primary output file

    bool is_ok() const { return success && error.empty(); }

    std::string status_icon() const {
        return is_ok() ? "✅" : "❌";
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "ExportResult(format=" << format_key
           << ", " << (is_ok() ? "ok" : "FAILED")
           << ", size=" << size_mb << "MB"
           << ", time=" << elapsed_sec << "s)";
        return ss.str();
    }
};

// ── ExportManifest ────────────────────────────────────────────────────────────

struct ExportManifest {
    std::string model_id;
    std::string source_path;
    std::string created_at;
    std::vector<ExportResult> results;

    void add_result(const ExportResult& r) {
        results.push_back(r);
    }

    int success_count() const {
        int n = 0;
        for (const auto& r : results) if (r.is_ok()) ++n;
        return n;
    }

    int failure_count() const {
        return static_cast<int>(results.size()) - success_count();
    }

    double total_size_mb() const {
        double total = 0;
        for (const auto& r : results) total += r.size_mb;
        return total;
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "ExportManifest(model=" << model_id
           << ", formats=" << results.size()
           << ", ok=" << success_count()
           << ", total=" << total_size_mb() << "MB)";
        return ss.str();
    }
};

// ── Bindings ──────────────────────────────────────────────────────────────────

void bind_export(py::module_& m) {
    py::class_<ExportFormatSpec>(m, "ExportFormatSpec")
        .def(py::init<>())
        .def_readwrite("format_key",   &ExportFormatSpec::format_key)
        .def_readwrite("quant_method", &ExportFormatSpec::quant_method)
        .def_readwrite("bits",         &ExportFormatSpec::bits)
        .def_readwrite("group_size",   &ExportFormatSpec::group_size)
        .def_readwrite("merge_lora",   &ExportFormatSpec::merge_lora)
        .def_readwrite("push_to_hub",  &ExportFormatSpec::push_to_hub)
        .def_readwrite("output_dir",   &ExportFormatSpec::output_dir)
        .def_readwrite("hub_repo_id",  &ExportFormatSpec::hub_repo_id)
        .def_readwrite("optimize_for", &ExportFormatSpec::optimize_for)
        .def("key", &ExportFormatSpec::key)
        .def("to_dict", [](const ExportFormatSpec& s) {
            return py::dict(
                py::arg("format_key")   = s.format_key,
                py::arg("quant_method") = s.quant_method,
                py::arg("bits")         = s.bits,
                py::arg("group_size")   = s.group_size,
                py::arg("merge_lora")   = s.merge_lora,
                py::arg("push_to_hub")  = s.push_to_hub,
                py::arg("output_dir")   = s.output_dir,
                py::arg("optimize_for") = s.optimize_for
            );
        })
        .def("__repr__", &ExportFormatSpec::repr);

    py::class_<ExportResult>(m, "ExportResult")
        .def(py::init<>())
        .def_readwrite("format_key",       &ExportResult::format_key)
        .def_readwrite("output_path",      &ExportResult::output_path)
        .def_readwrite("success",          &ExportResult::success)
        .def_readwrite("error",            &ExportResult::error)
        .def_readwrite("size_mb",          &ExportResult::size_mb)
        .def_readwrite("elapsed_sec",      &ExportResult::elapsed_sec)
        .def_readwrite("perplexity_delta", &ExportResult::perplexity_delta)
        .def_readwrite("checksum",         &ExportResult::checksum)
        .def("is_ok",       &ExportResult::is_ok)
        .def("status_icon", &ExportResult::status_icon)
        .def("to_dict", [](const ExportResult& r) {
            return py::dict(
                py::arg("format_key")       = r.format_key,
                py::arg("output_path")      = r.output_path,
                py::arg("success")          = r.success,
                py::arg("error")            = r.error,
                py::arg("size_mb")          = r.size_mb,
                py::arg("elapsed_sec")      = r.elapsed_sec,
                py::arg("perplexity_delta") = r.perplexity_delta
            );
        })
        .def("__repr__", &ExportResult::repr);

    py::class_<ExportManifest>(m, "ExportManifest")
        .def(py::init<>())
        .def_readwrite("model_id",    &ExportManifest::model_id)
        .def_readwrite("source_path", &ExportManifest::source_path)
        .def_readwrite("created_at",  &ExportManifest::created_at)
        .def_readwrite("results",     &ExportManifest::results)
        .def("add_result",    &ExportManifest::add_result)
        .def("success_count", &ExportManifest::success_count)
        .def("failure_count", &ExportManifest::failure_count)
        .def("total_size_mb", &ExportManifest::total_size_mb)
        .def("__len__", [](const ExportManifest& m) { return m.results.size(); })
        .def("__repr__", &ExportManifest::repr);
}
