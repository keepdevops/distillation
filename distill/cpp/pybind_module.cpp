/**
 * pybind_module.cpp — PYBIND11_MODULE entry point for distill_cpp.
 *
 * Aggregates all sub-module bindings declared in the other .cpp files.
 * Import from Python:
 *
 *     from distill.cpp import distill_cpp
 *     r = distill_cpp.ThermalReading()
 *     m = distill_cpp.ModelMetrics()
 *     h = distill_cpp.MetricsHistory()
 *     l = distill_cpp.LoRAConfig()
 *     e = distill_cpp.ExportFormatSpec()
 */
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward-declare bind_* functions defined in sibling translation units
void bind_thermal(py::module_& m);
void bind_llama(py::module_& m);
void bind_metrics(py::module_& m);
void bind_lora(py::module_& m);
void bind_export(py::module_& m);

PYBIND11_MODULE(distill_cpp, m) {
    m.doc() = "distill_cpp — High-performance telemetry structs for the Wow Sausage Maker";

    m.attr("__version__") = "0.2.0";

    // Sub-namespaces keep the flat module tidy
    auto thermal = m.def_submodule("thermal", "Hardware thermal structs");
    bind_thermal(thermal);

    auto llama = m.def_submodule("llama", "Quantization and model metric structs");
    bind_llama(llama);

    auto metrics = m.def_submodule("metrics", "Training step metrics and history");
    bind_metrics(metrics);

    auto lora = m.def_submodule("lora", "LoRA adapter config and training metrics");
    bind_lora(lora);

    auto exports = m.def_submodule("export", "Export format specs and results");
    bind_export(exports);

    // Also expose at top level for convenience
    bind_thermal(m);
    bind_llama(m);
    bind_metrics(m);
    bind_lora(m);
    bind_export(m);
}
