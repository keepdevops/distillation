/**
 * thermal_bindings.cpp — ThermalReading and HardwareProfile structs.
 *
 * Exposed to Python via pybind11.  Populated by the Python bridge
 * (distill/backends/cpp_thermal_bridge.py) from mactop JSON output.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

namespace py = pybind11;

// ── ThermalReading ────────────────────────────────────────────────────────────

struct ThermalReading {
    double cpu_temp   = 0.0;
    double gpu_temp   = 0.0;
    double soc_temp   = 0.0;
    double cpu_power  = 0.0;
    double gpu_power  = 0.0;
    double total_power = 0.0;
    bool   available  = false;
    std::string error;

    double peak_temp() const {
        return std::max({cpu_temp, gpu_temp, soc_temp});
    }

    // "low" / "medium" / "high"
    std::string oom_risk(double threshold = 85.0) const {
        double peak = peak_temp();
        if (peak >= threshold)        return "high";
        if (peak >= threshold * 0.85) return "medium";
        return "low";
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "ThermalReading(cpu=" << cpu_temp
           << "°C, gpu=" << gpu_temp
           << "°C, soc=" << soc_temp
           << "°C, power=" << total_power << "W"
           << ", risk=" << oom_risk() << ")";
        return ss.str();
    }
};

// ── HardwareProfile ───────────────────────────────────────────────────────────

struct HardwareProfile {
    std::string device_label;
    std::string machine;       // "arm64", "x86_64"
    std::string backend_hint;  // "mlx", "unsloth", "sft", "cpu"
    double      ram_gb        = 0.0;
    double      vram_gb       = 0.0;
    bool        has_mps       = false;
    bool        has_cuda      = false;
    std::vector<ThermalReading> history;  // rolling window of readings

    void push_reading(const ThermalReading& r) {
        history.push_back(r);
        // Keep last 120 readings (≈ 2 min at 1s interval)
        if (history.size() > 120) {
            history.erase(history.begin());
        }
    }

    double avg_cpu_temp() const {
        if (history.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& r : history) sum += r.cpu_temp;
        return sum / static_cast<double>(history.size());
    }

    std::string repr() const {
        std::ostringstream ss;
        ss << "HardwareProfile(device=" << device_label
           << ", ram=" << ram_gb << "GB"
           << ", backend=" << backend_hint
           << ", history_len=" << history.size() << ")";
        return ss.str();
    }
};

// ── Bindings ──────────────────────────────────────────────────────────────────

void bind_thermal(py::module_& m) {
    py::class_<ThermalReading>(m, "ThermalReading")
        .def(py::init<>())
        .def_readwrite("cpu_temp",    &ThermalReading::cpu_temp)
        .def_readwrite("gpu_temp",    &ThermalReading::gpu_temp)
        .def_readwrite("soc_temp",    &ThermalReading::soc_temp)
        .def_readwrite("cpu_power",   &ThermalReading::cpu_power)
        .def_readwrite("gpu_power",   &ThermalReading::gpu_power)
        .def_readwrite("total_power", &ThermalReading::total_power)
        .def_readwrite("available",   &ThermalReading::available)
        .def_readwrite("error",       &ThermalReading::error)
        .def("peak_temp", &ThermalReading::peak_temp)
        .def("oom_risk",  &ThermalReading::oom_risk, py::arg("threshold") = 85.0)
        .def("to_dict", [](const ThermalReading& r) {
            return py::dict(
                py::arg("cpu_temp")    = r.cpu_temp,
                py::arg("gpu_temp")    = r.gpu_temp,
                py::arg("soc_temp")    = r.soc_temp,
                py::arg("cpu_power")   = r.cpu_power,
                py::arg("gpu_power")   = r.gpu_power,
                py::arg("total_power") = r.total_power,
                py::arg("available")   = r.available,
                py::arg("error")       = r.error
            );
        })
        .def("__repr__", &ThermalReading::repr);

    py::class_<HardwareProfile>(m, "HardwareProfile")
        .def(py::init<>())
        .def_readwrite("device_label",  &HardwareProfile::device_label)
        .def_readwrite("machine",       &HardwareProfile::machine)
        .def_readwrite("backend_hint",  &HardwareProfile::backend_hint)
        .def_readwrite("ram_gb",        &HardwareProfile::ram_gb)
        .def_readwrite("vram_gb",       &HardwareProfile::vram_gb)
        .def_readwrite("has_mps",       &HardwareProfile::has_mps)
        .def_readwrite("has_cuda",      &HardwareProfile::has_cuda)
        .def_readwrite("history",       &HardwareProfile::history)
        .def("push_reading",   &HardwareProfile::push_reading)
        .def("avg_cpu_temp",   &HardwareProfile::avg_cpu_temp)
        .def("__repr__",       &HardwareProfile::repr);
}
