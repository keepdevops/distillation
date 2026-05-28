"""Reflex app-level state — single source of truth for the UI.

All reactive state lives here. Components read from and dispatch to
this state class. Follows Reflex's event-handler pattern.
"""
from __future__ import annotations

import time
from typing import Any

try:
    import reflex as rx  # type: ignore[import]
    _HAS_REFLEX = True
except ImportError:
    _HAS_REFLEX = False
    # Provide a stub so the module is importable without Reflex installed
    class _RxStub:
        class State:
            pass
        class Var:
            pass
    rx = _RxStub()  # type: ignore[assignment]


if _HAS_REFLEX:
    class AppState(rx.State):
        """Root application state shared across all pages."""

        # ── Session config ─────────────────────────────────────────────────
        backend: str   = "mlx"
        teacher: str   = "Qwen/Qwen2-1.5B-Instruct"
        student: str   = "Qwen/Qwen2-0.5B-Instruct"
        dataset: str   = "yahma/alpaca-cleaned"
        output_dir: str = "outputs/distilled"
        epochs: int    = 3
        lr: float      = 2e-4
        batch_size: int = 4
        lora_rank: int  = 16
        preset_name: str = ""

        # ── Job tracking ───────────────────────────────────────────────────
        job_status: str    = "idle"     # idle/running/paused/completed/failed
        job_phase: str     = ""
        job_step: int      = 0
        job_total: int     = 0
        job_loss: float    = 0.0
        job_best_loss: float = 9999.0
        job_elapsed: str   = "00:00"
        job_id: str        = ""
        job_log: list[str] = []

        # ── Hardware snapshot ──────────────────────────────────────────────
        cpu_temp: float  = 0.0
        gpu_temp: float  = 0.0
        total_power: float = 0.0
        ram_used_gb: float = 0.0
        backend_hint: str = "mlx"
        hw_device: str    = ""
        thermal_risk: str = "low"

        # ── Training metrics series ────────────────────────────────────────
        loss_steps: list[int]   = []
        loss_values: list[float] = []

        # ── UI chrome ─────────────────────────────────────────────────────
        active_tab: str    = "hardware"
        show_cli_mirror: bool = True
        is_airgap: bool    = False

        # ── Event handlers ─────────────────────────────────────────────────

        def set_backend(self, value: str) -> None:
            self.backend = value

        def set_teacher(self, value: str) -> None:
            self.teacher = value

        def set_student(self, value: str) -> None:
            self.student = value

        def load_preset(self, preset_name: str) -> None:
            """Load a named preset into state fields."""
            try:
                from distill.launch_ui.presets import get_preset
                p = get_preset(preset_name)
                if not p:
                    return
                self.teacher    = p.get("teacher", self.teacher)
                self.student    = p.get("student", self.student)
                self.backend    = p.get("backend", self.backend)
                self.dataset    = p.get("dataset", self.dataset)
                self.epochs     = int(p.get("epochs", self.epochs))
                self.lr         = float(p.get("lr", self.lr))
                self.batch_size = int(p.get("batch_size", self.batch_size))
                self.lora_rank  = int(p.get("lora_rank", self.lora_rank))
                self.preset_name = preset_name
            except Exception:
                pass

        def refresh_hardware(self) -> None:
            """Pull fresh thermal + hardware readings."""
            try:
                from distill.backends.cpp_thermal_bridge import read_thermal_dict, build_hardware_profile_dict
                td = read_thermal_dict()
                hw = build_hardware_profile_dict()
                self.cpu_temp     = td.get("cpu_temp", 0.0)
                self.gpu_temp     = td.get("gpu_temp", 0.0)
                self.total_power  = td.get("total_power", 0.0)
                self.backend_hint = hw.get("backend_hint", "mlx")
                self.hw_device    = hw.get("device", "")
            except Exception:
                pass
            try:
                import psutil
                ram = psutil.virtual_memory()
                self.ram_used_gb = ram.used / (1024 ** 3)
            except Exception:
                pass

        def refresh_thermal_risk(self) -> None:
            try:
                from distill.backends.cpp_thermal_bridge import oom_risk
                self.thermal_risk = oom_risk()
            except Exception:
                pass

        def check_airgap(self) -> None:
            try:
                from distill.ui.components.airgap_mode import is_airgap
                self.is_airgap = is_airgap()
            except Exception:
                pass

        def append_log(self, line: str) -> None:
            self.job_log = (self.job_log + [line])[-200:]

        def reset_job(self) -> None:
            self.job_status = "idle"
            self.job_phase  = ""
            self.job_step   = 0
            self.job_total  = 0
            self.job_loss   = 0.0
            self.job_best_loss = 9999.0
            self.job_elapsed = "00:00"
            self.job_log     = []
            self.loss_steps  = []
            self.loss_values = []

        @rx.var
        def progress_pct(self) -> float:
            if self.job_total <= 0:
                return 0.0
            return min(100.0, 100.0 * self.job_step / self.job_total)

        @rx.var
        def cli_command(self) -> str:
            """Live CLI mirror of current config."""
            return (
                f"python -m distill.orchestration.agent \\\n"
                f"    --backend {self.backend} \\\n"
                f"    --teacher {self.teacher} \\\n"
                f"    --student {self.student} \\\n"
                f"    --dataset {self.dataset} \\\n"
                f"    --output_dir {self.output_dir} \\\n"
                f"    --epochs {self.epochs} --lr {self.lr}"
            )

else:
    # Stub for import without Reflex
    class AppState:  # type: ignore[no-redef]
        """Stub AppState when Reflex is not installed."""
        backend: str = "mlx"
        teacher: str = ""
        student: str = ""

        def load_preset(self, name: str) -> None:
            pass
