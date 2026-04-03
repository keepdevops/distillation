"""Infrastructure utilities — device, metrics, paths, CLI helpers, callbacks."""
from .train_utils import get_device, clear_device_cache, check_pause_flag, write_metric, load_student_model
from .metrics_io import load_metrics
from .paths import project_dir, scripts_dir, package_dir

__all__ = [
    "get_device",
    "clear_device_cache",
    "check_pause_flag",
    "write_metric",
    "load_student_model",
    "load_metrics",
    "project_dir",
    "scripts_dir",
    "package_dir",
]
