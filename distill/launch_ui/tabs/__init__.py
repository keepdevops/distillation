"""Tab sub-package — layout builders and event wiring."""

from .tab_configure import build_tab_configure
from .tab_data_prep import build_tab_data_prep
from .tab_domain import build_tab_domain
from .tab_eval import build_tab_eval
from .tab_expert import build_tab_expert
from .tab_logs import build_tab_logs
from .tab_help import build_tab_help
from .wiring import wire_events

__all__ = [
    "build_tab_configure",
    "build_tab_data_prep",
    "build_tab_domain",
    "build_tab_eval",
    "build_tab_expert",
    "build_tab_logs",
    "build_tab_help",
    "wire_events",
]
