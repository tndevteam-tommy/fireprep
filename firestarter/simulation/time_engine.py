"""
Firestarter - Time Passage Engine & Temporal Fast-Forward
Implements DESIGN Section 4.2 Time Passage Engine ('T' key).
"""

from typing import List, Dict, Any, Callable


class TimeSkipOption:
    def __init__(self, label: str, hours: float, description: str):
        self.label = label
        self.hours = hours
        self.description = description


TIME_SKIP_OPTIONS: List[TimeSkipOption] = [
    TimeSkipOption("1 Hour", 1.0, "Short rest. The fire burns down and the coals settle."),
    TimeSkipOption("6 Hours", 6.0, "Through the night. Mist rises, the ground dampens."),
    TimeSkipOption("24 Hours (1 Day)", 24.0, "A full day-night cycle. Surface drying on the racks."),
    TimeSkipOption("7 Days (1 Week)", 168.0, "Weekly weathering. Logs on the soil wick up moisture."),
    TimeSkipOption("30 Days (1 Month)", 720.0, "A month of seasoning. Split wood cures quickly."),
    TimeSkipOption("60 Days (Cure Period)", 1440.0, "Ideal curing. Tarped, raised logs reach ~12 % EMC."),
    TimeSkipOption("90 Days (1 Season)", 2160.0, "A full season turns. Ground logs may rot into punk wood."),
]


class TimeEngine:
    def __init__(self):
        self.is_menu_open = False
        self.selected_option_index = 2  # default: 1 day
        self.total_days_elapsed = 0.0
        self.last_report: Dict[str, Any] = {}

    def toggle_menu(self) -> bool:
        self.is_menu_open = not self.is_menu_open
        return self.is_menu_open

    def next_option(self):
        self.selected_option_index = (self.selected_option_index + 1) % len(TIME_SKIP_OPTIONS)

    def prev_option(self):
        self.selected_option_index = (self.selected_option_index - 1) % len(TIME_SKIP_OPTIONS)

    @property
    def selected(self) -> TimeSkipOption:
        return TIME_SKIP_OPTIONS[self.selected_option_index]

    def execute_selected_skip(self, on_skip_callback: Callable[[float], Any]) -> Dict[str, Any]:
        """Advances the simulation by the selected option."""
        opt = self.selected
        days = opt.hours / 24.0
        report = on_skip_callback(opt.hours)
        if not isinstance(report, dict):
            self.total_days_elapsed += days
            report = {}
        self.last_report = report
        self.is_menu_open = False
        return {"label": opt.label, "hours": opt.hours, "days": days, "report": report,
                "message": f"Advanced time by {opt.label}. {opt.description}"}
