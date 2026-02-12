"""Header bar widget — slim branded header with model info"""
from textual.widget import Widget
from textual.reactive import reactive
from rich.table import Table
from rich.text import Text


class HeaderBar(Widget):
    """Slim single-line header — app name left, model name right.

    Replaces the old ModelLabel. Provides branding context without
    consuming excessive vertical space.
    """

    model_name: reactive[str] = reactive("")

    def render(self):
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right", ratio=1)

        left = Text()
        left.append(" ShiYi", style="bold #7aa2f7")

        right = Text()
        if self.model_name:
            right.append(f"{self.model_name} ", style="#565f89")

        grid.add_row(left, right)
        return grid
