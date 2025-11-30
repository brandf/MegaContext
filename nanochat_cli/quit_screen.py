from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Label, Button
from textual.containers import Vertical, Horizontal


class QuitConfirmScreen(ModalScreen[bool]):
    """Simple yes/no quit confirmation."""

    CSS_PATH = "styles/quit_modal.tcss"

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-modal"):
            yield Label("Do you want to quit?", id="quit-label")
            with Horizontal(id="quit-actions"):
                yield Button("No", id="quit-no")
                yield Button("Yes", id="quit-yes")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-yes":
            self.dismiss(True)
        elif event.button.id == "quit-no":
            self.dismiss(False)
