from textual.message import Message


class TabSwitchRequest(Message):
    """Generic request to switch to a tab by id (e.g., 'tab-dataset')."""

    def __init__(self, tab_id: str) -> None:
        super().__init__()
        self.tab_id = tab_id
