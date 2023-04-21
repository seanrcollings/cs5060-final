import typing as t


class EventContainer:
    def __init__(self) -> None:
        self.event_handlers: dict[str, list[t.Callable[[dict[str, t.Any]], None]]] = {}

    def on(self, event: str, handler: t.Callable[[dict[str, t.Any]], None]):
        self.event_handlers.setdefault(event, []).append(handler)

    def event(self, name: str, data: dict[str, t.Any]):
        for handler in self.event_handlers.get(name, []):
            handler(data)
