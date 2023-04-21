import typing as t


class EventContainer:
    def __init__(self) -> None:
        self.event_handlers: dict[
            str, list[t.Callable[[object, dict[str, t.Any]], None]]
        ] = {}

    def on(self, event: str):
        def inner(handler: t.Callable[[object, dict[str, t.Any]], None]) -> None:
            self.event_handlers.setdefault(event, []).append(handler)

        return inner

    def event(self, name: str, data: dict[str, t.Any]):
        for handler in self.event_handlers.get(name, []):
            handler(self, data)
