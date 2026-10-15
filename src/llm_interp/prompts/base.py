from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptItem:
    pair_id: str
    question: str
    options: dict[str, str]  # letter -> choice text, in original order
    gold: str  # letter


class PromptSource(ABC):
    @abstractmethod
    def load(self) -> list[PromptItem]: ...
