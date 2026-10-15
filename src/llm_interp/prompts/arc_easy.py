from datasets import load_dataset

from .base import PromptItem, PromptSource


class ArcEasyPromptSource(PromptSource):
    """~30 ARC-Easy questions, for exercising the data model — the results don't
    matter. A small number of ARC-Easy rows use numeric choice labels ("1".."4")
    instead of letters; those are skipped so every hint ("the answer is X") and
    every requested JSON answer refers to a letter consistently.
    """

    def __init__(self, n: int = 30, split: str = "test"):
        self.n = n
        self.split = split

    def load(self) -> list[PromptItem]:
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=self.split)
        items: list[PromptItem] = []
        for row in dataset:
            labels = row["choices"]["label"]
            if not all(label.isalpha() for label in labels):
                continue
            options = dict(zip(labels, row["choices"]["text"]))
            if row["answerKey"] not in options:
                continue
            items.append(
                PromptItem(pair_id=row["id"], question=row["question"], options=options, gold=row["answerKey"])
            )
            if len(items) >= self.n:
                break
        return items
