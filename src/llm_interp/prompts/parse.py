import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ParsedAnswer:
    parsed_answer: str | None
    parse_status: Literal["ok", "not_found", "malformed", "multiple"]


def _iter_json_object_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) character spans of every balanced top-level `{...}`
    substring that starts at each '{' in `text`, tracking string literals (and
    escapes within them) so braces inside strings don't throw off the depth
    count. Spans may be found starting at nested '{' too (e.g. for `{"a": {...}}`,
    both the outer and inner braces yield a span) — that's intentional, since the
    model's JSON object could legitimately be nested inside other text structure.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < n:
                c = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            yield (i, j + 1)
                            break
                j += 1
        i += 1


def parse_answer(response_text: str) -> ParsedAnswer:
    """Extract the parsed answer letter from the LAST valid JSON object in
    `response_text` that has an "answer" key with a string value. Parse failures
    are recorded via `parse_status`, not raised:
      - "not_found": no JSON object with an "answer" key at all.
      - "malformed": found one, but its "answer" value isn't a non-empty string.
      - "ok": exactly one distinct string answer value across all valid JSON found.
      - "multiple": more than one *distinct* string answer value found — still
        resolves to the last one, but flags that the model's JSON disagreed with
        itself somewhere in the response.
    """
    all_candidates: list[tuple[int, object]] = []
    for start, end in _iter_json_object_spans(response_text):
        try:
            obj = json.loads(response_text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            all_candidates.append((start, obj["answer"]))

    if not all_candidates:
        return ParsedAnswer(parsed_answer=None, parse_status="not_found")

    string_candidates = [(start, value) for start, value in all_candidates if isinstance(value, str) and value != ""]
    if not string_candidates:
        return ParsedAnswer(parsed_answer=None, parse_status="malformed")

    last_value = string_candidates[-1][1]
    distinct_values = {value for _, value in string_candidates}
    status: Literal["ok", "multiple"] = "ok" if len(distinct_values) == 1 else "multiple"
    return ParsedAnswer(parsed_answer=last_value, parse_status=status)
