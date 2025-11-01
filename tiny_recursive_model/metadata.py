import json
from pathlib import Path

import pydantic


class PuzzleMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: int | None
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: list[str]


def load_metadata(path: Path) -> PuzzleMetadata:
    metadata_file = path / "dataset.json"
    return PuzzleMetadata(**json.loads(metadata_file.read_text()))
