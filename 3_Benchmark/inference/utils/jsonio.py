from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple


def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def write_json_atomic(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, output_path)


def read_jsonl_records(output_path: Path) -> Tuple[Optional[List[Any]], Optional[str]]:
    decoder = json.JSONDecoder()
    records: List[Any] = []
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                text = raw_line.strip()
                if not text:
                    continue
                idx = 0
                length = len(text)
                while idx < length:
                    while idx < length and text[idx].isspace():
                        idx += 1
                    if idx >= length:
                        break
                    try:
                        record, offset = decoder.raw_decode(text, idx)
                    except json.JSONDecodeError as exc:
                        context_start = max(0, exc.pos - 32)
                        context_end = min(length, exc.pos + 32)
                        snippet = text[context_start:context_end]
                        return None, (
                            f"invalid JSON at line {line_no}: {exc.msg} (pos={exc.pos}); context='{snippet}'"
                        )
                    records.append(record)
                    idx = offset
    except OSError as exc:
        return None, str(exc)
    return records, None
