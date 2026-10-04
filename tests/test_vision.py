import json
import types
from pathlib import Path
from unittest import mock

import pytest

from app.pipeline import vision

BATCH = [(0.5, Path("a")), (1.5, Path("b")), (2.5, Path("c"))]
BOX = {"banner": True, "confidence": 0.9, "bbox": [100, 100, 900, 200]}


def run(frames):
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=json.dumps({"frames": frames})))])
    with mock.patch.object(Path, "read_bytes", return_value=b"jpg"), \
         mock.patch.object(vision, "chat", return_value=resp):
        return [d.t for d in vision._detect_batch(BATCH) if d.banner]


@pytest.mark.parametrize("frames, expected", [
    ([{"frame": 1, **BOX}, {"frame": 2}, {"frame": 3}], [0.5]),
    ([{"frame": "frame 1", **BOX}, {"frame": "frame 2"}, {"frame": "frame 3"}], [0.5]),  # строкой — падало
    ([{"frame": None, **BOX}, {"frame": None}, {"frame": None}], [0.5]),  # null — падало
    ([{"frame": 0, **BOX}, {"frame": 1}, {"frame": 2}], [0.5]),  # с нуля — баннер терялся
    ([{"frame": 3, **BOX}, {"frame": 1}, {"frame": 2}], [2.5]),  # не по порядку — по номеру
    ([{"frame": 2, **BOX}], [1.5]),  # пропущенные кадры — без баннера
    (["мусор", {"frame": 1, **BOX}], [0.5]),  # не-объекты игнорируются
])
def test_frame_numbering_variants(frames, expected):
    assert run(frames) == expected
