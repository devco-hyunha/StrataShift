"""p6-03 AC3 gate: imatrix stats writer→loader round-trip against the consumer contract.

Normalization transcribed from `quantize.cpp:204-215` (count>0 → sum2/count,
count==0 → 1). Synthetic stats (accumulator output), no model files.
"""
from __future__ import annotations

import numpy as np

from stratashift.cli.main import StreamingGGUFWriter
from stratashift.meta.imatrix import add_activation_rows, load_imatrix_stats, new_stats

BASE = "blk.0.ffn_gate_exps.weight"


def _emit_stats(tmp: str) -> str:
    """Emit a stats GGUF from synthetic accumulator data via our writer.

    The in_sum2 plane is stored in ggml column-major order: tensor shape [ne0, nmat]
    lays element (i, j) at data[i + j*ne0], so the flat buffer is matrix 0 row, matrix
    1 row, ... — exactly `stats["in_sum2"].tobytes()` for a (nmat, ne0) array.
    """
    stats = new_stats(ne0=4, nmat=3)
    add_activation_rows(stats, 0, np.array([[1.0, 2.0, 0.0, -1.0], [2.0, 0.0, 1.0, 1.0]]))
    add_activation_rows(stats, 2, np.ones((2, 4)))  # expert 1 never touched
    w = StreamingGGUFWriter(tmp_path=tmp + ".tmp")
    w.add("general.architecture", "imatrix")
    w.add("general.type", "imatrix")
    w.add_tensor(BASE + ".in_sum2", [4, 3], "F32", stats["in_sum2"].astype("<f4").tobytes())
    w.add_tensor(BASE + ".counts", [1, 3], "F32", stats["counts"].astype("<f4").tobytes())
    return str(w.write(tmp))


def test_roundtrip_normalization(tmp_path):
    """count>0 → sum2/count; count==0 → ones (quantize.cpp:204-215 verbatim)."""
    path = _emit_stats(str(tmp_path / "stats.gguf"))
    stats = load_imatrix_stats(path)
    vecs = stats[BASE]
    assert vecs.shape == (3, 4)
    # matrix 0: sum2 [5,4,1,2] / count 1 = mean of squares
    assert np.allclose(vecs[0], [5.0, 4.0, 1.0, 2.0])
    # matrix 1: untouched → ones
    assert np.allclose(vecs[1], np.ones(4))
    # matrix 2: 2 ones rows → sum2 [2,2,2,2], count 1
    assert np.allclose(vecs[2], [2.0, 2.0, 2.0, 2.0])
