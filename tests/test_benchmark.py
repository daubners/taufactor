import numpy as np
import pytest

from taufactor.benchmark import resolve_structure, run_benchmark_study
from taufactor.utils import create_stacked_blocks


def test_resolve_structure_predefined_name():
    cube, name = resolve_structure("blocks", N=12, features=1)
    assert name == "blocks"
    assert cube.shape == (12, 12, 12)


def test_resolve_structure_custom_hook():
    def my_structure(Nx, features=None):
        arr = np.zeros((Nx, Nx, Nx), dtype=int)
        arr[:, :, : Nx // 2] = 1
        return arr

    cube, name = resolve_structure(my_structure, N=10, features=3)
    assert name == "my_structure"
    assert cube.shape == (10, 10, 10)
    assert cube.dtype == int


def test_resolve_structure_rejects_invalid_input():
    with pytest.raises(TypeError):
        resolve_structure(123, N=10, features=1)


def test_run_benchmark_study_blocks_string_and_hook_match():
    def structure_fn(N):
        return create_stacked_blocks(N)

    res1 = run_benchmark_study(
        Ns=[16], solver="Solver", structure="blocks",
        write_file=False, devices=['cpu']
    )
    res2 = run_benchmark_study(
        Ns=[16], solver="Solver", structure=structure_fn,
        write_file=False, devices=['cpu']
    )
    assert np.isclose(res1[0]["taufactor"], res2[0]["taufactor"])
