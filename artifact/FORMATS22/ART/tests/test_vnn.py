import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from art.acas import AcasNetID
from art.vnn import VNN20Info


def test_vnn_results():
    """ Validate the correctness of loaded results by checking some samples. """
    info = VNN20Info()

    def _check(prop: int, net_id: AcasNetID, for_all: bool, label: str, times: List):
        _label, _times, _ = info.query(prop, net_id, for_all)
        assert _label == label
        assert len(_times) == len(times)
        for lhs, rhs in zip(_times, times):
            if isinstance(rhs, float):
                assert float(lhs) == rhs
            elif isinstance(rhs, int):
                assert int(lhs) == rhs
            else:
                assert lhs == rhs
        return

    _check(1, AcasNetID(1, 1), True, 'UNSAT', [0.51, '-', 18.58, '-', 1.32, 0.62])
    _check(2, AcasNetID(3, 2), True, 'SAT', [0.23, '-', 18.5, '-', 0.82, 8.02])
    _check(4, AcasNetID(1, 7), True, 'SAT', [0.16, 2.03, 164.72, 0.07, 0.2982, 1.44])
    _check(4, AcasNetID(3, 2), True, 'UNSAT', [0.24, 25.68, 179.99, 0.24, 65.56, 0.96])
    _check(10, AcasNetID(4, 5), True, 'UNSAT', [0.7, '-', '-', '-', 130.63, 2.07])
    _check(1, AcasNetID(4, 6), False, 'UNSAT', [5.3, '-', 3191.34, '-', 179.98, 5.38])
    _check(2, AcasNetID(3, 3), False, 'UNSAT', [7.46, '-', '-', '-', 294.53, 167])
    _check(9, AcasNetID(3, 3), False, 'UNSAT', [2.52, 13326.19, 1121.38, '-', 1795.17, 9.21])
    return
