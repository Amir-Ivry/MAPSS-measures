import numpy as np
from utils import safe_corr_np, hungarian

def test_safe_corr_np_basic():
    a = np.array([1,2,3,4,5], dtype=float)
    b = np.array([2,4,6,8,10], dtype=float)
    r = safe_corr_np(a, b)
    assert 0.99 < r <= 1.0

def test_hungarian_square():
    C = np.array([[1, 2],[2, 1]], dtype=float)
    r, c = hungarian(C)
    assert set(zip(r.tolist(), c.tolist())) in [{(0,0),(1,1)}, {(0,1),(1,0)}]
