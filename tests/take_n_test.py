import pytest
import sys

sys.path.append("./utils")

from take_n import *


def test_it_works():
    data = [1,2,3,4,5]
    out  = list(take_n(data, 1))

    assert [0, 1, 2, 3, 4] == out

def test_it_uses_condition_to_filter_items():
    data = [1,2,1,4,5]
    out  = list(take_n(data, 2, lambda x: x < 3))

    assert [0, 1, 2] == out

def test_all_ones():
    data = [1,1,1,1,1]
    out  = list(take_n(data, 5, lambda x: x == 1))

    assert [0, 1, 2, 3, 4] == out

def test_all_ones_and_twos():
    data = [1,2,1,2,1,2,1,2,1,2]
    out  = list(take_n(data, 5, lambda x: x == 1 or x == 2))

    assert [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] == out

def test_all_ones_and_twos_and_threes():
    data = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
    keeplist = {1: 0, 2: 0, 3: 0}
    out  = list(take_n(data, 5, lambda x: x > 0 and x < 4))

    assert [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] == out
