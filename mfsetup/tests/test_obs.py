import numpy as np
import pandas as pd
import pytest
from ..obs import make_obsname


def test_make_obsname():
    maxlen = 13
    names = ['345612091522401',
             '345612091522401',
             '345612091522401',
             '445612091522401',
             '345612091522401',
]
    expected = [names[0][-maxlen:],
                names[1][-maxlen-1:-1],
                names[2][-maxlen-2:-2],
                names[3][-maxlen-2:-2],
                names[0][-maxlen:]
                ]
    unique_names = set()
    for i, n in enumerate(names):
        result = make_obsname(n, unique_names, maxlen=maxlen)
        unique_names.add(result)
        assert len(result) <= maxlen
        assert result == expected[i]
