import pytest
import numpy as np

@pytest.fixture
def _fI0OEdE():
    np.random.seed(42)
    return np.random.randn(20, 4)

@pytest.fixture
def _fIIIEdf():
    return {'price': 100.5, 'volume': 1000.0, 'bid': 100.45, 'ask': 100.55, 'symbol': 'TEST'}

@pytest.fixture
def _f11lEEO():
    return {'depth': 5000.0, 'rop': 50.0, 'wob': 20000.0, 'rpm': 120.0, 'torque': 15000.0, 'surprise': 1.5, 'mu': [0.1, 0.2, 0.3]}