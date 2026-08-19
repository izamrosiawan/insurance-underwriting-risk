import pytest
import pandas as pd
import numpy as np
from src.underwriting_engine import UnderwritingEngine

def test_underwriting_decision_range():
    engine = UnderwritingEngine()
    sample = pd.read_csv('data/train.csv', nrows=5)
    decisions = engine.assess_risk(sample)
    
    assert len(decisions) == 5
    assert np.all(decisions >= 1)
    assert np.all(decisions <= 8)
