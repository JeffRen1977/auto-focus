"""
Pytest tests for the Autofocus system.

Covers PDAF, CDAF, configuration, and the hybrid autofocus system.
Uses synthetic images so no external data files are needed.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import AFConfig, AFMethod, AFState, ROIData
from core.pdaf import PDAFProcessor
from core.cdaf import CDAFProcessor
from core.autofocus_system import AutoFocusSystem


def make_roi(x=10, y=10, width=40, height=40):
    return ROIData(x=x, y=y, width=width, height=height, confidence=1.0, method="center")


def make_synthetic_pd_pair(height=64, width=128, shift=2):
    """Create synthetic left/right PD images with known disparity."""
    left = np.random.randint(50, 200, (height, width), dtype=np.uint16)
    right = np.roll(left, shift, axis=1)
    return left, right


def make_synthetic_bayer(height=128, width=128):
    """Create a synthetic Bayer image with some texture."""
    img = np.zeros((height, width), dtype=np.uint16)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            val = np.random.randint(100, 900)
            img[i : i + 8, j : j + 8] = val
    return img


class TestAFConfig:
    def test_default_config(self):
        config = AFConfig()
        assert config.pdaf_block_size > 0
        assert config.pdaf_search_range > 0
        assert 0.0 <= config.pdaf_min_confidence <= 1.0

    def test_custom_config(self):
        config = AFConfig(pdaf_block_size=16, pdaf_search_range=32)
        assert config.pdaf_block_size == 16
        assert config.pdaf_search_range == 32


class TestAFMethod:
    def test_methods_exist(self):
        assert AFMethod.PDAF.value == "pdaf"
        assert AFMethod.CDAF.value == "cdaf"
        assert AFMethod.HYBRID.value == "hybrid"


class TestAFState:
    def test_states_exist(self):
        assert AFState.IDLE.value == "idle"
        assert AFState.SEARCHING.value == "searching"
        assert AFState.CONVERGED.value == "converged"
        assert AFState.FAILED.value == "failed"


class TestPDAFProcessor:
    def test_init(self):
        config = AFConfig()
        pdaf = PDAFProcessor(config)
        assert pdaf is not None

    def test_process_pdaf_with_synthetic_data(self):
        config = AFConfig()
        pdaf = PDAFProcessor(config)
        left, right = make_synthetic_pd_pair()
        roi = make_roi()
        result = pdaf.process_pdaf(left, right, roi)
        assert result is not None
        assert hasattr(result, "confidence") or isinstance(result, dict)


class TestCDAFProcessor:
    def test_init(self):
        config = AFConfig()
        cdaf = CDAFProcessor(config)
        assert cdaf is not None

    def test_process_cdaf_with_synthetic_data(self):
        config = AFConfig()
        cdaf = CDAFProcessor(config)
        bayer = make_synthetic_bayer()
        roi = make_roi(x=10, y=10, width=64, height=64)
        result = cdaf.process_cdaf(bayer, roi, current_position=500.0, step_size=20)
        assert result is not None


class TestAutoFocusSystem:
    def test_init(self):
        config = AFConfig()
        af = AutoFocusSystem(config)
        assert af is not None

    def test_trigger_pdaf(self):
        config = AFConfig()
        af = AutoFocusSystem(config)
        left, right = make_synthetic_pd_pair()
        bayer = make_synthetic_bayer()
        result = af.trigger_autofocus(
            image=bayer,
            left_pd=left,
            right_pd=right,
            method=AFMethod.PDAF,
        )
        assert result is not None

    def test_trigger_cdaf(self):
        config = AFConfig()
        af = AutoFocusSystem(config)
        bayer = make_synthetic_bayer()
        result = af.trigger_autofocus(
            image=bayer,
            method=AFMethod.CDAF,
        )
        assert result is not None
