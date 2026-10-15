import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shapely.geometry import Polygon

from analyse_hazard import _site_for_zone, magnitude_return_period
from analyse_hazard_fixed_rp import amplitude_at_return_period


def test_amplitude_at_return_period_interpolates_log_rate_and_log_amplitude():
    levels = np.log([1.0, 2.0, 4.0, 8.0])
    rates = np.array([0.2, 0.1, 0.01, 0.0])

    return_period = 1 / np.sqrt(0.1 * 0.01)
    amplitude = amplitude_at_return_period(levels, rates, return_period)

    assert amplitude == pytest.approx(2.0 * np.sqrt(2.0))


def test_amplitude_at_return_period_returns_none_outside_curve_range():
    levels = np.log([1.0, 2.0, 4.0])
    rates = np.array([0.2, 0.1, 0.01])

    assert amplitude_at_return_period(levels, rates, return_period=1000) is None


def test_centroid_edge_uses_centroid_for_convex_zone():
    zone = Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])

    site, placement = _site_for_zone(zone, "centroid-edge")

    assert placement == "centroid"
    assert site.equals(zone.centroid)


def test_centroid_edge_falls_back_only_when_centroid_is_outside():
    zone = Polygon([
        (0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4),
    ])
    assert not zone.contains(zone.centroid)

    site, placement = _site_for_zone(zone, "centroid-edge")

    assert placement == "edge"
    assert zone.exterior.distance(site) == pytest.approx(0)
    with pytest.raises(ValueError, match="centroid lies outside"):
        _site_for_zone(zone, "centroid")


@pytest.mark.parametrize("a_val, b_val", [(4.0, 0.9), (3.2, 1.12), (5.1, 0.61)])
def test_magnitude_return_period_inverts_the_eshm20_branch(a_val, b_val):
    """Feeding m_RP back in has to return RP, for any branch.

    The magnitude domain reads its levels off the published a-value and
    b-value, so the map back to a return period has to use the same two.
    Decaying with a refitted b-value instead leaves this round trip anywhere
    between a tenth and ten times the return period asked for.
    """
    for return_period in (25.0, 250.0, 2475.0):
        m_rp = (a_val + np.log10(return_period)) / b_val

        assert magnitude_return_period(m_rp, a_val, b_val) == pytest.approx(
            return_period
        )
