"""Test qmap feature"""
import pathlib
import warnings

import numpy as np

from nanite import qmap


datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "map2x2_extracted.jpk-force-map"


def test_metadata():
    qm = qmap.QMap(jpkfile)
    assert np.allclose(qm.extent,
                       [1.97265625, 601.97265625,
                        -783.53515625, -183.53515625000006])
    assert qm.shape == (10, 10)


def test_feat_scan_order():
    qm = qmap.QMap(jpkfile)
    order = qm.get_qmap("meta scan order", qmap_only=True)
    assert order[0, 0] == 0
    assert order[0, -1] == 1
    assert order[-1, -1] == 2
    assert order[-1, 0] == 3
    assert np.isnan(order[0, 1])


def test_feat_min_height():
    qm = qmap.QMap(jpkfile)
    qd = qm.get_qmap("data min height", qmap_only=True)
    assert np.allclose(qd[0, 0], 40.55030392499141)
    assert np.allclose(qd[0, -1], 47.354988549298945)
    assert np.allclose(qd[-1, -1], 96.1627883099352)
    assert np.allclose(qd[-1, 0], 89.95170867840217)


def test_feat_emod_nofit():
    qm = qmap.QMap(jpkfile)
    with warnings.catch_warnings(record=True) as w:
        # No data availabale, because there is no fit
        qd = qm.get_qmap("fit young's modulus", qmap_only=True)
        assert len(w) == 4
        assert w[0].category is qmap.DataMissingWarning
    assert np.alltrue(np.isnan(qd))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
