"""Test of basic opening functionalities"""
import pathlib
import shutil

from nanite.read import read_jpk, read_jpk_meta


dp = pathlib.Path(__file__).parent / "data"
jpkfile = dp / "spot3-0192.jpk-force"


def test_open_jpk_simple():
    tdir = read_jpk_meta.extract_jpk(jpkfile)
    segroot = tdir / "segments"
    seg = sorted(segroot.glob("[0-1]"))
    chan_data = {}
    chroot = seg[0] / "channels"
    for ch in chroot.glob("*.dat"):
        key = ch.stem
        chan_data[key] = read_jpk.load_dat_raw(ch)
    assert chan_data["height"][0] == 50.425720226584403
    shutil.rmtree(tdir, ignore_errors=True)


def test_open_jpk_calibration():
    cf = dp / "calibration_force-save-2015.02.04-11.25.21.294.jpk-force"
    try:
        read_jpk.load_jpk(cf)
    except read_jpk_meta.ReadJPKMetaKeyError:
        pass
    else:
        assert False, "no spring constant should raise error"


def test_open_jpk_conversion():
    tdir = read_jpk_meta.extract_jpk(jpkfile)
    segroot = tdir / "segments"
    seg = sorted(segroot.glob("[0-1]"))
    chan_data = {}
    chroot = segroot / seg[0] / "channels"
    for ch in chroot.glob("*.dat"):
        key = ch.stem
        chan_data[key] = read_jpk.load_dat_unit(ch)

    assert chan_data["vDeflection"][2] == "vDeflection (Force)"
    assert chan_data["vDeflection"][1] == "N"
    assert chan_data["vDeflection"][0][0] == -5.145579192349918e-10
    assert chan_data["height"][0][0] == 2.8783223430683289e-05
    assert chan_data["strainGaugeHeight"][0][0] == 2.2815672438768612e-05
    shutil.rmtree(tdir, ignore_errors=True)


def test_get_single_curves():
    read_jpk.load_jpk_single_curve(jpkfile, column="vDeflection", slot="force")
    read_jpk.load_jpk_single_curve(jpkfile, column="height")
    read_jpk.load_jpk_single_curve(
        jpkfile, column="strainGaugeHeight", slot="nominal")
    read_jpk.load_jpk_single_curve(jpkfile, column="height", slot="volts")
    read_jpk.load_jpk_single_curve(
        jpkfile, segment=1, column="height", slot="volts")
    # This is height in the txt files
    a = read_jpk.load_jpk_single_curve(
        jpkfile, column="height", slot="nominal")
    assert a[0][0] == 4.9574279773415606e-05


def test_meta():
    md = read_jpk_meta.get_meta_data(jpkfile)
    assert md["spring constant [N/m]"] == 0.043493666407368466


def test_load_jpk():
    segs = read_jpk.load_jpk(jpkfile)
    md = read_jpk_meta.get_meta_data(jpkfile)
    assert md["curve type"] == "extend-retract"
    assert len(segs) == 1, "Only one measurement"
    assert len(segs[0]) == 2, "approach and retract curves"
    ap, ret = segs[0]
    assert ap[1]["curve type"] == "extend"
    assert ret[1]["curve type"] == "retract"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
