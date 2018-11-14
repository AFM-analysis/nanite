import pathlib
import tempfile

from nanite.cli import plotting
from nanite import load_group

datadir = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = datadir / "spot3-0192.jpk-force"


def test_plotting():
    # use temporary file
    _, name = tempfile.mkstemp(suffix=".png", prefix="test_nanite_plotting_")
    name = pathlib.Path(name)

    grp = load_group(jpkfile)
    idnt = grp[0]
    idnt.apply_preprocessing(["compute_tip_position",
                              "correct_force_offset"])
    idnt.fit_model(model_key="hertz_para",
                   params_initial=None,
                   x_axis="tip position",
                   y_axis="force",
                   weight_cp=False,
                   segment="retract")
    plotting.plot_data(idnt=idnt, path=name)
    assert name.stat().st_size > 90000

    try:
        name.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
