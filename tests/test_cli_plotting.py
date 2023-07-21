import pathlib
import tempfile

from nanite.cli import plotting
from nanite import load_group

data_path = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"


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
