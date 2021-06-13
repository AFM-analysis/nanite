import pathlib
import shutil
import tempfile

import nanite.read


data_dir = pathlib.Path(__file__).parent / "data"


def test_recursive_callback():
    # set up a directory with recursive data
    td = pathlib.Path(tempfile.mkdtemp(prefix="nanite_recursive_load_"))
    td2 = td / "data1"
    td2.mkdir(parents=True)
    td3 = td / "data3" / "data4"
    td3.mkdir(parents=True)

    shutil.copy2(data_dir / "spot3-0192.jpk-force", td2 / "spot1.jpk-force")
    shutil.copy2(data_dir / "spot3-0192.jpk-force", td2 / "spot2.jpk-force")
    shutil.copy2(data_dir / "spot3-0192.jpk-force", td3 / "spot3.jpk-force")

    # trace the callback calls
    calls = []

    def mycallback(value):
        calls.append(value)

    files = nanite.read.load_data(td, callback=mycallback)
    assert len(files) == 3

    assert calls == [1/3, 2/3, 1]
