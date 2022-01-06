import pathlib

import nanite


data_dir = pathlib.Path(__file__).parent / "data"


def test_load_model_from_file():
    mpath = data_dir / "model_external_basic.py"
    md = nanite.model.load_model_from_file(mpath, register=True)
    assert md.model_key == "hans_peter"
    assert md.model_key in nanite.model.models_available
    nanite.model.deregister_model(md)
    assert md.model_key not in nanite.model.models_available


def test_load_model_from_model():
    mpath = data_dir / "model_external_basic.py"
    md = nanite.model.load_model_from_file(mpath, register=False)
    assert md.model_key == "hans_peter"
    assert md.model_key not in nanite.model.models_available
    md2 = nanite.model.register_model(md)
    assert md is md2
    assert md.model_key in nanite.model.models_available
    nanite.model.deregister_model(md)
    assert md.model_key not in nanite.model.models_available
