"""Test of basic opening functionalities"""
from nanite import model


def test_model_parameter_name_order():
    for key in model.models_available:
        md = model.models_available[key]
        std = md.get_parameter_defaults()
        for n1, n2 in zip(list(std.keys()), md.parameter_keys):
            assert n1 == n2, "Parameter defaults probably in wrong order!"


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
