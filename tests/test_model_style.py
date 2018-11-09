"""Test model integrity"""
from nanite import model


NAME_MAPPING = {
    "alpha": ["Face Angle", "Half Cone Angle"],
    "baseline": ["Force Baseline"],
    "contact_point": ["Contact Point"],
    "E": ["Young's Modulus"],
    "nu": ["Poisson's Ratio"],
    "R": "Tip Radius",
    }

UNIT_MAPPING = {
    "alpha": "°",
    "baseline": "N",
    "contact_point": "m",
    "E": "Pa",
    "nu": "",
    "R": "m",
    }


def test_model_parameter_name_order():
    for key in model.models_available:
        md = model.models_available[key]
        std = md.get_parameter_defaults()
        for n1, n2 in zip(list(std.keys()), md.parameter_keys):
            assert n1 == n2, "Parameter defaults probably in wrong order!"


def test_model_parameter_names():
    for key in model.models_available:
        md = model.models_available[key]
        for key, nn in zip(md.parameter_keys, md.parameter_names):
            if key in NAME_MAPPING:
                assert nn in NAME_MAPPING[key], "bad {} in {}".format(key, md)
            else:
                msg = "Parameter {} not registered for test!".format(key)
                assert False, msg


def test_model_parameter_units():
    for key in model.models_available:
        md = model.models_available[key]
        for key, un in zip(md.parameter_keys, md.parameter_units):
            if key in UNIT_MAPPING:
                assert UNIT_MAPPING[key] == un, "bad {} in {}".format(key, md)
            else:
                msg = "Parameter {} not registered for test!".format(key)
                assert False, msg


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
