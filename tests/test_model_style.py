"""Test model integrity"""
from nanite import model


NAME_MAPPING = {
    "alpha": ["Face Angle", "Half Cone Angle"],
    "baseline": ["Force Baseline"],
    "contact_point": ["Contact Point"],
    "E": ["Young's Modulus"],
    "E_S": ["Sample Young's Modulus"],
    "E_L": ["Layer Young's Modulus"],
    "nu": ["Poisson's Ratio"],
    "nu_S": ["Sample Poisson's Ratio"],
    "nu_L": ["Layer Poisson's Ratio"],
    "R": "Tip Radius",
    "t": "Layer Thickness"
}

UNIT_MAPPING = {
    "alpha": "°",
    "baseline": "N",
    "contact_point": "m",
    "E": "Pa",
    "E_S": "Pa",
    "E_L": "Pa",
    "nu": "",
    "nu_S": "",
    "nu_L": "",
    "R": "m",
    "t": "m",
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
        for key2, nn in zip(md.parameter_keys, md.parameter_names):
            if key2 in NAME_MAPPING:
                assert nn in NAME_MAPPING[key2], "bad {} in {}".format(
                    key2, md)
            else:
                msg = "Parameter {} not registered for test!".format(key2)
                assert False, msg


def test_model_parameter_units():
    for key in model.models_available:
        md = model.models_available[key]
        for key2, un in zip(md.parameter_keys, md.parameter_units):
            if key2 in UNIT_MAPPING:
                assert UNIT_MAPPING[key2] == un, "bad {} in {}".format(
                    key2, md)
            else:
                msg = "Parameter {} not registered for test!".format(key2)
                assert False, msg
