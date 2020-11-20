"""Test of model basics"""
import warnings

import nanite
import nanite.model


class MockModel:
    def __init__(self, model_key, **kwargs):
        # rebase on hertz model
        md = nanite.model.models_available["hertz_para"]
        for akey in dir(md):
            setattr(self, akey, getattr(md, akey))
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
        self.model_key = model_key

    def __enter__(self):
        nanite.model.register_model(self, self.__repr__())

    def __exit__(self, a, b, c):
        nanite.model.models_available.pop(self.model_key)


def test_bad_parameter_order():
    swapped_keys = ["R", "E", "nu", "contact_point", "baseline"]
    mod = MockModel("test_bad_order", parameter_keys=swapped_keys)
    try:
        mod.__enter__()
    except nanite.model.ModelImplementationError:
        pass
    else:
        assert False, "Bad parameter order should not be possible"
    finally:
        try:
            mod.__exit__(None, None, None)
        except KeyError:
            pass


def test_bad_parameter_number():
    short_keys = ["E", "R", "nu", "contact_point"]
    mod = MockModel("test_bad_number", parameter_keys=short_keys)
    try:
        mod.__enter__()
    except nanite.model.ModelImplementationError:
        pass
    else:
        assert False, "Bad parameter number should not be possible"
    finally:
        try:
            mod.__exit__(None, None, None)
        except KeyError:
            pass


def test_bad_parameter_units_number():
    short_units = ["Pa", "m", "", "m"]
    mod = MockModel("test_bad_units_number", parameter_units=short_units)
    try:
        mod.__enter__()
    except nanite.model.ModelImplementationError:
        pass
    else:
        assert False, "Bad parameter units number should not be possible"
    finally:
        try:
            mod.__exit__(None, None, None)
        except KeyError:
            pass


def test_bad_model_func_args():
    with warnings.catch_warnings():
        warnings.simplefilter("error", nanite.model.ModelImplementationWarning)

        def bad_func(E, delta, R, nu, contact_point=0, baseline=0):
            return delta
        mod = MockModel("test_bad_func", model_func=bad_func)
        try:
            mod.__enter__()
        except nanite.model.ModelImplementationWarning:
            pass
        else:
            assert False, "Bad parameter order should not be possible"
        finally:
            try:
                mod.__exit__(None, None, None)
            except KeyError:
                pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
