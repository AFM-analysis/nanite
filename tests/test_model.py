"""Test of model basics"""
import warnings

import nanite
import nanite.model

from common import MockModelModule


def test_bad_parameter_order():
    swapped_keys = ["R", "E", "nu", "contact_point", "baseline"]
    mod = MockModelModule("test_bad_order", parameter_keys=swapped_keys)
    try:
        mod.__enter__()
    except nanite.model.core.ModelImplementationError:
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
    mod = MockModelModule("test_bad_number", parameter_keys=short_keys)
    try:
        mod.__enter__()
    except nanite.model.core.ModelImplementationError:
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
    mod = MockModelModule("test_bad_units_number", parameter_units=short_units)
    try:
        mod.__enter__()
    except nanite.model.core.ModelImplementationError:
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
        warnings.simplefilter("error",
                              nanite.model.core.ModelImplementationWarning)

        def bad_func(E, delta, R, nu, contact_point=0, baseline=0):
            return delta
        mod = MockModelModule("test_bad_func", model_func=bad_func)
        try:
            mod.__enter__()
        except nanite.model.core.ModelImplementationWarning:
            pass
        else:
            assert False, "Bad parameter order should not be possible"
        finally:
            try:
                mod.__exit__(None, None, None)
            except KeyError:
                pass
