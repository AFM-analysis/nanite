import argparse
import pathlib

import appdirs
import numpy as np

from .. import model
from .. import preproc
from .. import rate

APP_DIR = pathlib.Path(appdirs.user_config_dir(appname="nanite"))
PROFILE_PATH = APP_DIR / "cli_profile.cfg"

DEFAULTS = {"model_key": "sneddon_spher_approx",
            "preprocessing": ["compute_tip_position",
                              "correct_force_offset",
                              "correct_tip_offset"],
            "range_type": "absolute",
            "range_x": [0, 0],
            "segment": "approach",
            "weight_cp": 5e-7,
            "rating regressor": "Extra Trees",
            "rating training set": "zef18",
            }


class Profile(object):
    def __init__(self, path=PROFILE_PATH, create=True):
        """Initialize settings file (create if it does not exist)"""
        path = pathlib.Path(path)
        if create:
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch()
        elif not path.exists():
            raise ValueError("Please run `nanite-setup-profile` first!")
        self.path = path
        # initialize with defaults
        for key in DEFAULTS:
            self[key]

    def __getitem__(self, key):
        default = DEFAULTS[key]
        data = self.load()

        if key in data:
            if isinstance(default, str):
                val = data[key]
            elif isinstance(default, list):
                val = data[key].split(",")
                if isfloat(val[0]) and isfloat(val[1]):
                    val = [float(vv) for vv in val]
            else:
                val = float(data[key])
        else:
            val = default
        # also set default
        self[key] = val
        return val

    def __setitem__(self, key, value):
        if key.startswith("fit param"):
            if not (key.endswith("value")
                    or key.endswith("vary")):
                raise ValueError("Invalid key: '{}'".format(key))
            default = None
        else:
            default = DEFAULTS[key]
        data = self.load()
        if isinstance(default, list):
            # convert items to string
            value = ["{}".format(vv) for vv in value]
            # convert list to string
            value = ",".join(value)
        data[key] = value
        self.save(data)

    def set_fit_params(self, params):
        for p in params:
            self["fit param {} value".format(p)] = params[p].value
            self["fit param {} vary".format(p)] = params[p].vary

    def get_fit_params(self):
        cdict = self.load()
        # get model key
        model_key = cdict["model_key"]
        default = model.get_init_parms(model_key)
        for p in default:
            vkey = "fit param {} value".format(p)
            if vkey in cdict:
                default[p].value = float(cdict[vkey])
            fkey = "fit param {} vary".format(p)
            if fkey in cdict:
                fval = cdict[fkey]
                if fval.lower() == "true":
                    default[p].vary = True
                else:
                    default[p].vary = False
        # write
        self.set_fit_params(default)
        return default

    def load(self):
        """Loads the profile file returning a dictionary"""
        with self.path.open() as fop:
            fc = fop.readlines()
        cdict = {}
        for line in fc:
            line = line.strip()
            var, val = line.split("=", 1)
            cdict[var.strip()] = val.strip()
        return cdict

    def save(self, cdict):
        """Save a settings dictionary into a file"""
        skeys = list(cdict.keys())
        skeys.sort()
        outlist = []
        for sk in skeys:
            sval = cdict[sk]
            outlist.append(u"{} = {}\n".format(sk, sval))

        with self.path.open('w') as fop:
            fop.writelines(outlist)


def setup_profile():
    """Help user to create a fitting profile"""
    parser = setup_profile_parser()
    parser.parse_args()

    pf = Profile()

    print("\nDefine preprocessing:")
    steps = preproc.available_preprocessors
    cur = pf["preprocessing"]
    curid = ",".join([str(steps.index(cc) + 1) for cc in cur])
    for ii, st in enumerate(steps):
        print("  {}: {}".format(ii+1, st))
    stp = input("(currently '{}'): ".format(curid))
    if stp:
        pf["preprocessing"] = [steps[int(ii) - 1] for ii in stp.split(",")]

    print("\nSelect model number:")
    models = sorted(model.models_available.keys())
    idx = models.index(pf["model_key"])
    for ii, mm in enumerate(models):
        print("  {}: {}".format(ii+1, mm))
    mod = input("(currently '{}'): ".format(idx + 1))
    if mod:
        newmod = models[int(mod) - 1]
        if newmod != pf["model_key"]:
            pf["model_key"] = newmod
            pf["params_initial"] = ""

    print("\nSet fit parameters:")
    params = pf.get_fit_params()
    usedmod = model.models_available[pf["model_key"]]
    for ii, p in enumerate(params):
        unit = usedmod.parameter_units[ii]
        if unit:
            unit = " [{}]".format(unit)
        value = input(
            "- initial value for {}{} (currently '{}'): ".format(
                p, unit, params[p].value))
        if value:
            params[p].value = float(value)
        while True:
            vary = input(
                "  vary {} (currently '{}'): ".format(p, params[p].vary))
            if vary:
                if vary.strip().lower() == "true":
                    vary = True
                elif vary.strip().lower() == "false":
                    vary = False
                else:
                    print("Please type 'true' or 'false'.")
                    continue
                params[p].vary = vary
            break
        pf.set_fit_params(params)

    print("\nSelect range type (absolute or relative):")
    while True:
        rt = input("(currently '{}'): ".format(pf["range_type"]))
        if rt:
            if rt not in ["absolute", "relative"]:
                print("Please choose 'absolute' or 'relative'.")
                continue
            pf["range_type"] = rt
        break

    print("\nSelect fitting interval:")
    ival = np.array(pf["range_x"]) * 1e6
    left = input("left [µm] (currently '{}'): ".format(ival[0]))
    if left:
        ival[0] = float(left)
    right = input("right [µm] (currently '{}'): ".format(ival[1]))
    if left:
        ival[1] = float(right)
    pf["range_x"] = list(ival*1e-6)

    print("\nSuppress residuals near contact point:")
    wcpd = pf["weight_cp"] * 1e6
    wcp = input("size [µm] (currently '{}'): ".format(wcpd))
    if wcp:
        pf["weight_cp"] = float(wcp) * 1e-6

    print("\nSelect training set:")
    cts = pf["rating training set"]
    while True:
        ts = input("training set (path or label) "
                   + "(currently '{}'): ".format(cts))
        if ts:
            # Test whether the training set is implemented in nanite or
            # whether the data exist and are complete.
            ir = rate.IndentationRater
            if not pathlib.Path(ts).exists():
                pp = ir.get_training_set_path(ts)
            else:
                pp = ts
            try:
                ir.load_training_set(path=pp)
            except OSError:
                print("No training set found for '{}'!".format(ts))
                continue
            else:
                pf["rating training set"] = ts
                break
        else:
            break

    print("\nSelect rating regressor:")
    regs = rate.reg_names
    rcur = pf["rating regressor"]
    rcurid = regs.index(rcur) + 1
    for ii, st in enumerate(regs):
        print("  {}: {}".format(ii+1, st))
    rstp = input("(currently '{}'): ".format(rcurid))
    if rstp:
        pf["rating regressor"] = regs[int(rstp) - 1]

    print("\nDone. You may edit all parameters in '{}' ".format(pf.path)
          + "or alternatively run nanite-setup-profile again.")


def setup_profile_parser():
    descr = "Set up a profile for fitting and rating. The profile is stored " \
            + "in the user's default configuration directory. Setting up a " \
            + "profile is required prior to running `nanite-fit` and " \
            + "`nanite-rate`."
    parser = argparse.ArgumentParser(description=descr)
    return parser


def isfloat(value):
    try:
        value = float(value)
    except ValueError:
        return False
    else:
        return True
