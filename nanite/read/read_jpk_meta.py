"""Methods to open JPK data files and to obtain meta data"""
import functools
import pathlib
import shutil
import tempfile
import zipfile

import jprops
import numpy as np


class ReadJPKMetaKeyError(BaseException):
    pass


def extract_jpk(path_jpk, props_only=False):
    """Extract the JPK data files and return the extracted path"""
    tdir = tempfile.mkdtemp(prefix="nanite_jpk_")
    with zipfile.ZipFile(str(path_jpk)) as fd:
        if props_only:
            for name in fd.namelist():
                if name.endswith(".properties"):
                    fd.extract(name, tdir)
        else:
            fd.extractall(tdir)
    return pathlib.Path(tdir)


def isnan(obj):
    if isinstance(obj, (str)):
        if obj.lower() == "nan":
            return True
    elif isinstance(obj, float):
        return np.isnan(obj)
    else:
        return False


def get_data_list(path_jpk):
    path_jpk = pathlib.Path(path_jpk)
    with zipfile.ZipFile(str(path_jpk)) as fd:
        files = fd.namelist()
    dfiles = [f for f in files if f.endswith(".dat")]
    # sort files correctly
    segdict = {}
    if "segments/" in files:
        app = [aa for aa in dfiles if aa.startswith("segments/0")]
        ret = [aa for aa in dfiles if aa.startswith("segments/1")]
        segdict["0"] = (app, ret)
    elif "index/" in files:
        segids = [dd.split("/")[1] for dd in dfiles]
        segids = list(set(segids))
        for sid in segids:
            sidstrapp = "index/{}/segments/0".format(sid)
            app = [dd for dd in dfiles if dd.startswith(sidstrapp)]
            sidstrret = "index/{}/segments/1".format(sid)
            ret = [dd for dd in dfiles if dd.startswith(sidstrret)]
            segdict[sid] = (app, ret)
    else:
        msg = "Unknown JPK file format: {}".format(path_jpk)
        raise NotImplementedError(msg)

    segkeys = list(segdict.keys())

    def sort_key(x): return int(x)
    segkeys.sort(key=sort_key)

    seglist = []
    for key in segkeys:
        seglist.append(segdict[key])
    return seglist


def get_meta_data(jpk_file):
    """Obtain meta-data from a jpk data file

    Parameters
    ----------
    path_jpk: str
        Path to a jpk data file (e.g. .jpk-force)

    Returns
    -------
    metadata: dict
        Dictionary containing meta-data.

    """
    jpk_file = pathlib.Path(jpk_file)
    tdir = extract_jpk(jpk_file, props_only=True)
    segs = []
    # global search for "segments" and approach (0) / retract (1)
    # to `segs`.
    for ss in sorted(tdir.rglob("segments")):
        segs.append(ss / "0")
        segs.append(ss / "1")
    # multiple keys in case of map data
    ignore_keys = ["position x [µm]",
                   "position y [µm]",
                   "position x [px]",
                   "position y [px]",
                   ]
    md = {}
    segs.sort()
    for seg in segs:
        mdi = get_meta_data_seg(seg)
        for k in list(mdi.keys()):
            if k in ignore_keys:
                continue
            elif k in md:
                if md[k] != mdi[k]:
                    if isnan(md[k]) and not isnan(mdi[k]):
                        md[k] = mdi[k]
                    elif isnan(mdi[k]):
                        pass
                    else:
                        if not isinstance(md[k], list):
                            md[k] = [md[k]]
                        md[k].append(mdi[k])
            else:
                md[k] = mdi[k]

    # join curve types
    md["curve type"] = "-".join(md["curve type"])
    md["file"] = jpk_file
    # cleanup
    shutil.rmtree(tdir, ignore_errors=True)
    return md


def get_meta_data_seg(path_segment):
    """Obtain most important meta-data for a segment folder
    """
    segment = pathlib.Path(path_segment)
    conv = "channel.vDeflection.conversion-set.conversion"
    sens_str = conv + ".distance.scaling.multiplier"
    sprc_str = conv + ".force.scaling.multiplier"
    pos_str = "force-segment-header.environment.xy-scanner-position-map"\
              + ".xy-scanner.motorstage.position."

    # These are properties that will be returned, if they exist in the header
    # files.
    dvars = [
        ["sensitivity [m/V]", sens_str],
        ["spring constant [N/m]", sprc_str],
        ["duration [s]", "force-segment-header.duration"],
        ["points", "force-segment-header.num-points"],
        ["feedback mode", "force-segment-header.settings.feedback-mode.name"],
        ["approach id", "force-segment-header.approach-id"],
        ["curve type", "force-segment-header.settings.style"],
        ["grid size x [px]", "force-scan-map.position-pattern.grid.ilength"],
        ["grid size y [px]", "force-scan-map.position-pattern.grid.jlength"],
    ]

    # These are properties that will not be returned, but are used for the
    # computation of properties.
    dvars_im = [
        ["z start", "force-segment-header.settings.segment-settings.z-start"],
        ["z end", "force-segment-header.settings.segment-settings.z-end"],
        # Setpoint is stored in Volts and needs to be converted still
        ["setpoint [V]", "force-segment-header.settings.segment-settings.setpoint"],
        ["position x [m]", pos_str + "x"],
        ["position y [m]", pos_str + "y"],
        ["grid size x [m]", "force-scan-map.position-pattern.grid.ulength"],
        ["grid size y [m]", "force-scan-map.position-pattern.grid.vlength"],
        ["grid center x [m]", "force-scan-map.position-pattern.grid.xcenter"],
        ["grid center y [m]", "force-scan-map.position-pattern.grid.ycenter"],
    ]

    md = {}
    md_im = {}
    header_file = segment / "segment-header.properties"
    prop = get_seg_head_prop(header_file)
    for mdi, dvarsi in [[md, dvars], [md_im, dvars_im]]:
        for name, var in dvarsi:
            if var in prop:
                mdi[name] = prop[var]

    for mkey in ["spring constant [N/m]",
                 "sensitivity [m/V]",
                 ]:
        if mkey not in md:
            msg = "Missing meta data: '{}'".format(mkey)
            raise ReadJPKMetaKeyError(msg)

    md["setpoint [N]"] = md_im["setpoint [V]"] * \
        md["spring constant [N/m]"]*md["sensitivity [m/V]"]
    md["data rate [Hz]"] = md["points"]/md["duration [s]"]
    md["z scan size [m]"] = abs(md_im["z start"] - md_im["z end"])
    md["speed [m/s]"] = md["z scan size [m]"]/md["duration [s]"]
    if "position x [m]" in md_im:
        md["position x [µm]"] = md_im["position x [m]"]*1e6
    if "position y [m]" in md_im:
        md["position y [µm]"] = md_im["position y [m]"]*1e6
    if "grid size x [m]" in md_im:
        md["grid size x [µm]"] = md_im["grid size x [m]"]*1e6
    if "grid size y [m]" in md_im:
        md["grid size y [µm]"] = md_im["grid size y [m]"]*1e6
    if "grid center x [m]" in md_im:
        md["grid center x [µm]"] = md_im["grid center x [m]"]*1e6
    if "grid center y [m]" in md_im:
        md["grid center y [µm]"] = md_im["grid center y [m]"]*1e6
    if ("position x [m]" in md_im and
        "position y [m]" in md_im and
        "grid size x [m]" in md_im and
        "grid size y [m]" in md_im and
        "grid center x [m]" in md_im and
        "grid center y [m]" in md_im and
        "grid size x [px]" in md and
            "grid size y [px]" in md):
        pxpx = position_m2px(pos_m=md_im["position x [m]"],
                             size_m=md_im["grid size x [m]"],
                             center_m=md_im["grid center x [m]"],
                             size_px=md["grid size x [px]"])
        pypx = position_m2px(pos_m=md_im["position y [m]"],
                             size_m=md_im["grid size y [m]"],
                             center_m=md_im["grid center y [m]"],
                             size_px=md["grid size y [px]"])
        md["position x [px]"] = pxpx
        md["position y [px]"] = pypx

    integer_keys = ["position x [px]",
                    "position y [px]",
                    "grid size x [px]",
                    "grid size y [px]",
                    "points",
                    ]
    for ik in integer_keys:
        if ik in md:
            md[ik] = int(round(md[ik]))
    return md


@functools.lru_cache(maxsize=10)
def get_seg_head_prop(path_seg_head_prop):
    """ Obtain the properies of a "segment-header.properties" file

    Parameters
    ----------
    path_seg_head_prop : str
        Full path to a "segment-header.properties" file

    Notes
    -----
    This method also parses these files if present:
    - "shared-data/header.properties"
    - "header.properties"
    """
    path = pathlib.Path(path_seg_head_prop).resolve()
    with path.open(mode="rb") as fd:
        prop = jprops.load_properties(fd)

    # Determine if we have a shared-data file
    # These are positions in the file system that could contain the
    # shared properties files:
    shared_locs = [path.parents[2] / "shared-data",
                   path.parents[4] / "shared-data",
                   ]
    general_locs = [path.parents[2],
                    path.parents[4]
                    ]

    # Loop through the candidates
    for cc in shared_locs:
        shared = cc / "header.properties"
        if shared.exists():
            # A candidate exists. Load its properties.
            psprop = load_prop_file(shared)
            # Generate lists of keys and sort them for easier debugging.
            proplist = list(prop.keys())
            proplist.sort()
            pslist = list(psprop.keys())
            pslist.sort()
            # Loop through the segment data and search for lcd-info tags
            for key in proplist:
                # Get line channel data
                if key.count(".*"):
                    # Replace the lcd-info tag by the values in the shared
                    # properties file:
                    # 0, 1, 2, 3, etc.
                    index = prop[key]
                    # lcd-info, force-segment-header-info
                    mediator = ".".join(key.split(".")[-2:-1])
                    # channel.vDeflection, force-segment-header
                    headkey = key.rsplit(".", 2)[0]
                    # append a "." here to make sure not to confuse "1" with "10".
                    startid = "{}.{}.".format(mediator, index)

                    for k2 in pslist:
                        if k2.startswith(startid):
                            var = ".".join(k2.split(".")[2:])
                            prop[".".join([headkey, var])] = psprop[k2]

    for gg in general_locs:
        gen = gg / "header.properties"
        if gen.exists():
            gsprop = load_prop_file(gen)
            # Add all other keys
            for pk in gsprop:
                prop[pk] = gsprop[pk]

    for p in prop:
        try:
            prop[p] = float(prop[p])
        except:
            pass

    return prop


@functools.lru_cache(maxsize=10)
def load_prop_file(path):
    path = pathlib.Path(path)
    with path.open(mode="rb") as fd:
        props = jprops.load_properties(fd)
    return props


def position_m2px(pos_m, size_m, center_m, size_px):
    """Convert qmap positions from [m] to array coordinates in [px]

    Parameters
    ----------
    pos_m: float
        positions [m]
    size_m: float
        grid size [m]
    center_m: float
        grid center position [m]
    size_px: int
        grid size [px]

    Returns
    -------
    pos_px: int
        index position of `pos_m`
    """
    if size_px != int(size_px):
        raise ValueError("`size_px` must be integer, got {}!".format(size_px))
    size_px = int(size_px)
    s1 = center_m - size_m / 2
    s2 = center_m + size_m / 2

    x, dx = np.linspace(s1, s2, size_px, endpoint=False, retstep=True)
    x += dx/2

    xpx = np.nanargmin(np.abs(x-pos_m))
    return xpx
