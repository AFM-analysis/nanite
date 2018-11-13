"""Methods to open JPK data files and to obtain meta data"""
import pathlib
import shutil
import zipfile

import numpy as np

from . import read_jpk_meta as meta


class ReadJPKError(BaseException):
    pass


class ReadJPKColumnError(BaseException):
    pass


extensions = [".jpk-force", ".jpk-force-map"]
VALID_IDS_FORCE = ["vDeflection"]
VALID_IDS_HEIGHT = ["strainGaugeHeight", "capacitiveSensorHeight"]


def load_jpk(path, callback=None):
    """ Extracts force, measured height, and time from JPK files

    All columns are returned in SI units.

    Parameters
    ----------
    path: str
        Path to a JPK force file
    callback: callable or None
        A method that accepts a float between 0 and 1
        to externally track the process of loading the data.
    """
    path = pathlib.Path(path)
    if callback:
        callback(0)
    # First, only extract the properties
    tdir = meta.extract_jpk(path, props_only=True)
    # Get data file names
    indexlist = meta.get_data_list(path)

    measurements = []
    with zipfile.ZipFile(str(path)) as arc:
        for ii, item in enumerate(indexlist):
            mm = []
            segroot = tdir / item[0][0].rsplit("segments", 1)[0]
            # go through the segments
            for mi, curve in enumerate(item):
                # mi == 0: approach
                # mi == 1: retract
                # get meta data (segfolder contains "segment-header.properties")
                segfolder = tdir / curve[0].rsplit("channels")[0]
                try:
                    mdi = meta.get_meta_data_seg(segfolder)
                except meta.ReadJPKMetaKeyError as exc:
                    exc.args = ("{}, File: '{}'".format(exc.args[0], path),)
                    raise
                segment = {}
                # segment time
                segment["time"] = np.linspace(0, mdi["duration [s]"],
                                              mdi["points"], endpoint=False)
                # load force data
                force_col = None
                for dd in curve:
                    for vc in VALID_IDS_FORCE:
                        if vc in dd:
                            force_col = vc
                            arc.extract(dd, str(tdir))
                            break
                    if force_col:
                        break
                else:
                    msg = "No force data: {} - {}".format(path, segfolder)
                    raise ReadJPKError(msg)
                force, unit, _n = load_jpk_single_curve(segroot,
                                                        segment=mi,
                                                        column=force_col,
                                                        slot="force")
                if unit != "N":
                    msg = "Unknown unit for force: {}".format(unit)
                    raise ReadJPKError(msg)
                segment["force"] = force

                # load height data
                height_col = None
                for dd in curve:
                    for vc in VALID_IDS_HEIGHT:
                        if vc in dd:
                            height_col = vc
                            arc.extract(dd, str(tdir))
                            break
                    if height_col:
                        break
                else:
                    msg = "No height data: {} - {}".format(path, segfolder)
                    raise ReadJPKError(msg)
                height, unit, _n = load_jpk_single_curve(segroot,
                                                         segment=mi,
                                                         column=height_col,
                                                         slot="nominal")
                if unit != "m":
                    msg = "Unknown unit for height: {}".format(unit)
                    raise ReadJPKError(msg)
                segment["height (measured)"] = height

                mm.append([segment, mdi, path])
            if callback:
                # Callback with a float between 0 and 1 to update
                # a progress dialog or somesuch.
                callback(ii/len(indexlist))

            measurements.append(mm)
            shutil.rmtree(str(segroot))

    if tdir.is_dir():
        shutil.rmtree(str(tdir))
    return measurements


def load_jpk_single_curve(path_jpk, segment=0, column="vDeflection", slot="default"):
    """ Load a single curve from a jpk-force file

    Parameters
    ----------
    path_jpk : str
        Path to a jpk-force file or to a directory containing "segments"
    segment: int
        Index of the segment to use.
    column: str
        Column name; one of:

            - "height" : piezzo height
            - "vDeflection": measured deflection
            - "straingGaugeHeight": measured height

    slot: str
        The .dat files in the JPK measurement zip files come with different
        calibration slots. Valid values are

            - For the height of the piezzo crystal during measurement:
              "height.dat": "volts", "nominal", "calibrated"

            - For the measured height of the cantilever:
              "strainGaugeHeight.dat": "volts", "nominal", "absolute"

            - For the recorded cantilever deflection:
              "vDeflection.dat": "volts", "distance", "force"


    Returns
    -------
    data: 1d ndarray
        A numpy array containing the scaled data.
    unit: str
        A string representing the metric unit of the data.
    name: str
        The name of the data column.


    Notes
    -----
    This method does is not designed for directly opening JPK files.
    Please use the `load_jpk` method instead, which wraps around this
    method and handles exceptions better.
    """
    path_jpk = pathlib.Path(path_jpk)
    if path_jpk.is_dir():
        tdir = path_jpk
        cleanup = False
    else:
        tdir = meta.extract_jpk(path_jpk)
        cleanup = True
    segroot = tdir / "segments"
    if not segroot.exists():
        raise OSError("No `segments` subdir found in {}!".format(tdir))
    if not (segroot / str(segment)).exists():
        raise ValueError("Segment {} not found in {}!".format(segment,
                                                              path_jpk))
    chroot = segroot / str(segment) / "channels"
    channels = chroot.glob("*.dat")
    for ch in channels:
        key = ch.stem
        if key == column:
            data = load_dat_unit(ch, slot=slot)
            break
    else:
        msg = "No data for column '{}' and slot '{}'".format(column, slot)
        raise ReadJPKColumnError(msg)
    if cleanup:
        shutil.rmtree(str(tdir))

    return data


def retrieve_segments_data(path_dir):
    """ From an extracted jpk file, retrieve the containing segments.

    This is a convenience method that returns a list of the measurement
    data with the default slot, including units and column names.

    Parameters
    ----------
    path_dir: str
        Path to a directory containing a "segments" folder.

    Returns
    -------
    segment_list: list
        A list with items: [data, unit, column_name]

    """
    path_dir = pathlib.Path(path_dir)
    segroot = path_dir / "segments"
    segment_data = []
    for se in sorted(segroot.glob("[0-1]")):
        chan_data = []
        chroot = se / "channels"
        for ch in chroot.glob("*.dat"):
            chan_data.append(load_dat_unit(ch))
        segment_data.append(chan_data)
    return segment_data


def load_dat_raw(path_dat):
    """ Load data from binary JPK .dat files.

    Parameters
    ----------
    path_dat: str
        Path to a .dat file. A `segment-header.properties`
        file must be present in the parent folder.

    Returns
    -------
    data: 1d ndarray
        A numpy array with the raw data.


    Notes
    -----
    This method tries to correctly determine the data type of the
    binary data and scales it with the `data.encoder.scaling` 
    values given in the header files.

    See Also
    --------
    load_dat_unit: Includes conversion to useful units

    """
    path_dat = pathlib.Path(path_dat).resolve()
    key = path_dat.stem

    # open header file
    header_file = path_dat.parents[1] / "segment-header.properties"
    prop = meta.get_seg_head_prop(header_file)

    # extract multiplier and offset from header
    # multiplier
    mult_str1 = "channel.{}.data.encoder.scaling.multiplier".format(key)
    mult_str2 = "channel.{}.encoder.scaling.multiplier".format(key)
    try:
        mult = prop[mult_str1]
    except:
        mult = prop[mult_str2]
    # offset
    off_str1 = "channel.{}.data.encoder.scaling.offset".format(key)
    off_str2 = "channel.{}.encoder.scaling.offset".format(key)
    try:
        off = prop[off_str1]
    except:
        off = prop[off_str2]
    # get encoder
    enc_str1 = "channel.{}.data.encoder.type".format(key)
    enc_str2 = "channel.{}.encoder.type".format(key)
    try:
        enc = prop[enc_str1]
    except:
        enc = prop[enc_str2]
    # determine encoder
    if enc == "signedshort":
        mydtype = np.dtype(">i2")
    elif enc == "unsignedshort":
        mydtype = np.dtype(">u2")
    elif enc == "signedinteger":
        mydtype = np.dtype(">i4")
    elif enc == "unsignedinteger":
        mydtype = np.dtype(">u4")
    elif enc == "signedlong":
        mydtype = np.dtype(">i8")
    else:
        raise NotImplementedError("Data file format '{}' not supported".
                                  format(enc))

    data = np.fromfile(str(path_dat), dtype=mydtype) * mult + off
    return data


def load_dat_unit(path_dat, slot="default"):
    """Load data from a JPK .dat file with a specific calibration slot


    Parameters
    ----------
    path_dat : str
        Path to a .dat file
    slot: str
        The .dat files in the JPK measurement zip files come with different
        calibration slots. Valid values are

            For the height of the piezzo crystal during measurement:
            "height.dat": "volts", "nominal", "calibrated"

            For the measured height of the cantilever:
            "strainGaugeHeight.dat": "volts", "nominal", "absolute"

            For the recorded cantilever deflection:
            "vDeflection.dat": "volts", "distance", "force"


    Returns
    -------
    data: 1d ndarray
        A numpy array containing the scaled data.
    unit: str
        A string representing the metric unit of the data.
    name: str
        The name of the data column.


    Notes
    -----    
    The raw data (see `load_dat_raw`) is usually stored in "volts" and 
    needs to be converted to e.g. "force" for "vDeflection" or "nominal"
    for "strainGaugeHeight". The conversion parameters (offset, multiplier)
    are stored in the header files and they are not stored separately for
    each slot, but the conversion parameters are stored relative to the
    slots. For instance, to compute the "force" slot from the raw "volts"
    data, one first needs to compute the "distance" slot. This conversion
    is taken care of by this method.

    This is an example header:

        channel.vDeflection.data.file.name=channels/vDeflection.dat
        channel.vDeflection.data.file.format=raw
        channel.vDeflection.data.type=short
        channel.vDeflection.data.encoder.type=signedshort
        channel.vDeflection.data.encoder.scaling.type=linear
        channel.vDeflection.data.encoder.scaling.style=offsetmultiplier
        channel.vDeflection.data.encoder.scaling.offset=-0.00728873489143207
        channel.vDeflection.data.encoder.scaling.multiplier=3.0921021713588157E-4
        channel.vDeflection.data.encoder.scaling.unit.type=metric-unit
        channel.vDeflection.data.encoder.scaling.unit.unit=V
        channel.vDeflection.channel.name=vDeflection
        channel.vDeflection.conversion-set.conversions.list=distance force
        channel.vDeflection.conversion-set.conversions.default=force
        channel.vDeflection.conversion-set.conversions.base=volts
        channel.vDeflection.conversion-set.conversion.volts.name=Volts
        channel.vDeflection.conversion-set.conversion.volts.defined=false
        channel.vDeflection.conversion-set.conversion.distance.name=Distance
        channel.vDeflection.conversion-set.conversion.distance.defined=true
        channel.vDeflection.conversion-set.conversion.distance.type=simple
        channel.vDeflection.conversion-set.conversion.distance.comment=Distance
        channel.vDeflection.conversion-set.conversion.distance.base-calibration-slot=volts
        channel.vDeflection.conversion-set.conversion.distance.calibration-slot=distance
        channel.vDeflection.conversion-set.conversion.distance.scaling.type=linear
        channel.vDeflection.conversion-set.conversion.distance.scaling.style=offsetmultiplier
        channel.vDeflection.conversion-set.conversion.distance.scaling.offset=0.0
        channel.vDeflection.conversion-set.conversion.distance.scaling.multiplier=7.000143623002982E-8
        channel.vDeflection.conversion-set.conversion.distance.scaling.unit.type=metric-unit
        channel.vDeflection.conversion-set.conversion.distance.scaling.unit.unit=m
        channel.vDeflection.conversion-set.conversion.force.name=Force
        channel.vDeflection.conversion-set.conversion.force.defined=true
        channel.vDeflection.conversion-set.conversion.force.type=simple
        channel.vDeflection.conversion-set.conversion.force.comment=Force
        channel.vDeflection.conversion-set.conversion.force.base-calibration-slot=distance
        channel.vDeflection.conversion-set.conversion.force.calibration-slot=force
        channel.vDeflection.conversion-set.conversion.force.scaling.type=linear
        channel.vDeflection.conversion-set.conversion.force.scaling.style=offsetmultiplier
        channel.vDeflection.conversion-set.conversion.force.scaling.offset=0.0
        channel.vDeflection.conversion-set.conversion.force.scaling.multiplier=0.043493666407368466
        channel.vDeflection.conversion-set.conversion.force.scaling.unit.type=metric-unit
        channel.vDeflection.conversion-set.conversion.force.scaling.unit.unit=N

    To convert from the raw "volts" data to force data, these steps are
    performed:

    - Convert from "volts" to "distance" first, because the
      "base-calibration-slot" for force is "distance".

      distance = volts*7.000143623002982E-8 + 0.0

    - Convert from "distance" to "force":

      force = distance*0.043493666407368466 + 0.0

    The multipliers shown above are the values for sensitivity and spring
    constant:
    sensitivity = 7.000143623002982E-8 m/V
    spring_constant = 0.043493666407368466 N/m
    """
    path_dat = pathlib.Path(path_dat).resolve()
    key = path_dat.stem

    # open header file
    header_file = path_dat.parents[1] / "segment-header.properties"
    prop = meta.get_seg_head_prop(header_file)

    conv = "channel.{}.conversion-set".format(key)

    if slot == "default":
        slot = prop[conv+".conversions.default"]

    # get base unit
    base = prop[conv+".conversions.base"]

    # Now iterate through the conversion sets until we have the base converter.
    # A list of multipliers and offsets
    converters = []
    curslot = slot

    while curslot != base:
        # Get current slot multipliers and offsets
        off_str = conv+".conversion.{}.scaling.offset".format(curslot)
        off = prop[off_str]
        mult_str = conv+".conversion.{}.scaling.multiplier".format(curslot)
        mult = prop[mult_str]
        converters.append([mult, off])
        sl_str = conv+".conversion.{}.base-calibration-slot".format(curslot)
        curslot = prop[sl_str]

    # Get raw data
    data = load_dat_raw(path_dat)
    for c in converters[::-1]:
        data[:] = c[0] * data[:] + c[1]

    if base == slot:
        unit_str = "channel.{}.data.encoder.scaling.unit.unit".format(key)
        unit = prop[unit_str]
    else:
        try:
            unit_str = conv+".conversion.{}.scaling.unit".format(slot)
            unit = prop[unit_str]
        except KeyError:
            unit_str = conv+".conversion.{}.scaling.unit.unit".format(slot)
            unit = prop[unit_str]

    name_str = conv+".conversion.{}.name".format(slot)
    name = prop[name_str]
    return data, unit, "{} ({})".format(key, name)
