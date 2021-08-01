import warnings

import afmformats
from .indent import Indentation

#: The default imaging modality when loading AFM data. Set this to `None`
#: to also be able to load e.g. creep-compliance data. See issue
#: https://github.com/AFM-analysis/nanite/issues/11 for more
#: information. Note that especially the export of rating containers
#: may not work with any imaging modality other than force-distance.
DEFAULT_MODALITY = "force-distance"


def get_data_paths(path):
    """Return list of data paths with force-distance data

    DEPRECATED
    """
    warnings.warn("`get_data_paths` is deprecated! Please use "
                  + "afmformats.find_data(path, modality='force-distance') "
                  + "instead!",
                  DeprecationWarning)
    return afmformats.find_data(path, modality=DEFAULT_MODALITY)


def get_data_paths_enum(path, skip_errors=False):
    """Return a list with paths and their internal enumeration

    Parameters
    ----------
    path: str or pathlib.Path or list of str or list of pathlib.Path
        path to data files or directory containing data files;
        if directories are given, they are searched recursively
    skip_errors: bool
        skip paths that raise errors

    Returns
    -------
    path_enum: list of lists
        each entry in the list is a list of [pathlib.Path, int],
        enumerating all curves in each file
    """
    paths = afmformats.find_data(path, modality=DEFAULT_MODALITY)
    enumpaths = []
    for pp in paths:
        try:
            data = load_data(pp)
        except BaseException:
            if skip_errors:
                continue
            else:
                raise
        for dd in data:
            enumpaths.append([pp, dd.enum])
    return enumpaths


def get_load_data_modality_kwargs():
    """Return imaging modality kwargs for afmformats.load_data

    Uses :const:`DEFAULT_MODALITY`.

    Returns
    -------
    kwargs: dict
        keyword arguments for :func:`afmformats.load_data`
    """
    kwargs = {
        "modality": DEFAULT_MODALITY,
        "data_classes_by_modality": {
            "force-distance": Indentation,
            "creep-compliance": Indentation,
            "stress-relaxation": Indentation}
    }
    return kwargs


def load_data(path, callback=None, meta_override=None):
    """Load data and return list of :class:`afmformats.AFMForceDistance`

    This is essentially a wrapper around
    :func:`afmformats.formats.find_data` and
    :func:`afmformats.formats.load_data` that returns
    force-distance datasets.

    Parameters
    ----------
    path: str or pathlib.Path or list of str or list of pathlib.Path
        path to data files or directory containing data files;
        if directories are given, they are searched recursively
    callback: callable
        function for progress tracking; must accept a float in
        [0, 1] as an argument.
    meta_override: dict
        if specified, contains key-value pairs of metadata that
        are used when loading the files
        (see :data:`afmformats.meta.META_FIELDS`)
    """
    paths = afmformats.find_data(path, modality=DEFAULT_MODALITY)
    data = []
    for ii, pp in enumerate(paths):
        measurements = afmformats.load_data(
            pp,
            # recurse callback function with None as default
            callback=lambda x: callback((ii + x) / len(paths))
            if callback else None,
            meta_override=meta_override,
            **get_load_data_modality_kwargs()
        )
        data += measurements
    return data
