import pathlib

import afmformats


def get_data_paths(path):
    """Obtain a list of data files

    Parameters
    ----------
    path: str or pathlib.Path
        Path to a data file or a directory containing
        data files.

    Returns
    -------
    paths: list of pathlib.Path
        All supported data files found in `path`. If `path`
        is a file, `[pathlib.Path(path)]` is returned. If
        `path` has an unsupported extion, an empty list is
        returned.
    """
    path = pathlib.Path(path)
    paths = []
    if path.is_dir():
        # recurse into directories
        for ext in afmformats.supported_extensions:
            paths += sorted(path.rglob("*{}".format(ext)))
    elif path.suffix in afmformats.supported_extensions:
        paths = [path]
    return sorted(paths)


def get_data_paths_enum(path, skip_errors=False):
    paths = get_data_paths(path)
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


def load_data(path, callback=None):
    """Load data and return list of :class:`afmformats.AFMForceDistance`"""
    paths = get_data_paths(path)
    data = []
    for pp in paths:
        measurements = afmformats.load_data(pp, callback=callback)
        for dd in measurements:
            if dd.mode == "force-distance":
                data.append(dd)
    return data
