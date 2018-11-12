import pathlib

from . import read_jpk
from .data import IndentationData, type_indentation

#: All available readers and associated file extensions
readers = [(read_jpk.load_jpk, read_jpk.extensions),
           ]


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
        for ext in supported_extensions:
            paths += sorted(path.rglob("*{}".format(ext)))
    elif path.suffix in supported_extensions:
        paths = [path]
    return sorted(paths)


def get_data_paths_enum(path, skip_errors=False):
    paths = get_data_paths(path)
    enumpaths = []
    for pp in paths:
        try:
            data = load_raw_data(pp)
        except BaseException:
            if skip_errors:
                continue
            else:
                raise
        for enum in range(len(data)):
            enumpaths.append([pp, enum])
    return enumpaths


def load_data(path, callback=None):
    """Load data and return list of Indentation"""
    measurements = load_raw_data(path, callback=callback)
    data = []
    for enum, mm in enumerate(measurements):
        app, ret = mm
        metadata = app[1]
        if metadata["curve type"] in type_indentation:
            data.append(IndentationData(approach=app[0],
                                        retract=ret[0],
                                        metadata=metadata,
                                        path=app[2],
                                        enum=enum
                                        ))
    return data


def load_raw_data(path, callback=None):
    """Load raw data

    Parameters
    ----------
    path: str or pathlib.Path
        Path to a data file or a directory containing
        data files. The data format is determined using the
        extension of the file.
    callback: callable or None
        A method that accepts a float between 0 and 1
        to externally track the process of loading the data.
    ret_indentation: bool
        Return the indentation

    Returns
    -------
    data: list
        A measurements list that contains the data.
    """
    paths = get_data_paths(path)
    data = []
    for ii, pp in enumerate(paths):
        for reader in readers:
            load, exts = reader
            if pp.suffix in exts:
                if callback is None:
                    cbck = None
                else:
                    # modified callback for multiple files
                    cbck = lambda x: (callback(x) + ii)/len(paths)
                data += load(pp, callback=cbck)
                break
        else:
            raise NotImplementedError("Unknown file format: {}".
                                      format(pp.name))
    return data


#: All supported file extensions
supported_extensions = [ext for reader in readers for ext in reader[1]]
