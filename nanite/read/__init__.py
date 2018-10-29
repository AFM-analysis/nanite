import pathlib

from . import read_jpk

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
            data = load_data(pp)
        except BaseException:
            if skip_errors:
                continue
            else:
                raise
        for enum in range(len(data)):
            enumpaths.append([pp, enum])
    return enumpaths


def load_data(path, callback=None):
    """Load an experimental data file

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
    path = pathlib.Path(path)
    for reader in readers:
        load, exts = reader
        if path.suffix in exts:
            return load(path, callback=callback)
    else:
        raise NotImplementedError("Unknown file format: {}".
                                  format(path.name))


#: All supported file extensions
supported_extensions = [ext for reader in readers for ext in reader[1]]
