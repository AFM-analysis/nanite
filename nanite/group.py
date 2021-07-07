import pathlib

import afmformats

from .read import get_load_data_modality_kwargs, load_data


def load_group(path, callback=None, meta_override=None):
    """Load indentation data from disk

    Parameters
    ----------
    path: path-like
        Path to experimental data
    callback: callable
        function for tracking progress; must accept a float in
        [0, 1] as an argument.
    meta_override: dict
        if specified, contains key-value pairs of metadata that
        should be used when loading the files
        (see :data:`afmformats.meta.META_FIELDS`)

    Returns
    -------
    group: nanite.IndetationGroup
        Indentation group with force-distance data
    """
    path = pathlib.Path(path)
    data = load_data(path,
                     callback=callback,
                     meta_override=meta_override,
                     )
    grp = IndentationGroup()
    grp += data
    grp.path = path
    return grp


class IndentationGroup(afmformats.AFMGroup):
    def __init__(self, path=None, meta_override=None, callback=None):
        """Group of Indentation

        Parameters
        ----------
        path: str or pathlib.Path or None
            The path to the data file. The data format is determined
            and the file is loaded using :ref:`afmformats:index`.
        meta_override: dict
            if specified, contains key-value pairs of metadata that
            should be used when loading the files
            (see :data:`afmformats.meta.META_FIELDS`)
        callback: callable or None
            A method that accepts a float between 0 and 1
            to externally track the process of loading the data.
        """
        super(IndentationGroup, self).__init__(
            path=path,
            meta_override=meta_override,
            callback=callback,
            **get_load_data_modality_kwargs()
        )
