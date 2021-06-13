import pathlib

from .indent import Indentation
from .read import load_data


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
    data = load_data(path, callback=callback, meta_override=meta_override)
    grp = IndentationGroup()
    for dd in data:
        grp.append(Indentation(dd))
    grp.path = path
    return grp


class IndentationGroup(object):
    def __init__(self, path=None, callback=None, meta_override=None):
        """Group of Indentation

        Parameters
        ----------
        path: str or pathlib.Path or None
            The path to the data file. The data format is determined
            and the file is loaded using :ref:`afmformats:index`.
        callback: callable or None
            A method that accepts a float between 0 and 1
            to externally track the process of loading the data.
        """
        if path is not None:
            path = pathlib.Path(path)
        self._mmlist = []

        if path is not None:
            self += load_group(path,
                               callback=callback,
                               meta_override=meta_override)

        self.path = path

    def __add__(self, grp):
        out = IndentationGroup()
        out._mmlist = self._mmlist + grp._mmlist
        return out

    def __iadd__(self, grp):
        self._mmlist += grp._mmlist
        self.path = None
        return self

    def __iter__(self):
        return iter(self._mmlist)

    def __getitem__(self, idx):
        return self._mmlist[idx]

    def __len__(self):
        return len(self._mmlist)

    def __repr__(self):
        rep = ["IndentationGroup: '{}'".format(self.path)]
        for idnt in self._mmlist:
            rep.append("- {}".format(idnt))
        return "\n".join(rep)

    def get_enum(self, enum):
        """Return the indentation curve with this enum value

        Raises
        ------
        ValueError if multiple curves with the same enum value exist.
        KeyErrir if the enum value is not found
        """
        curves = []
        for item in self._mmlist:
            if item.enum == enum:
                curves.append(item)
        if len(curves) == 0:
            raise KeyError("Could not find dataset with enum {}".format(enum))
        elif len(curves) == 1:
            return curves[0]
        else:
            raise ValueError("Multiple curves with the same enum value exist!")

    def append(self, item):
        """Append an indentation dataset

        Parameters
        ----------
        item: nanite.indent.Indentation
            Force-indentation dataset
        """
        if not isinstance(item, Indentation):
            raise ValueError("`item` must be an instance of `Indentation`!")
        self._mmlist.append(item)

    def index(self, item):
        return self._mmlist.index(item)

    def subgroup_with_path(self, path):
        """Return a subgroup with measurements matching `path`"""
        path = pathlib.Path(path)
        subgroup = IndentationGroup()
        for idnt in self:
            if pathlib.Path(idnt.path).resolve() == path.resolve():
                subgroup.append(idnt)
        subgroup.path = path
        return subgroup
