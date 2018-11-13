import pathlib

from .indent import Indentation
from .read import load_data


def load_group(path, callback=None):
    """Load indentation data from disk

    Parameters
    ----------
    path: path-like
        Path to experimental data
    callback: callable or None
        Callback function for tracking loading progress

    Returns
    -------
    group: nanite.IndetationGroup
        Indentation group with force-indentation data
    """
    path = pathlib.Path(path)
    data = load_data(path, callback=callback)
    grp = IndentationGroup()
    for dd in data:
        grp.append(Indentation(dd))
    grp.path = path
    return grp


class IndentationGroup(object):
    def __init__(self, path=None, callback=None):
        """Group of Indentation

        Parameters
        ----------
        path: str
            The path to the data file. The data format is determined
            using the extension of the file and the data is loaded
            with the correct method.
        callback: callable or None
            A method that accepts a float between 0 and 1
            to externally track the process of loading the data.
        """
        self._mmlist = []

        if path is not None:
            self += load_group(path, callback=callback)

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

    def append(self, item):
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
