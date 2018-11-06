import pathlib

from .read import load_data
from .indent import Indentation, type_indentation


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
            measurements = load_data(path, callback=callback)
            for enum, m in enumerate(measurements):
                app, ret = m
                metadata = app[1]
                if metadata["curve type"] in type_indentation:
                    self.append(Indentation(approach=app[0],
                                            retract=ret[0],
                                            metadata=metadata,
                                            path=path,
                                            enum=enum
                                            ))

    def __add__(self, ds):
        out = IndentationGroup()
        out._mmlist = self._mmlist + ds._mmlist
        return out

    def __iadd__(self, ds):
        self._mmlist += ds._mmlist
        return self

    def __iter__(self):
        return iter(self._mmlist)

    def __getitem__(self, idx):
        return self._mmlist[idx]

    def __len__(self):
        return len(self._mmlist)

    def __repr__(self):
        return "IndentationGroup: {} ".format(self._mmlist.__repr__())

    def append(self, item):
        self._mmlist.append(item)

    def index(self, item):
        return self._mmlist.index(item)

    def subgroup_with_path(self, path):
        """Return a subgroup with measurements matching `path`"""
        subgroup = IndentationGroup()
        for idnt in self:
            if pathlib.Path(idnt.path) == pathlib.Path(path):
                subgroup.append(idnt)
        return subgroup
