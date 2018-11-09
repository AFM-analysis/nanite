import pathlib

from pandas import DataFrame


class IndentationData(object):
    def __init__(self, approach, retract, metadata, path, enum=0):
        """Initialize an indentation data set

        A microindentation data set contains approach and
        retract curves.

        Parameters
        ----------
        approach, retract: [1d ndarray, unit, title]
            The respective curves obtained by the `readfiles` module.
            The columns must be available:

              - "time" [s]
              - "force" [N]
              - "height (measured)" [m]
        metadata: dict
            Metadata information obtained by the `readfiles` module.
        """
        self.metadata = metadata
        self.path = pathlib.Path(path)
        self.enum = enum

        app = DataFrame(approach)
        app["segment"] = False
        ret = DataFrame(retract)
        ret["segment"] = True
        ret["time"] += metadata["duration [s]"]
        ret.index += len(app.index)
        #: All data in a Pandas DataFrame
        self.data = app.append(ret)


#: metadata identifiers for force-indentation data
type_indentation = ["extend", "retract"]
