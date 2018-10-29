import argparse
import io
from functools import lru_cache
import getpass
from os import fspath
import pathlib
import tkinter as tk
import types

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import tifffile

from ..indent import Indentation
from ..dataset import IndentationDataSet
from ..read import get_data_paths, get_data_paths_enum
from ..rate import io as rio

from .profile import Profile, PROFILE_PATH
from .plotting import plot_data


class RatingGUI():
    def __init__(self, root, data_paths, h5path):
        self.data_paths = data_paths
        self.h5path = h5path
        self.current_index = 0
        self.idnt = None
        self.root = root

        root.wm_protocol("WM_DELETE_WINDOW", self.close)
        tk.Grid.rowconfigure(root, 0, weight=1)
        tk.Grid.columnconfigure(root, 0, weight=1)

        root.title("afmfit curve rater")

        master = tk.Frame(root)
        master.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.master = master

        tk.Grid.rowconfigure(master, 1, weight=1)

        for ii in range(7):
            tk.Grid.columnconfigure(master, ii, weight=1)

        l1 = tk.Label(master, text="User text!")
        l1.grid(row=0, column=0, columnspan=6)

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.get_tk_widget().grid(row=1,
                                         column=0,
                                         columnspan=6,
                                         sticky=tk.N+tk.S+tk.E+tk.W)

        l2 = tk.Label(master, text="user:")
        l2.grid(row=2, column=0, sticky=tk.E)
        self.user = tk.Entry(master, text="user:", width=30)
        self.user.insert(0, getpass.getuser())
        self.user.grid(row=2, column=1, sticky=tk.W)

        l3 = tk.Label(master, text="rating:")
        l3.grid(row=2, column=2, sticky=tk.E)
        self.rating = tk.Spinbox(master, from_=-1, to=10)
        self.rating.grid(row=2, column=3, sticky=tk.W)

        self.prev_button = tk.Button(master,
                                     text="Previous (Alt+Left)",
                                     command=self.previous)
        root.bind('<Alt-Left>', self.previous)
        self.prev_button.grid(row=2, column=4, sticky=tk.W+tk.E)

        self.next_button = tk.Button(master,
                                     text="Next (Alt+Right)",
                                     command=self.next)
        root.bind('<Alt-Right>', self.next)
        self.next_button.grid(row=2, column=5, sticky=tk.W+tk.E)

        l4 = tk.Label(master, text="comment:")
        l4.grid(row=3, column=0, sticky=tk.E)
        self.comment = tk.Entry(master, width=100)
        self.comment.grid(row=3, column=1, columnspan=5, sticky=tk.W+tk.E)

        # skip to first unrated curve
        while True:
            fakeidnt = types.SimpleNamespace()
            fakeidnt.path = self.current_path
            fakeidnt.enum = self.current_enum
            rated, _, _ = rio.hdf5_rated(self.h5path, indent=fakeidnt)
            if not rated:
                break
            else:
                self.current_index += 1
        self.load()

    @property
    def current_path(self):
        return self.data_paths[self.current_index][0]

    @property
    def current_enum(self):
        return self.data_paths[self.current_index][1]

    def close(self):
        self.save()
        self.root.destroy()

    def next(self, e=None):
        self.save()
        self.current_index += 1
        if self.current_index == len(self.data_paths):
            self.current_index = 0
        self.load()

    def previous(self, e=None):
        self.save()
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.data_paths) - 1
        self.load()

    def load(self):
        self.set_label_path()
        path = self.current_path
        enum = self.current_enum
        self.idnt = fit_data(path=path, enum=enum)
        plot_data(self.idnt, figure=self.figure)
        self.canvas.draw()

        # set rating
        self.rating.delete(0, tk.END)
        self.comment.delete(0, tk.END)
        rated, rating, comment = rio.hdf5_rated(self.h5path, indent=self.idnt)
        if rated:
            self.rating.insert(0, "{}".format(rating))
            self.comment.insert(0, "{}".format(comment))

    def save(self):
        rating = self.rating.get()
        if rating:
            rio.save_hdf5(h5path=self.h5path,
                          indent=self.idnt,
                          user_rate=int(rating),
                          user_name=self.user.get(),
                          user_comment=self.comment.get(),
                          h5mode="a")

    def set_label_path(self):
        if hasattr(self, "label_path"):
            self.label_path.destroy()
        self.label_path = tk.Label(self.master,
                                   text="curve {}/{}: {} - {}".format(
                                       self.current_index + 1,
                                       len(self.data_paths),
                                       self.current_path,
                                       self.current_enum))
        self.label_path.grid(row=0, column=0, columnspan=6)


def fit():
    parser = fit_parser()
    args = parser.parse_args()
    path = pathlib.Path(args.data_path).resolve()
    pout = pathlib.Path(args.out_dir).resolve()
    pout.mkdir(exist_ok=True, parents=True)
    ptsv = pout / "statistics.tsv"
    ptif = pout / "plots.tif"
    # exported data columns
    pf = Profile(create=False)
    dlist = [["path", lambda x: x.path],
             ["enum", lambda x: x.enum],
             ["E", lambda x: x.fit_properties["params_fitted"]["E"].value],
             ["rating", lambda x: round(x.rate_quality(
                 method=pf["rating method"],
                 ts_label=pf["rating train set"]),
                 ndigits=1)],
             ]
    ddict = dict(dlist)
    header = "\t".join([dd[0] for dd in dlist])
    with ptsv.open(mode="w") as ts:
        ts.write(header + "\n")
    # get all files in path
    datapaths = get_data_paths(path)
    with tifffile.TiffWriter(fspath(ptif), imagej=True) as tf, \
            ptsv.open(mode="a") as ts:
        for pp in datapaths:
            print("Processing: {}".format(pp))
            ds = IndentationDataSet(pp)
            for idnt in ds:
                fit_data(idnt)
                # save statistics
                stats = [str(dd[1](idnt)) for dd in dlist]
                ts.write("\t".join(stats) + "\n")
                # save plot
                imio = io.BytesIO()
                rating_text = "Rating parameters:\n" \
                    + "method: {}\n".format(pf["rating method"]) \
                    + "label: {}\n".format(pf["rating train set"]) \
                    + "rating: {:.1f}\n".format(ddict["rating"](idnt))
                plot_data(idnt,
                          add_text=rating_text,
                          path=imio)
                imio.seek(0)
                imdat = (mpimg.imread(imio) * 255).astype("uint8")
                tf.save(imdat)


@lru_cache(maxsize=5)
def fit_data(path, enum=0, profile_path=PROFILE_PATH):
    if isinstance(path, Indentation):
        idnt = path
    else:
        idnt = IndentationDataSet(path)[enum]

    pf = Profile(path=profile_path, create=False)

    idnt.apply_preprocessing(pf["preprocessing"])
    params = pf.get_fit_params()

    idnt.fit_model(model_key=pf["model_key"],
                   params_initial=params,
                   range_type=pf["range_type"],
                   range_x=pf["range_x"],
                   segment=pf["segment"],
                   weight_cp=pf["weight_cp"],
                   )
    return idnt


def fit_parser():
    descr = "Fit a model to experimental AFM curves."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('data_path', type=str,
                        help='Input folder containing AFM data')
    parser.add_argument('out_dir', type=str,
                        help='Results directory')
    return parser


def rate():
    parser = rate_parser()
    args = parser.parse_args()
    path = pathlib.Path(args.data_path).resolve()
    h5 = pathlib.Path(args.rating_path).resolve()
    if not h5.name.endswith(".h5"):
        h5 = h5.with_name(h5.name + ".h5")
    # get all files in path
    enumpaths = get_data_paths_enum(path, skip_errors=True)
    # for now, only compare single curves
    enumpaths = [ee for ee in enumpaths if ee[0].suffix == ".jpk-force"]
    # start GUI
    root = tk.Tk()
    RatingGUI(root, data_paths=enumpaths, h5path=h5)
    root.mainloop()


def rate_parser():
    descr = "Manually rate experimental AFM curves."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('data_path', type=str,
                        help='Input folder containing AFM data')
    parser.add_argument('rating_path', type=str,
                        help='hdf5 output file with user rating')
    return parser
