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
from ..group import IndentationGroup
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

        root.title("nanite force-indentation rater")

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
    fit_perform(path, path_results=pout)


def fit_perform(path, path_results, profile_path=PROFILE_PATH):
    path_results = pathlib.Path(path_results)
    ptsv = path_results / "statistics.tsv"
    ptif = path_results / "plots.tif"
    # exported data columns
    pf = Profile(path=profile_path, create=False)
    dlist = [["path", lambda x: x.path],
             ["enum", lambda x: x.enum],
             ["E", lambda x: x.fit_properties["params_fitted"]["E"].value],
             ["rating", lambda x: round(x.rate_quality(
                 training_set=pf["rating training set"],
                 regressor=pf["rating regressor"]),
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
            grp = IndentationGroup(pp)
            for idnt in grp:
                fit_data(idnt, profile_path=profile_path)
                # save statistics
                stats = [str(dd[1](idnt)) for dd in dlist]
                ts.write("\t".join(stats) + "\n")
                # save plot
                imio = io.BytesIO()
                rating_text = "Rating parameters:\n" \
                    + "regressor: {}\n".format(pf["rating regressor"]) \
                    + "training set: {}\n".format(pf["rating training set"]) \
                    + "rating: {:.1f}\n".format(ddict["rating"](idnt))
                plot_data(idnt,
                          add_text=rating_text,
                          path=imio)
                imio.seek(0)
                imdat = (mpimg.imread(imio) * 255).astype("uint8")
                tf.save(imdat, compress=9)


@lru_cache(maxsize=5)
def fit_data(path, enum=0, profile_path=PROFILE_PATH):
    if isinstance(path, Indentation):
        idnt = path
    else:
        idnt = IndentationGroup(path)[enum]

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
    descr = "Fit AFM force-indentation data. Statistics (.tsv file) and " \
            + "visualizations of the fits (multi-page .tif file) are stored " \
            + "in the results directory."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('data_path', type=str,
                        help='input folder containing AFM force-indentation '
                             + 'data')
    parser.add_argument('out_dir', type=str,
                        help='results directory')
    return parser


def generate_training_set_parser():
    descr = "Create a training set for usage in nanite from rating " \
            "containers (.h5 files manually created with `nanite-rate`)."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('data_path', type=str,
                        help='path to a rating container or a folder '
                             + 'containing rating containers')
    parser.add_argument('out_dir', type=str,
                        help='directory where the training set will be '
                             ' stored')
    return parser


def generate_training_set():
    parser = generate_training_set_parser()
    args = parser.parse_args()
    data_path = pathlib.Path(args.data_path).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    name = data_path.stem
    if data_path.exists():
        print("Generating training set '{}'.".format(name))
        rm = rio.RateManager(data_path, verbose=1)
        rm.export_training_set(out_dir / "ts_{}".format(name))
    else:
        print("Path does not exist: '{}'".format(data_path))


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
    descr = "Manually rate (the fit to) AFM force-indentation data. " \
            + "A graphical user interface allows to rate and comment on " \
            + "each data set. The fits and all data are stored in a rating " \
            + "container that can then be passed to " \
            + "`nanite-generate-training-set`."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('data_path', type=str,
                        help='input folder containing AFM force-indentation '
                             + 'data')
    parser.add_argument('rating_path', type=str,
                        help='path to the output rating container (will '
                             + 'be created if it does not already exist)')
    return parser
