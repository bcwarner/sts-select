import argparse
import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import FIGURE_DIR, MI_CACHE_FNAME, RESULTS_DIR, STS_CACHE_FNAME
from tabtex import nice_name_map
import matplotlib

GLOBAL_COLORMAP = "viridis"

# Unpickle the MI cache
with open(MI_CACHE_FNAME, "rb") as f:
    datadict = pickle.load(f)
    MI_X_pairings = datadict["X_pairings"]
    MI_X_y_pairings = datadict["X_y_pairings"]

# Unpiclke the STS cache
with open(STS_CACHE_FNAME, "rb") as f:
    datadict = pickle.load(f)
    STS_X_pairings = datadict["X_pairings"]
    STS_X_y_pairings = datadict["X_y_pairings"]


def two_param(data, param_dict):
    # Three subplots, side by side.
    # First one is test data.
    # Second one is train data.
    # Third one is the difference.

    # Set current figure width to a 3:1 aspect ratio
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(data[0], cmap=GLOBAL_COLORMAP, aspect="auto")
    plt.title("Test")
    for fn, lab in [
        (plt.xlabel, param_dict["x_name"]),
        (plt.ylabel, param_dict["y_name"]),
    ]:
        fn(f"${lab}$" if ("\\" in lab) else f"{lab}")

    plt.subplot(1, 3, 2)
    plt.imshow(data[1], cmap=GLOBAL_COLORMAP, aspect="auto")
    plt.colorbar()

    plt.title("Train")
    plt.subplot(1, 3, 3)
    plt.imshow(data[0] - data[1], cmap=GLOBAL_COLORMAP, aspect="auto")
    plt.colorbar()
    plt.title("Difference")

    x_steps = 10
    y_steps = 5

    xrange = range(x_steps, len(param_dict["x"]), x_steps)
    yrange = range(0, len(param_dict["y"]), y_steps)

    xtrun = ["{:.2f}".format(z) for z in param_dict["x"][x_steps::x_steps]]

    plt.xticks(xrange, labels=xtrun)
    plt.yticks(yrange, labels=param_dict["y"][::y_steps])

    if param_dict["type"] == "au_map":
        plt.suptitle(
            f"{param_dict['au_name']} for {param_dict['model_name']}, {param_dict['feature_selector']}"
        )


def line_param(data, param_dict, color_idx):
    # Wacky but I don't want to regen plots.
    try:
        xs = [x for x in zip(*data)]
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=plt.cm.get_cmap(GLOBAL_COLORMAP)(np.linspace(0, 1, len(xs))))
    except Exception as e:
        xs = [data]
    for idx, x in enumerate(xs):
        label = "Test" if idx == 0 else "Train"
        label += " (" + ("STS" if param_dict["scorer"] == "SentenceTransformerScorer" else "MI") + ")"
        # Select values of x indexed at 1-20
        start_x = param_dict["x"].index(1)
        end_x = param_dict["x"].index(21)

        # Pick a color from the colormap
        color = matplotlib.colors.to_rgb(plt.rcParams["axes.prop_cycle"].by_key()["color"][color_idx])
        # Make the test data slightly darker
        if idx == 1:
            color = tuple([x * 0.8 for x in color])

        plt.plot(
            param_dict["x"][start_x:end_x], x[start_x:end_x], marker=".", label=label, color=color
        )

    if "model_params" in param_dict:
        model_name_full = param_dict["model_path"].split("/")
        model_vocab = nice_name_map(model_name_full[-1])
        model_name = nice_name_map(model_name_full[-2])
    if len(xs) > 1:
        plt.legend()
    plt.xlabel(f"{param_dict['x_name']}")
    plt.ylabel(f"{param_dict['au_name']}")


def plot_sts_mi_heatmap(heatmap_args):
    def plot_pairings(X_pairings, X_y_pairings, title):
        """
        Plot a 2D heatmap of either the MI or STS pairings.
        :return:
        """
        print(f"Plotting {title} pairings")
        # Iterate through the keys of X_pairings and turn it into a numpy array. The keys of X_pairings are a
        # tuple of two ints indicating index.
        ft_idx, y_idx = zip(*X_y_pairings.keys())
        max_fts = max(ft_idx) + 1
        max_y = max(y_idx) + 1

        # Create a numpy array to store the results of X_pairings, and X_y_pairings
        X_pairings_np = np.array(
            [[X_pairings[(i, j)] for j in range(max_fts)] for i in range(max_fts)],
        )
        # Reflect upper triangular to lower triangular for X_pairings_np
        X_pairings_np = (
            X_pairings_np + X_pairings_np.T - np.diag(X_pairings_np.diagonal())
        )

        if title == "MI":
            max_y = 1

        X_y_pairings_np = np.array(
            [[X_y_pairings[(i, j)] for j in range(max_y)] for i in range(max_fts)],
        )

        plt.clf()
        f, (a1, a0, ab) = plt.subplots(1, 3, gridspec_kw={"width_ratios": [10, 2, 1]})
        f.set_size_inches(7, 4)

        vmin = 0
        vmax = max(np.max(X_pairings_np), np.max(X_y_pairings_np))

        a1.imshow(X_pairings_np, cmap=GLOBAL_COLORMAP, aspect="auto", vmin=vmin, vmax=vmax)
        a1.set_title(f"{title} X-X Pairings")
        a1.set_xlabel("Feature Index")
        a1.set_ylabel("Feature Index")

        a0.imshow(
            X_y_pairings_np, cmap=GLOBAL_COLORMAP, aspect="auto", interpolation=None, vmin=vmin,
            vmax=vmax
        )
        a0.set_title(f"{title} X-y Pairings")

        # Turn off y ticks
        plt.yticks([])
        # Turn off the xticks if y_ticks is 1, otherwise make it the range of max_y
        if max_y == 1:
            a0.set_xticks([])
            a0.set_xlabel("Target")
        else:
            a0.set_xticks(range(max_y), range(1, max_y + 1))
            a0.set_xlabel("Target Index")

        ab.set_xlabel("Score")
        plt.colorbar(a0.get_images()[0], cax=ab)

    plt.clf()
    plot_pairings(MI_X_pairings, MI_X_y_pairings, "MI")
    plt.savefig(os.path.join(os.path.curdir, FIGURE_DIR, "MI_pairings.pdf"), dpi=300)
    # Delete all conents in the sts_pairings directory
    for file in os.listdir(
        os.path.join(os.path.curdir, "redcap", FIGURE_DIR, "sts_pairings")
    ):
        os.remove(
            os.path.join(os.path.curdir, "redcap", FIGURE_DIR, "sts_pairings", file)
        )
    plt.clf()
    heatmaps = []
    if heatmap_args is not None:
        heatmaps = heatmap_args.split(",")
    else:
        heatmaps = [STS_CACHE_FNAME]
    for heatmap in heatmaps:
        with open(heatmap, "rb") as f:
            data_dict = pickle.load(f)
            STS_X_pairings = data_dict["X_pairings"]
            STS_X_y_pairings = data_dict["X_y_pairings"]

        heatmap_name = heatmap.split("/")[-1].replace(".pkl", "")
        plot_pairings(STS_X_pairings, STS_X_y_pairings, f"STS")
        plt.savefig(
            os.path.join(
                os.path.curdir, FIGURE_DIR, f"STS_{heatmap_name}_pairings.pdf"
            ),
            dpi=300,
        )


def swap_x(fn):
    with open(fn, "rb") as f:
        s, p_d = pickle.load(f)
        p_d["x"], p_d["y"] = p_d["y"], p_d["x"]
        p_d["x_name"], p_d["y_name"] = p_d["y_name"], p_d["x_name"]
    with open(fn, "wb") as f:
        pickle.dump((s, p_d), f)


if __name__ == "__main__":
    # Get date of last made figure
    # figure_mods = sorted([os.stat(f).st_mtime for f in os.listdir(os.path.join(os.path.curdir, FIGURE_DIR))], reverse=True)
    # last_fig_t = figure_mods[0]

    parser = argparse.ArgumentParser(prog="plot", description="Plot the results.")

    parser.add_argument(
        "-t",
        "--tikz",
        action="store",
        default=False,
        help="Make tikz figures?",
        type=bool,
    )
    parser.add_argument(
        "-m",
        "--heatmap",
        action="store",
        default=None,
        help="Just make the heatmaps if on.",
    )
    args = parser.parse_args()

    # Set matplotlib font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    if args.heatmap is not None:
        # Plot the MI and STS pairings as well.
        plot_sts_mi_heatmap(args.heatmap)
        sys.exit()

    results = os.listdir(os.path.join(os.path.curdir, RESULTS_DIR))
    new_figs = results
    score_map = defaultdict(list)
    for fname in tqdm(new_figs, desc="Plotting Figures"):
        if ".dat" not in fname:
            continue

        data, param_dict = pickle.load(
            open(os.path.join(os.path.curdir, RESULTS_DIR, fname), "rb")
        )
        param_dict = defaultdict(str, param_dict)
        score_map[param_dict["au_name"]].append((data, param_dict))
    # Plot the results
    for au_name, items in score_map.items():
        # Plot the results for each AU
        plt.clf()
        plt.title(f"{au_name} vs. N")
        for idx, (data, param_dict) in enumerate(items):
            if param_dict["type"] == "au_map":
                two_param(data, param_dict)
            if param_dict["type"] == "au_n_line":
                line_param(data, param_dict, idx)

            plt.savefig(
                os.path.join(os.path.curdir, FIGURE_DIR, f"{au_name}-N.pdf"), dpi=300
            )