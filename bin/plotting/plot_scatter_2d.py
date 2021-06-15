import sys
import click

import numpy as np
import pandas as pd

import json

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm
from pathlib import Path

from tqdm import trange

from bore.optimizers.utils import from_bounds

from bore_experiments.plotting.utils import GOLDEN_RATIO, WIDTH, pt_to_in, load_frame
from bore_experiments.benchmarks import make_benchmark


def contour(X, Y, Z, ax=None, *args, **kwargs):

    kwargs.pop("color")

    if ax is None:
        ax = plt.gca()

    ax.contour(X, Y, Z, *args, **kwargs)


@click.command()
@click.argument("benchmark_name")
@click.argument("method_name")
@click.option('--num-runs', '-n', default=20)
@click.option('--x-key', default='x')
@click.option('--y-key', default='y')
@click.option('--x-num', default=512)
@click.option('--y-num', default=512)
@click.option('--log-error-lim', type=(float, float), default=(-2, 3))
@click.option('--num-error-levels', default=20)
@click.option('--col-wrap', default=4)
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--input_dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True))
@click.option("--output-dir", default="figures/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, method_name, num_runs, x_key, y_key, x_num, y_num,
         log_error_lim, num_error_levels, col_wrap, transparent, context,
         style, palette, width, height, aspect, dpi, extension,
         input_dir, output_dir):

    # preamble
    if height is None:
        height = width / aspect
    # height *= num_iterations
    # figsize = size(width, aspect)
    figsize = (width, height)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir).joinpath(benchmark_name)
    output_path = Path(output_dir).joinpath(benchmark_name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark = make_benchmark(benchmark_name)

    bounds = benchmark.get_bounds()
    (low, high), input_dim = from_bounds(bounds)
    (x1_min, x2_min), (x1_max, x2_max) = low, high

    x2, x1 = np.ogrid[x2_min:x2_max:y_num*1j,
                      x1_min:x1_max:x_num*1j]
    X1, X2 = np.broadcast_arrays(x1, x2)
    Y = benchmark(X1, X2) - benchmark.get_minimum()

    frames = []
    for run in trange(num_runs):

        path = input_path.joinpath(method_name, f"{run:03d}.csv")

        frame = load_frame(path, run, loss_min=benchmark.get_minimum())
        frames.append(frame.assign(method=method_name))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    # TODO: height should actually be `height_in / row_wrap`, but we don't
    # know what `row_wrap` is.
    g = sns.relplot(x=x_key, y=y_key, hue="batch",  # size="error",
                    col="run", col_wrap=col_wrap,
                    palette="mako_r", alpha=0.8,
                    height=height, aspect=aspect,
                    kind="scatter", data=data, rasterized=True)
    # facet_kws=dict(subplot_kws=dict(rasterized=True)))
    g.map(contour, X=X1, Y=X2, Z=Y,
          levels=np.logspace(*log_error_lim, num_error_levels),
          norm=LogNorm(), alpha=0.4, cmap="turbo", zorder=-1)

    for ext in extension:
        g.savefig(output_path.joinpath(f"scatter_{context}_{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
