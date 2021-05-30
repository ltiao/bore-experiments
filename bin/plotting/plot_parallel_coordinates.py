import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import KBinsDiscretizer
from bore.optimizers.utils import from_bounds
from bore_experiments.benchmarks import Branin
from bore_experiments.plotting.utils import GOLDEN_RATIO, WIDTH, pt_to_in

from pathlib import Path


@click.command()
@click.argument("name")
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float, default=300)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option('--seed', '-s', default=8888)
def main(name, output_dir, transparent, context, style, palette, width, height,
         aspect, dpi, extension, seed):

    num_samples = 256
    random_state = np.random.RandomState(seed)

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

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)
    # / preamble

    benchmark = Branin()
    bounds = benchmark.get_bounds()

    (low, high), dim = from_bounds(bounds)

    x = random_state.uniform(low=low, high=high, size=(num_samples, dim))
    X = np.expand_dims(x, axis=1)

    y = benchmark(x[::, 0], x[::, 1])

    n_bins = 10
    scaler = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    z = 1 + scaler.fit_transform(y.reshape(-1, 1)).squeeze()

    frame = pd.DataFrame(data=x).assign(y=y, z=z)

    fig, ax = plt.subplots()

    pd.plotting.parallel_coordinates(frame, class_column="z", use_columns=False,
                                     colormap="turbo", sort_labels=False,
                                     linewidth=0.25, alpha=0.7, ax=ax)

    ax.legend()

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"foo_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
