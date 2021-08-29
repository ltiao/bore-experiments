import sys
import click

import seaborn as sns

from sklearn.preprocessing import KBinsDiscretizer

from pathlib import Path
from tqdm import trange

from bore_experiments.plotting.utils import GOLDEN_RATIO, WIDTH, pt_to_in, load_frame


@click.command()
@click.argument("benchmark_name")
@click.argument("method_name")
@click.option('--variables', '-v', multiple=True)
@click.option('--n-runs', '-n', default=20)
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
def main(benchmark_name, method_name, variables, n_runs, col_wrap, transparent,
         context, style, palette, width, height, aspect, dpi, extension,
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
        "text.usetex": False,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir).joinpath(benchmark_name)
    output_path = Path(output_dir).joinpath(benchmark_name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    for run in trange(n_runs):

        path = input_path.joinpath(method_name, f"{run:03d}.csv")
        frame = load_frame(path, run)

        scaler = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
        quartile = 1 + scaler.fit_transform(frame.evaluation.to_numpy().reshape(-1, 1)).squeeze()

        g = sns.pairplot(frame.assign(quartile=quartile), vars=variables,
                         hue="epoch", palette="vlag",
                         height=height, aspect=aspect, corner=True)
        for ext in extension:
            g.savefig(output_path.joinpath(f"scatter_{run:03d}_{context}_{suffix}.{ext}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
