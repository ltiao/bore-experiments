import sys
import click
import yaml

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from bore_experiments.plotting.utils import (GOLDEN_RATIO, WIDTH, pt_to_in,
                                             load_frame, extract_series,
                                             merge_stack_series, get_loss_min,
                                             sanitize, get_ci)


@click.command()
@click.argument("benchmark_name")
@click.argument("input_dir", default="results/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--num-runs', '-n', default=20)
@click.option('--methods', '-m', multiple=True)
@click.option('--ci')
@click.option('--duration-key', default=None)
@click.option('--legend/--no-legend', default=True)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
@click.option('--ylabel', default="immediate regret")
@click.option('--transparent', is_flag=True)
@click.option('--context', default="paper")
@click.option('--style', default="ticks")
@click.option('--palette', default="muted")
@click.option('--width', '-w', type=float, default=pt_to_in(WIDTH))
@click.option('--height', '-h', type=float)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dpi', type=float)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--config-file", type=click.File('r'))
def main(benchmark_name, input_dir, output_dir, num_runs, methods, ci,
         duration_key, legend, ymin, ymax, ylabel, transparent, context, style,
         palette, width, height, aspect, dpi, extension, config_file):

    if height is None:
        height = width / aspect
    # figsize = size(width, aspect)
    figsize = (width, height)

    print(figsize)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir)
    output_path = Path(output_dir).joinpath(benchmark_name)
    output_path.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_file) if config_file else {}
    method_names_mapping = config.get("names", {})

    if legend:
        legend = "auto"

    loss_min = get_loss_min(benchmark_name, data_dir="datasets/fcnet_tabular_benchmarks")

    frames = []
    frames_merged = []

    for method in methods:

        series = {}
        for run in range(num_runs):

            path = input_path.joinpath(benchmark_name, method, f"{run:03d}.csv")
            if not path.exists():
                click.secho(f"File `{path}` does not exist! Continuing...",
                            fg="yellow")
                continue

            frame = load_frame(path, run, loss_min=loss_min,
                               duration_key=duration_key)
            frames.append(frame.assign(method=method))
            series[run] = extract_series(frame, index="elapsed", column="regret")

        frame_merged = merge_stack_series(series, y_key="regret")
        frames_merged.append(frame_merged.assign(method=method))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
    data = sanitize(data, mapping=method_names_mapping)

    data_merged = pd.concat(frames_merged, axis="index", ignore_index=True, sort=True)
    data_merged = sanitize(data_merged, mapping=method_names_mapping)

    hue_order = style_order = list(map(method_names_mapping.get, methods))

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="evaluation", y="regret",
                 hue="method",  # hue_order=hue_order,
                 style="method",  # style_order=style_order,
                 # units="run", estimator=None,
                 ci=get_ci(ci), err_kws=dict(edgecolor='none'),
                 legend=legend, data=data, ax=ax)

    ax.set_xlabel("evaluations")
    ax.set_ylabel(ylabel)

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"regret_iterations_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.clf()

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="elapsed", y="regret",
                 hue="method",  # hue_order=hue_order,
                 style="method",  # style_order=style_order,
                 # units="run", estimator=None,
                 ci=get_ci(ci), err_kws=dict(edgecolor='none'),
                 legend=legend, data=data_merged, ax=ax)

    ax.set_xlabel("wall-clock time elapsed [s]")
    ax.set_ylabel(ylabel)

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"regret_elapsed_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.clf()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
