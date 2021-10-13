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
                                             parse_benchmark_name, sanitize,
                                             get_ci)


@click.command()
# @click.argument("name")
@click.argument("benchmark")
@click.argument("input_dir", default="results/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--num-runs', '-n', default=20)
@click.option('--methods', '-m', multiple=True)
@click.option('--ci')
@click.option('--show-runs/--no-show-runs', default=False)
@click.option('--show-title/--no-show-title', default=True)
@click.option('--duration-key', default=None)
@click.option('--legend/--no-legend', default=True)
@click.option('--ymin', type=float)
@click.option('--ymax', type=float)
@click.option('--ylabel', default="simple regret")
@click.option('--use-tex/--no-use-tex', default=True)
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
def main(benchmark, input_dir, output_dir, num_runs, methods, ci, show_runs,
         show_title,
         duration_key, legend, ymin, ymax, ylabel, use_tex, transparent,
         context, style, palette, width, height, aspect, dpi, extension,
         config_file):

    if height is None:
        height = width / aspect
    # figsize = size(width, aspect)
    figsize = (width, height)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": use_tex,
        # "text.usetex": False,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir)
    output_path = Path(output_dir).joinpath(benchmark)
    output_path.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_file) if config_file else {}
    methods_mapping = config.get("methods", {})
    benchmarks_mapping = config.get("benchmarks", {})
    datasets_mapping = config.get("datasets", {})

    legend = "auto" if legend else False

    frames = []
    frames_merged = []

    benchmark_name, benchmark_options = parse_benchmark_name(benchmark, input_dir="datasets/")
    loss_min = get_loss_min(benchmark_name, benchmark_options)

    for method in methods:

        # method_name = f"{method}"  # TODO(LT): revisit naming scheme later
        method_name = "-".join(method.split("-")[:-1])

        with input_path.joinpath(benchmark, method, "options.yaml").open('r') as f:
            options = yaml.safe_load(f)
            if "method" in options:
                options.pop("method")  # TODO(LT): rename to distinguish as "acquisition opt method"

        series = {}
        for run in range(num_runs):

            path = input_path.joinpath(benchmark, method, f"{run:03d}.csv")
            if not path.exists():
                click.secho(f"File `{path}` does not exist! Continuing...",
                            fg="yellow")
                continue

            frame = load_frame(path, run, loss_min=loss_min,
                               duration_key=duration_key) \
                .groupby(by="batch").min().reset_index()

            frames.append(frame.assign(benchmark=benchmark_name, method=method_name, **options, **benchmark_options))
            series[run] = extract_series(frame, index="elapsed", column="regret")

        frame_merged = merge_stack_series(series, y_key="regret")
        frames_merged.append(frame_merged.assign(benchmark=benchmark_name, method=method_name, **options, **benchmark_options))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=False)
    data = sanitize(data,
                    methods_mapping=methods_mapping,
                    benchmarks_mapping=benchmarks_mapping,
                    datasets_mapping=datasets_mapping)

    order = list(map(methods_mapping.get, ["-".join(method.split("-")[:-1]) for method in methods]))
    frame = data.groupby(["method", "run"]).last().regret.reset_index()

    fig, ax = plt.subplots()

    if show_title:
        ax.set_title("{benchmark}".format(benchmark=benchmarks_mapping[benchmark],
                                          **benchmark_options))

    sns.lineplot(x="batch", y="regret",
                 hue="method",  # hue_order=order,
                 ci=get_ci(ci),
                 err_style="band",  # err_kws=dict(edgecolor='none'),
                 # err_style="bars",  # err_kws=dict(edgecolor='none'),
                 alpha=0.8, zorder=2,
                 legend=legend, data=data, ax=ax)

    if show_runs:
        sns.lineplot(x="batch", y="regret",
                     hue="method",  # hue_order=order,
                     units="run", estimator=None,
                     linewidth=0.1, alpha=0.2, zorder=1,
                     legend=False, data=data, ax=ax)

    ax.set_xlabel("batch")

    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"resource_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.clf()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
