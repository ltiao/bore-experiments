import sys
import click
import yaml

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
def main(benchmark, input_dir, output_dir, num_runs, methods, ci,
         duration_key, legend, ymin, ymax, ylabel, use_tex, transparent,
         context, style, palette, width, height, aspect, dpi, extension,
         config_file):

    if height is None:
        height = width / aspect
    # figsize = size(width, aspect)
    figsize = (width, height)

    show_title = True

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
    # .assign(r=lambda row: row.resource / 50.)
    data = sanitize(data,
                    methods_mapping=methods_mapping,
                    benchmarks_mapping=benchmarks_mapping,
                    datasets_mapping=datasets_mapping)

    order = list(map(methods_mapping.get, ["-".join(method.split("-")[:-1]) for method in methods]))
    frame = data.groupby(["method", "run"]).last().regret.reset_index()

    print(order)
    print(data)
    print(frame)

    # fig, ax = plt.subplots()

    # sns.lineplot(x="batch", y="regret", hue="method",
    #              # hue_order=order,
    #              # hue="epoch",  # hue_order=hue_order,
    #              # style="method",  # style_order=style_order,
    #              # units="run", estimator=None,
    #              ci=get_ci(ci),
    #              # palette="viridis_r",
    #              # linewidth=0.4, alpha=0.4,
    #              legend=legend, data=data, ax=ax)
    # sns.lineplot(x="batch", y="regret", hue="method",
    #              # hue_order=order,
    #              # hue="epoch",  # hue_order=hue_order,
    #              # style="method",  # style_order=style_order,
    #              units="run", estimator=None,
    #              # ci=get_ci(ci), err_kws=dict(edgecolor='none'),
    #              # palette="viridis_r",
    #              linewidth=0.4, alpha=0.4,
    #              legend=False, data=data, ax=ax)

    # ax.set_ylabel(ylabel)
    # ax.set_yscale("log")

    # plt.tight_layout()

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"test_{context}_{suffix}.{ext}"),
    #                 dpi=dpi, transparent=transparent)

    # plt.clf()

    # return 0

    fig, ax = plt.subplots()

    # if show_title:
    #     ax.set_title(r"{benchmark} ($D={dimensions}$)".format(benchmark=benchmarks_mapping[benchmark_name], **benchmark_options))

    # sns.despine(fig=fig, ax=ax, top=True)
    # divider = make_axes_locatable(ax)

    sns.lineplot(x="batch", y="regret",
                 hue="method", hue_order=order,
                 ci=get_ci(ci),
                 err_style="band",  # err_kws=dict(edgecolor='none'),
                 # err_style="bars",  # err_kws=dict(edgecolor='none'),
                 alpha=0.8, zorder=2,
                 legend=legend, data=data, ax=ax)
    sns.lineplot(x="batch", y="regret",
                 hue="method", hue_order=order,
                 units="run", estimator=None,
                 linewidth=0.1, alpha=0.2, zorder=1,
                 legend=False, data=data, ax=ax)

    # ax.set_xlabel("wall-clock time elapsed [s]")
    # ax.set_xlabel(r"resource [\emph{epoch}]")
    # ax.set_xscale("log")

    ax.set_xlabel("batch")

    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    # ax.axvline(1.0, linestyle="dashed", linewidth=0.5, color="tab:gray",
    #            alpha=0.6, zorder=-1)

    # ax.legend(loc="lower left")

    # ax_y = divider.append_axes("right", size=0.5, pad=0.1, sharey=ax)

    # sns.boxplot(x="method", y="regret",  # hue="method",
    #             order=order,
    #               # hue="epoch",  # hue_order=hue_order,
    #               # style="method",  # style_order=style_order,
    #               # ci=get_ci(ci),
    #               # err_style="band",  # err_kws=dict(edgecolor='none'),
    #               # palette="viridis_r",
    #               # alpha=0.6,
    #               width=0.5,
    #               linewidth=0.5,
    #               fliersize=0.4,
    #               # legend=legend,
    #               # jitter=False,
    #               data=frame, ax=ax_y)

    # # ax_y.set_yticks([])
    # ax_y.set_ylabel(None)

    # ax_y.set_xticklabels([])
    # # ax_y.set_xticklabels(order, rotation=90, fontsize="xx-small")
    # ax_y.set_xlabel(None)

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"resource_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.clf()

    data_merged = pd.concat(frames_merged, axis="index", ignore_index=True, sort=True)
    data_merged = sanitize(data_merged,
                           methods_mapping=methods_mapping,
                           benchmarks_mapping=benchmarks_mapping,
                           datasets_mapping=datasets_mapping)

    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax, top=True)

    sns.lineplot(x="elapsed", y="regret", hue="method",
                 # hue="epoch",  # hue_order=hue_order,
                 # style="method",  # style_order=style_order,
                 ci=get_ci(ci),
                 err_style="band",  # err_kws=dict(edgecolor='none'),
                 # palette="viridis_r",
                 alpha=0.8, legend=legend, data=data_merged, ax=ax)
    sns.lineplot(x="elapsed", y="regret", hue="method",
                 # hue="epoch",  # hue_order=hue_order,
                 # style="method",  # style_order=style_order,
                 units="run", estimator=None,
                 # ci=get_ci(ci), err_kws=dict(edgecolor='none'),
                 # palette="viridis_r",
                 linewidth=0.1, alpha=0.2,
                 legend=False, data=data_merged, ax=ax)

    # ax.set_xlabel("wall-clock time elapsed [s]")
    ax.set_xlabel(r"time elapsed  [$s$]")
    ax.set_xscale("log")

    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    plt.tight_layout()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"time_elapsed_{context}_{suffix}.{ext}"),
                    dpi=dpi, transparent=transparent)

    plt.clf()

    return 0

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
