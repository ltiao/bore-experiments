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
@click.argument("name")
@click.argument("input_dir", default="results/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.argument("output_dir", default="figures/",
                type=click.Path(file_okay=False, dir_okay=True))
@click.option('--benchmarks', '-b', multiple=True)
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
def main(name, input_dir, output_dir, benchmarks, num_runs, methods, ci,
         duration_key, legend, ymin, ymax, ylabel, use_tex, transparent,
         context, style, palette, width, height, aspect, dpi, extension,
         config_file):

    if height is None:
        height = width / aspect
    # figsize = size(width, aspect)
    figsize = (width, height)

    print(figsize)

    suffix = f"{width*dpi:.0f}x{height*dpi:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": use_tex,
        # "text.usetex": False,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)

    input_path = Path(input_dir)
    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_file) if config_file else {}
    methods_mapping = config.get("methods", {})
    benchmarks_mapping = config.get("benchmarks", {})
    datasets_mapping = config.get("datasets", {})

    legend = "auto" if legend else False

    frames = []
    frames_merged = []

    for benchmark in benchmarks:

        benchmark_name, benchmark_options = parse_benchmark_name(benchmark, input_dir="datasets/")
        loss_min = get_loss_min(benchmark_name, benchmark_options)

        for method in methods:

            with input_path.joinpath(benchmark, method, "options.yaml").open('r') as f:
                options = yaml.safe_load(f)
                if "method" in options:
                    options.pop("method")  # TODO(LT): rename to distinguish as "acquisition opt method"

            method_name = "-".join(method.split("-")[:-1])

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

    # # fig, ax = plt.subplots()
    # # sns.despine(fig=fig, ax=ax, top=True)

    # # sns.scatterplot(x="elapsed", y="error",
    # #                 hue="batch",  # hue_order=hue_order,
    # #                 style="method",  # style_order=style_order,
    # #                 # units="run", estimator=None,
    # #                 # ci=get_ci(ci), err_kws=dict(edgecolor='none'),
    # #                 palette="viridis", alpha=0.6,
    # #                 legend=legend, data=data, ax=ax)

    # # # ax.set_xlabel("evaluations")
    # # ax.set_ylabel(ylabel)

    # # ax.set_yscale("log")
    # # # ax.set_ylim(ymin, ymax)

    # # plt.tight_layout()

    # # for ext in extension:
    # #     fig.savefig(output_path.joinpath(f"sanity_{context}_{suffix}.{ext}"),
    # #                 dpi=dpi, transparent=transparent)

    # # plt.clf()

    # # return 0

    g = sns.relplot(x="batch", y="regret",
                    col="benchmark", col_wrap=3,
                    # col="dimensions",  # hue_order=hue_order,
                    hue="method",  # style_order=style_order,
                    # units="run", estimator=None, linewidth=0.25,
                    # style="method", markers=True, dashes=False,
                    ci=get_ci(ci), err_style="band",  # err_kws=dict(edgecolor='none'),
                    facet_kws=dict(sharey=False),
                    height=height, aspect=aspect, kind="line",
                    alpha=0.6, legend=legend, data=data)
    g = g.set(yscale="log")
    g = g.set_axis_labels("batch", "regret")  # "best lap time [s]")
    g = g.set_titles(r"{col_name}")
    # g = g.set_titles(r"dataset -- {col_name}")
    # g = g.set_titles(r"{col_name} ($D={col_name}$)")
    # g = g.set_titles(r"{row_name} ($D={col_name}$)")

    for ext in extension:
        g.savefig(output_path.joinpath(f"grid_{context}_{suffix}.{ext}"),
                  dpi=dpi, transparent=transparent)

    return 0

    # # fig, ax = plt.subplots()
    # # sns.despine(fig=fig, ax=ax, top=True)

    # # sns.scatterplot(x="evaluation", y="best",
    # #                 hue="batch",  # hue_order=hue_order,
    # #                 style="method",  # style_order=style_order,
    # #                 # units="run", estimator=None,
    # #                 # ci=get_ci(ci), err_kws=dict(edgecolor='none'),
    # #                 palette="viridis", alpha=0.6,
    # #                 legend=legend, data=data, ax=ax)

    # # # ax.set_xlabel("evaluations")
    # # ax.set_ylabel(ylabel)

    # # # ax.set_yscale("log")
    # # # ax.set_ylim(ymin, ymax)

    # # plt.tight_layout()

    # # for ext in extension:
    # #     fig.savefig(output_path.joinpath(f"sanity_{context}_{suffix}.{ext}"),
    # #                 dpi=dpi, transparent=transparent)

    # # plt.clf()

    data_merged = pd.concat(frames_merged, axis="index", ignore_index=True, sort=True)
    data_merged = sanitize(data_merged, methods_mapping=methods_mapping,
                           benchmarks_mapping=benchmarks_mapping)

    # hue_order = style_order = list(map(method_names_mapping.get, methods))

    # fig, ax = plt.subplots()

    # sns.lineplot(x="batch", y="regret",
    #              hue="batch size",  # hue_order=hue_order,
    #              style="method",  # style_order=style_order,
    #              # units="run", estimator=None,
    #              palette="flare",
    #              ci=get_ci(ci), err_kws=dict(edgecolor='none'),
    #              legend=legend, data=data, ax=ax)

    # ax.set_xlabel("batch")
    # ax.set_ylabel(ylabel)

    # ax.set_yscale("log")
    # ax.set_ylim(ymin, ymax)

    # # ax.legend(ncol=2, loc="upper right")
    # # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    # #           ncol=2, borderaxespad=0.)
    # # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # sns.despine(fig=fig, ax=ax)

    # plt.tight_layout()

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"regret_iterations_{context}_{suffix}.{ext}"),
    #                 dpi=dpi, transparent=transparent)

    # plt.clf()

    # # return 0

    # print(data_merged)

    # g = sns.relplot(x="elapsed", y="regret",
    #                 col="batch size",  # hue_order=hue_order,
    #                 hue="method",  # style_order=style_order,
    #                 # units="run", estimator=None,
    #                 # ci=get_ci(ci), err_kws=dict(edgecolor='none'),
    #                 height=height, aspect=aspect, kind="line",
    #                 palette="viridis", alpha=0.6,
    #                 legend=legend, data=data_merged)
    # g = g.set(yscale="log")
    # g = g.set_axis_labels("wall-clock time elapsed [s]", "regret")

    # for ext in extension:
    #     g.savefig(output_path.joinpath(f"grid_{context}_{suffix}.{ext}"),
    #               dpi=dpi, transparent=transparent)

    # fig, ax = plt.subplots()
    # sns.despine(fig=fig, ax=ax, top=True)

    # sns.lineplot(x="elapsed", y="regret",
    #              hue="batch size",  # hue_order=hue_order,
    #              style="method",  # style_order=style_order,
    #              # units="run", estimator=None,
    #              ci=get_ci(ci), err_kws=dict(edgecolor='none'),
    #              legend=legend, data=data_merged, ax=ax)

    # ax.set_xlabel("wall-clock time elapsed [s]")
    # ax.set_ylabel(ylabel)

    # ax.set_yscale("log")
    # ax.set_ylim(ymin, ymax)

    # plt.tight_layout()

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"regret_elapsed_{context}_{suffix}.{ext}"),
    #                 dpi=dpi, transparent=transparent)

    # plt.clf()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
