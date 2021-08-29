import numpy as np
import pandas as pd

from ..benchmarks import make_benchmark


GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499


def pt_to_in(x):
    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):
    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def sanitize(data, methods_mapping, benchmarks_mapping, datasets_mapping):

    return data.replace(dict(method=methods_mapping,
                             benchmark=benchmarks_mapping,
                             dataset_name=datasets_mapping)) \
               .rename(lambda s: s.replace('_', ' '), axis="columns")


def get_ci(ci):

    if ci is not None:

        try:
            return int(ci)
        except ValueError:
            assert ci == "sd", "if `ci` is not integer and not None, " \
                "it must be 'sd'!"
    return ci


def parse_benchmark_name(benchmark_name, input_dir=None):

    kws = dict()
    if benchmark_name.startswith("fcnet") or \
            benchmark_name.startswith("bohb_surrogate"):
        *head, dataset_name = benchmark_name.split('_')
        name = '_'.join(head)
        kws = dict(dataset_name=dataset_name, input_dir=input_dir)
    elif any(map(benchmark_name.startswith, ("styblinski_tang",
                                             "michalewicz",
                                             "rosenbrock",
                                             "ackley"))):
        *head, dimensions_str = benchmark_name.split('_')
        name = '_'.join(head)
        dimensions = int(dimensions_str[:-1])
        kws = dict(dimensions=dimensions)
    else:
        name = benchmark_name

    return name, kws


def get_loss_min(name, kws):

    benchmark = make_benchmark(name, **kws)
    loss_min = benchmark.get_minimum()

    return loss_min


def load_frame(path, run, loss_min=None, loss_key="loss", sort_key="finished",
               duration_key=None, elapsed_key="finished"):

    frame = pd.read_csv(path, index_col=0)

    if sort_key in frame:
        # sort `sort_key` and drop old index
        frame.sort_values(by=sort_key, axis="index", ascending=True, inplace=True)
        frame.reset_index(drop=True, inplace=True)

    if duration_key is None:
        elapsed = frame[elapsed_key] if elapsed_key in frame else None
    else:
        duration = frame[duration_key] if duration_key in frame else None
        elapsed = duration.cumsum()

    loss = frame[loss_key]
    best = loss.cummin()

    frame = frame.assign(run=run, best=best, elapsed=elapsed,
                         evaluation=frame.index+1)

    # if "epoch" not in frame:
    # frame = frame.assign(epoch=50)

    if "epoch" in frame:
        target = frame.groupby(by="task").epoch.max()
        resource = frame.epoch.cumsum()
        frame = frame.assign(target=target, resource=resource)

    if loss_min is not None:
        error = loss.sub(loss_min).abs()
        regret = error.cummin()
        frame = frame.assign(error=error, regret=regret)

    return frame


def extract_series(frame, index="elapsed", column="regret"):

    frame_new = frame.set_index(index)
    series = frame_new[column]

    # (0) save last timestamp and value
    # tail = series.tail(n=1)
    # (1) de-duplicate the values (significantly speed-up
    # subsequent processing)
    # (2) de-duplicate the indices (it is entirely possible
    # for some epoch of two different tasks to complete
    # at the *exact* same time; we take the one with the
    # smaller value)
    # (3) add back last timestamp and value which can get
    # lost in step (1)
    series_new = series.drop_duplicates(keep="first") \
                       .groupby(level=index).min()
    # .append(tail)
    return series_new


def merge_stack_series(series_dict, run_key="run", y_key="regret"):

    frame = pd.DataFrame(series_dict)

    # fill missing values by propagating previous observation
    frame.ffill(axis="index", inplace=True)

    # NaNs can only remain if there are no previous observations
    # i.e. these occur at the beginning rows.
    # drop rows until all runs have recorded observations.
    frame.dropna(how="any", axis="index", inplace=True)

    frame.columns.name = run_key

    stacked = frame.stack(level=run_key)
    stacked.name = y_key
    stacked_frame = stacked.reset_index()

    return stacked_frame
