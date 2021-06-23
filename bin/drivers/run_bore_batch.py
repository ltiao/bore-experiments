import sys
import click
import yaml

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

from bore.data import Record
from bore.models import BatchMaximizableSequential
from bore.optimizers.svgd.base import DistortionExpDecay
from bore.plugins.hpbandster.types import (DenseConfigurationSpace,
                                           DenseConfiguration, array_from_dict,
                                           dict_from_array)

from bore_experiments.benchmarks import make_benchmark
from bore_experiments.utils import make_name

from pathlib import Path
from functools import partial
from datetime import datetime
from tqdm import trange


def get_steps_per_epoch(batch_size, dataset_size):
    return int(np.ceil(np.true_divide(dataset_size, batch_size)))


def is_unique(record):

    def func(res):
        return not record.is_duplicate(res.x)

    return func


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="bore-batch")
@click.option("--num-runs", "-n", default=20)
@click.option("--run-start", default=0)
@click.option("--num-iterations", "-i", default=500)
@click.option("--eta", default=3, help="Successive halving reduction factor.")
@click.option("--min-budget", default=100)
@click.option("--max-budget", default=100)
@click.option("--gamma", default=0.25, type=click.FloatRange(0., 1.),
              help="Quantile, or mixing proportion.")
@click.option("--num-random-init", default=10)
@click.option("--batch-size", default=4)
@click.option("--random-rate", default=0.1, type=click.FloatRange(0., 1.))
@click.option("--num-starts", default=5)
@click.option("--num-samples", default=1024)
@click.option("--batch-size-training", default=64)
@click.option("--num-steps-per-iter", default=100)
@click.option("--num-epochs", type=int)
@click.option("--optimizer", default="adam")
@click.option("--num-layers", default=2)
@click.option("--num-units", default=32)
@click.option("--activation", default="elu")
@click.option('--transform', default="sigmoid")
@click.option("--method", default="L-BFGS-B")
@click.option("--max-iter", default=1000)
@click.option("--step-size", default=1e-3)
@click.option("--length-scale")
@click.option("--ftol", default=1e-9)
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations, eta, min_budget, max_budget, gamma,
         num_random_init, batch_size, random_rate, num_starts, num_samples,
         batch_size_training, num_steps_per_iter, num_epochs, optimizer,
         num_layers, num_units, activation, transform, method, max_iter,
         step_size, length_scale, ftol, input_dir, output_dir):

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               data_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(eta=eta, min_budget=min_budget, max_budget=max_budget,
                   gamma=gamma, num_random_init=num_random_init,
                   batch_size=batch_size, random_rate=random_rate,
                   num_starts=num_starts, num_samples=num_samples,
                   batch_size_training=batch_size_training,
                   num_steps_per_iter=num_steps_per_iter,
                   num_epochs=num_epochs, optimizer=optimizer,
                   num_layers=num_layers, num_units=num_units,
                   activation=activation, transform=transform, method=method,
                   max_iter=max_iter, step_size=step_size,
                   length_scale=length_scale, ftol=ftol)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    # def func(array):
    #     config = dict_from_array(config_space, x_next)
    #     return benchmark.evaluate(config).value

    for run_id in trange(run_start, num_runs, unit="run"):

        t_start = datetime.now()

        frames = []

        random_state = np.random.RandomState(run_id)

        config_space = DenseConfigurationSpace(benchmark.get_config_space(),
                                               seed=run_id)
        input_dim = config_space.get_dimensions(sparse=False)
        bounds = config_space.get_bounds()

        model = BatchMaximizableSequential()
        model.add(Dense(num_units, input_dim=input_dim, activation=activation))
        for i in range(num_layers-1):
            model.add(Dense(num_units, activation=activation))
        model.add(Dense(1))

        model.compile(loss=BinaryCrossentropy(from_logits=True),
                      optimizer=optimizer)
        model.summary(print_fn=click.echo)

        record = Record()

        with trange(num_iterations) as iterations:

            for batch in iterations:

                if record.size() < num_random_init:
                    # config_batch = config_space.sample_configuration(size=batch_size)
                    # X_batch = [config.to_array() for config in config_batch]
                    X_batch = random_state.uniform(low=bounds.lb,
                                                   high=bounds.ub,
                                                   size=(batch_size, input_dim))
                else:
                    # construct binary classification problem
                    X, z = record.load_classification_data(gamma)

                    if num_epochs is None:
                        dataset_size = record.size()
                        steps_per_epoch = get_steps_per_epoch(batch_size_training,
                                                              dataset_size)
                        num_epochs = num_steps_per_iter // steps_per_epoch

                    # update classifier
                    model.fit(X, z, epochs=num_epochs,
                              batch_size=batch_size_training, verbose=False)

                    # suggest new candidate
                    # opt = model.argmax(method=method, bounds=bounds,
                    #                    num_starts=num_starts,
                    #                    num_samples=num_samples,
                    #                    options=dict(maxiter=max_iter, ftol=ftol),
                    #                    filter_fn=is_unique(record),
                    #                    print_fn=click.echo,
                    #                    random_state=random_state)

                    # if opt is None:
                    #     x_next = config_random.to_array()
                    #     config = config_random.get_dictionary()
                    # else:
                    #     x_next = opt.x
                    #     config = dict_from_array(config_space, x_next)

                    X_batch = model.argmax_batch(batch_size=batch_size,
                                                 n_iter=max_iter,
                                                 length_scale=length_scale,
                                                 step_size=step_size,
                                                 bounds=bounds,
                                                 random_state=random_state)

                # TODO(LT): Deliberately not doing broadcasting for now since
                # batch sizes are so small anyway. Can revisit later if there
                # is a compelling reason to do it.
                rows = []
                for j, x_next in enumerate(X_batch):

                    config = dict_from_array(config_space, x_next)

                    # evaluate
                    y_next = benchmark.evaluate(config).value

                    t = datetime.now()
                    delta = t - t_start

                    # update dataset
                    record.append(x=x_next, y=y_next)

                    row = dict(config)
                    row["batch"] = batch
                    row["loss"] = y_next
                    row["finished"] = delta.total_seconds()
                    rows.append(row)

                frame = pd.DataFrame(data=rows)  # .assign(batch=batch)
                frames.append(frame)

        data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
