import sys
import click
import yaml

import numpy as np
import pandas as pd

from scipy.stats.qmc import LatinHypercube

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

from bore.data import Record
from bore.math import ceil_divide, steps_per_epoch
from bore.models import BatchMaximizableSequential
from bore.optimizers.svgd.base import DistortionExpDecay
from bore.plugins.hpbandster.types import (DenseConfigurationSpace,
                                           dict_from_array)

from bore_experiments.benchmarks import make_benchmark
from bore_experiments.utils import make_name

from pathlib import Path
from datetime import datetime
from tqdm import trange


def create_model(input_dim, num_units, num_layers, activation, optimizer):

    model = BatchMaximizableSequential()
    model.add(Dense(num_units, input_dim=input_dim, activation=activation))
    for i in range(num_layers-1):
        model.add(Dense(num_units, activation=activation))
    model.add(Dense(1))

    model.compile(loss=BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer)
    model.summary(print_fn=click.echo)
    return model


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
@click.option("--init-method", default="uniform",
              type=click.Choice(["uniform", "latin_hypercube"]))
@click.option("--batch-size", default=4)
@click.option("--batch-size-training", default=64)
@click.option("--num-steps-per-iter", default=100)
@click.option("--num-epochs", type=int)
@click.option("--optimizer", default="adam")
@click.option("--num-layers", default=2)
@click.option("--num-units", default=32)
@click.option("--activation", default="elu")
@click.option('--transform', default="sigmoid")
@click.option("--max-iter", default=1000)
@click.option("--step-size", default=1e-3)
@click.option("--length-scale")
@click.option("--tau", default=1.0)
@click.option("--lambd", type=float)
@click.option("--input-dir", default="datasets/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations, eta, min_budget, max_budget, gamma,
         num_random_init, init_method, batch_size,
         batch_size_training, num_steps_per_iter, num_epochs, optimizer,
         num_layers, num_units, activation, transform, max_iter,
         step_size, length_scale, tau, lambd, input_dir, output_dir):

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               input_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(eta=eta, min_budget=min_budget, max_budget=max_budget,
                   gamma=gamma, num_random_init=num_random_init,
                   batch_size=batch_size,
                   batch_size_training=batch_size_training,
                   num_steps_per_iter=num_steps_per_iter,
                   num_epochs=num_epochs, optimizer=optimizer,
                   num_layers=num_layers, num_units=num_units,
                   activation=activation, transform=transform,
                   max_iter=max_iter, step_size=step_size, tau=tau,
                   lambd=lambd, length_scale=length_scale,
                   init_method=init_method)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    n_init_batches = ceil_divide(num_random_init, batch_size) * batch_size

    for run_id in trange(run_start, num_runs, unit="run"):

        run_begin_t = batch_end_t_adj = batch_end_t = datetime.now()

        frames = []

        random_state = np.random.RandomState(seed=run_id)

        config_space = DenseConfigurationSpace(benchmark.get_config_space(),
                                               seed=run_id)
        input_dim = config_space.get_dimensions(sparse=False)
        bounds = config_space.get_bounds()

        model = create_model(input_dim, num_units, num_layers, activation,
                             optimizer)

        record = Record()

        if init_method == "latin_hypercube":
            sampler = LatinHypercube(d=input_dim, seed=run_id)
            X_init = (bounds.ub - bounds.lb) * sampler.random(n_init_batches) + bounds.lb
        else:
            X_init = random_state.uniform(low=bounds.lb,
                                          high=bounds.ub,
                                          size=(n_init_batches, input_dim))

        with trange(num_iterations) as iterations:

            for batch in iterations:

                if record.size() < num_random_init:
                    # config_batch = config_space.sample_configuration(size=batch_size)
                    # X_batch = [config.to_array() for config in config_batch]
                    a = batch * batch_size
                    b = a + batch_size
                    X_batch = X_init[a:b]
                else:
                    # construct binary classification problem
                    X, z = record.load_classification_data(gamma)

                    if num_epochs is None:
                        dataset_size = record.size()
                        num_steps = steps_per_epoch(batch_size_training, dataset_size)
                        num_epochs = num_steps_per_iter // num_steps

                    # update classifier
                    model.fit(X, z, epochs=num_epochs,
                              batch_size=batch_size_training, verbose=False)

                    X_batch = model.argmax_batch(batch_size=batch_size,
                                                 n_iter=max_iter,
                                                 length_scale=length_scale,
                                                 step_size=step_size,
                                                 bounds=bounds,
                                                 tau=tau, lambd=lambd,
                                                 # print_fn=click.echo,
                                                 random_state=random_state)

                # begin batch evaluation
                batch_begin_t = datetime.now()
                decision_duration = batch_begin_t - batch_end_t
                batch_begin_t_adj = batch_end_t_adj + decision_duration

                eval_end_times = []

                # TODO(LT): Deliberately not doing broadcasting for now since
                # batch sizes are so small anyway. Can revisit later if there
                # is a compelling reason to do it.
                rows = []
                for j, x_next in enumerate(X_batch):

                    config = dict_from_array(config_space, x_next)

                    # eval begin time
                    eval_begin_t = datetime.now()

                    # evaluate blackbox objective
                    y_next = benchmark.evaluate(config).value

                    # eval end time
                    eval_end_t = datetime.now()

                    # eval duration
                    eval_duration = eval_end_t - eval_begin_t

                    # adjusted eval end time is the duration added to the
                    # time at which batch eval was started
                    eval_end_t_adj = batch_begin_t_adj + eval_duration

                    eval_end_times.append(eval_end_t_adj)
                    elapsed = eval_end_t_adj - run_begin_t

                    # update dataset
                    record.append(x=x_next, y=y_next)

                    row = dict(config)
                    row["loss"] = y_next
                    row["cost_eval"] = eval_duration.total_seconds()
                    row["finished"] = elapsed.total_seconds()
                    rows.append(row)

                batch_end_t = datetime.now()
                batch_end_t_adj = max(eval_end_times)

                frame = pd.DataFrame(data=rows) \
                          .assign(batch=batch,
                                  cost_decision=decision_duration.total_seconds())
                frames.append(frame)

        data = pd.concat(frames, axis="index", ignore_index=True)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
