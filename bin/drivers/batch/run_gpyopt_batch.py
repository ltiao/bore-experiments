import sys
import click
import yaml

import numpy as np
import pandas as pd
import ConfigSpace as CS

import GPyOpt
from pathlib import Path

from bore.math import ceil_divide
from bore_experiments.benchmarks import make_benchmark
from bore_experiments.utils import make_name
from tqdm import trange


def dict_from_array(array, cs):

    kws = {}
    for i, h in enumerate(cs.get_hyperparameters()):
        if isinstance(h, CS.OrdinalHyperparameter):
            value = h.sequence[int(array[0, i])]
        elif isinstance(h, CS.CategoricalHyperparameter):
            value = h.choices[int(array[0, i])]
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            value = int(array[0, i])
        else:
            value = array[0, i]
        kws[h.name] = value
    return kws


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="gpyopt-batch")
@click.option("--num-runs", "-n", default=20)
@click.option("--run-start", default=0)
@click.option("--num-iterations", "-i", default=500)
@click.option("--acquisition-name", default="EI")
@click.option("--acquisition-optimizer-name", default="lbfgs",
              type=click.Choice(["lbfgs", "DIRECT", "CMA"]))
@click.option("--num-random-init", default=10)
@click.option("--batch-size", default=4)
@click.option('--use-ard', is_flag=True)
@click.option('--use-input-warping', is_flag=True)
@click.option('--standardize-targets/--no-standardize-targets', default=True)
@click.option("--input-dir", default="datasets/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations, acquisition_name,
         acquisition_optimizer_name, num_random_init, batch_size, use_ard,
         use_input_warping, standardize_targets, input_dir, output_dir):

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               input_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(batch_size=batch_size, acquisition_name=acquisition_name,
                   acquisition_optimizer_name=acquisition_optimizer_name,
                   use_ard=use_ard, use_input_warping=use_input_warping,
                   standardize_targets=standardize_targets)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    space = benchmark.get_domain()
    config_space = benchmark.get_config_space()

    def func(array, *args, **kwargs):
        kws = dict_from_array(array, cs=config_space)
        return benchmark.evaluate(kws).value

    model_type = "input_warped_GP" if use_input_warping else "GP"

    initial_design_numdata = ceil_divide(num_random_init, batch_size) * batch_size

    for run_id in trange(run_start, num_runs, unit="run"):

        BO = GPyOpt.methods.BayesianOptimization(f=func, domain=space,
                                                 initial_design_numdata=initial_design_numdata,
                                                 model_type=model_type,
                                                 ARD=use_ard,
                                                 normalize_Y=standardize_targets,
                                                 exact_feval=False,
                                                 acquisition_type=acquisition_name,
                                                 acquisition_optimizer_type=acquisition_optimizer_name,
                                                 batch_size=batch_size,
                                                 evaluator_type="local_penalization")
        BO.run_optimization(num_iterations,
                            # evaluations_file="bar/evaluations.csv",
                            # models_file="bar/models.csv",
                            verbosity=True)

        cost_decision_arr = np.hstack(BO.cost_decision)
        cost_eval_arr = np.hstack(BO.cost_eval)

        data = pd.DataFrame(data=BO.X, columns=[d["name"] for d in space]) \
                 .assign(loss=BO.Y, batch=lambda row: row.index // batch_size,
                         cost_decision=lambda row: cost_decision_arr[row.batch],
                         cost_eval=cost_eval_arr)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
