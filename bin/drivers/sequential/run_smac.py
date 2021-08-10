import sys
import click
import yaml

import numpy as np

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory import RunHistory
# from smac.tae.execute_func import ExecuteTAFuncDict

from bore_experiments.benchmarks import make_benchmark
from bore_experiments.utils import make_name, SMACLogs

from pathlib import Path


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="smac")
@click.option("--num-runs", "-n", default=20)
@click.option("--run-start", default=0)
@click.option("--num-iterations", "-i", default=500)
@click.option("--input-dir", default="datasets/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations, input_dir, output_dir):

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               input_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict()
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    def objective(config, seed):
        return benchmark.evaluate(config).value

    for run_id in range(run_start, num_runs):

        random_state = np.random.RandomState(run_id)
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": num_iterations,
                             "cs": benchmark.get_config_space(),
                             "deterministic": "true",
                             "output_dir": "foo/"})
        run_history = RunHistory()

        smac = SMAC4HPO(scenario=scenario, tae_runner=objective,
                        runhistory=run_history, rng=random_state)
        smac.optimize()

        data = SMACLogs(run_history).to_frame()
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
