import sys
import click
import yaml

import torch
import pandas as pd

from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
# from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from botorch.utils.sampling import draw_sobol_samples
# from sklearn.preprocessing import StandardScaler

from pathlib import Path
from tqdm import trange
from datetime import datetime

from bore.plugins.hpbandster.types import DenseConfigurationSpace, DenseConfiguration
from bore_experiments.benchmarks import make_benchmark
from bore_experiments.utils import make_name


def dict_from_tensor(tensor, cs):
    array = tensor.detach().cpu().numpy()
    config = DenseConfiguration.from_array(cs, array_dense=array)
    return config.get_dictionary()


def create_bounds(bounds, device=None, dtype=None):
    return torch.tensor([bounds.lb, bounds.ub], device=device, dtype=dtype)


@click.command()
@click.argument("benchmark_name")
@click.option("--dataset-name", help="Dataset to use for `fcnet` benchmark.")
@click.option("--dimensions", type=int, help="Dimensions to use for `michalewicz` and `styblinski_tang` benchmarks.")
@click.option("--method-name", default="gp-botorch-batch")
@click.option("--num-runs", "-n", default=20)
@click.option("--run-start", default=0)
@click.option("--num-iterations", "-i", default=500)
# @click.option("--acquisition-name", default="EI")
# @click.option("--acquisition-optimizer-name", default="lbfgs",
#               type=click.Choice(["lbfgs", "DIRECT", "CMA"]))
@click.option("--gamma", default=0., type=click.FloatRange(0., 1.),
              help="Quantile, or mixing proportion.")
@click.option("--num-random-init", default=10)
@click.option("--mc-samples", default=256)
@click.option("--batch-size", default=4)
@click.option("--num-restarts", default=10)
@click.option("--raw-samples", default=512)
# @click.option('--use-ard', is_flag=True)
# @click.option('--use-input-warping', is_flag=True)
@click.option("--input-dir", default="datasets/fcnet_tabular_benchmarks",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations,
         # acquisition_name,
         # acquisition_optimizer_name,
         gamma, num_random_init, mc_samples, batch_size,
         num_restarts, raw_samples,
         # use_ard,
         # use_input_warping,
         input_dir, output_dir):

    # TODO(LT): Turn into options
    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               data_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(gamma=gamma, num_random_init=num_random_init,
                   mc_samples=mc_samples, batch_size=batch_size,
                   num_restarts=num_restarts, raw_samples=raw_samples)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    config_space = DenseConfigurationSpace(benchmark.get_config_space())
    bounds = create_bounds(config_space.get_bounds(), device=device, dtype=dtype)
    dim = config_space.get_dimensions()

    def func(tensor, *args, **kwargs):
        """
        Wrapper that receives and returns torch.Tensor
        """
        config = dict_from_tensor(tensor, cs=config_space)
        # turn into maximization problem
        res = - benchmark.evaluate(config).value
        return torch.tensor(res, device=device, dtype=dtype)

    # TODO(LT): make initial value option
    noise_variance_init = 1e-3

    for run_id in trange(run_start, num_runs, unit="run"):

        t_start = datetime.now()

        frames = []

        features = []
        targets = []

        noise_variance = torch.tensor(noise_variance_init, device=device, dtype=dtype)

        with trange(num_iterations) as iterations:

            for batch in iterations:

                if len(targets) < num_random_init:
                    # click.echo(f"Completed {i}/{num_random_init} initial runs. "
                    #            "Suggesting random candidate...")
                    # TODO(LT): support random seed
                    X_batch = torch.rand(size=(batch_size, dim), device=device, dtype=dtype)
                else:
                    # scaler = StandardScaler()

                    # construct dataset
                    X = torch.vstack(features)
                    y = torch.hstack(targets).unsqueeze(axis=-1)
                    z = standardize(y)

                    # TODO(LT): persist model state
                    # construct model
                    # model = FixedNoiseGP(X, standardize(y), noise_variance.expand_as(y),
                    model = FixedNoiseGP(X, z, noise_variance.expand_as(y),
                                         input_transform=None).to(X)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)

                    # update model
                    fit_gpytorch_model(mll)

                    # construct acquisition function
                    # TODO(LT): standardize?
                    tau = torch.quantile(z, q=1-gamma)
                    iterations.set_postfix(tau=tau.item())

                    qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)

                    ei = qExpectedImprovement(model=model, best_f=tau,
                                              sampler=qmc_sampler)

                    # optimize acquisition function
                    # TODO(LT): turn kwargs into command-line options
                    X_batch, b = optimize_acqf(acq_function=ei, bounds=bounds,
                                               q=batch_size,
                                               num_restarts=num_restarts,
                                               raw_samples=raw_samples,
                                               options=dict(batch_limit=5,
                                                            maxiter=200))

                # TODO(LT): Deliberately not doing broadcasting for now since
                # batch sizes are so small anyway. Can revisit later if there
                # is a compelling reason to do it.
                rows = []
                for x_new in X_batch:

                    y_new = func(x_new)

                    t = datetime.now()
                    delta = t - t_start

                    # update dataset
                    features.append(x_new)
                    targets.append(y_new)

                    row = dict_from_tensor(x_new, cs=config_space)
                    row["loss"] = y_new.item()
                    row["finished"] = delta.total_seconds()
                    rows.append(row)

                frame = pd.DataFrame(data=rows).assign(batch=batch)
                frames.append(frame)

        data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
        data.to_csv(output_path.joinpath(f"{run_id:03d}.csv"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
