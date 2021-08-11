import sys
import click
import yaml

import torch
import pandas as pd

from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils import standardize

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
@click.option("--acquisition-name", default="q-EI",
              type=click.Choice(["q-EI", "q-KG"], case_sensitive=False))
# @click.option("--acquisition-optimizer-name", default="lbfgs",
#               type=click.Choice(["lbfgs", "DIRECT", "CMA"]))
@click.option("--gamma", default=0., type=click.FloatRange(0., 1.),
              help="Quantile, or mixing proportion.")
@click.option("--num-random-init", default=10)
@click.option("--batch-size", default=4)
@click.option("--num-fantasies", default=128)
@click.option("--mc-samples", default=256)
@click.option("--num-restarts", default=10)
@click.option("--raw-samples", default=512)
@click.option("--noise-variance-init", default=5e-2)
# @click.option('--use-ard', is_flag=True)
# @click.option('--use-input-warping', is_flag=True)
@click.option('--standardize-targets/--no-standardize-targets', default=True)
@click.option("--input-dir", default="datasets/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Input data directory.")
@click.option("--output-dir", default="results/",
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(benchmark_name, dataset_name, dimensions, method_name, num_runs,
         run_start, num_iterations,
         acquisition_name,
         # acquisition_optimizer_name,
         gamma, num_random_init, mc_samples, batch_size, num_fantasies,
         num_restarts, raw_samples, noise_variance_init,
         # use_ard,
         # use_input_warping,
         standardize_targets,
         input_dir, output_dir):

    # TODO(LT): Turn into options
    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    benchmark = make_benchmark(benchmark_name,
                               dimensions=dimensions,
                               dataset_name=dataset_name,
                               input_dir=input_dir)
    name = make_name(benchmark_name,
                     dimensions=dimensions,
                     dataset_name=dataset_name)

    output_path = Path(output_dir).joinpath(name, method_name)
    output_path.mkdir(parents=True, exist_ok=True)

    options = dict(gamma=gamma, num_random_init=num_random_init,
                   acquisition_name=acquisition_name,
                   mc_samples=mc_samples, batch_size=batch_size,
                   num_restarts=num_restarts, raw_samples=raw_samples,
                   num_fantasies=num_fantasies,
                   noise_variance_init=noise_variance_init,
                   standardize_targets=standardize_targets)
    with output_path.joinpath("options.yaml").open('w') as f:
        yaml.dump(options, f)

    config_space = DenseConfigurationSpace(benchmark.get_config_space())
    bounds = create_bounds(config_space.get_bounds(), device=device, dtype=dtype)
    input_dim = config_space.get_dimensions()

    def func(tensor, *args, **kwargs):
        """
        Wrapper that receives and returns torch.Tensor
        """
        config = dict_from_tensor(tensor, cs=config_space)
        # turn into maximization problem
        res = - benchmark.evaluate(config).value
        return torch.tensor(res, device=device, dtype=dtype)

    for run_id in trange(run_start, num_runs, unit="run"):

        run_begin_t = batch_end_t_adj = batch_end_t = datetime.now()

        frames = []

        features = []
        targets = []

        noise_variance = torch.tensor(noise_variance_init, device=device, dtype=dtype)
        state_dict = None

        with trange(num_iterations) as iterations:

            for batch in iterations:

                if len(targets) < num_random_init:
                    # click.echo(f"Completed {i}/{num_random_init} initial runs. "
                    #            "Suggesting random candidate...")
                    # TODO(LT): support random seed
                    X_batch = torch.rand(size=(batch_size, input_dim),
                                         device=device, dtype=dtype)
                else:

                    # construct dataset
                    X = torch.vstack(features)
                    y = torch.hstack(targets).unsqueeze(axis=-1)
                    y = standardize(y) if standardize_targets else y

                    # construct model
                    # model = FixedNoiseGP(X, standardize(y), noise_variance.expand_as(y),
                    model = FixedNoiseGP(X, y, noise_variance.expand_as(y),
                                         input_transform=None).to(X)
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)

                    if state_dict is not None:
                        model.load_state_dict(state_dict)

                    # update model
                    fit_gpytorch_model(mll)

                    # construct acquisition function
                    tau = torch.quantile(y, q=1-gamma)
                    iterations.set_postfix(tau=tau.item())

                    if acquisition_name == "q-KG":
                        assert num_fantasies is not None and num_fantasies > 0
                        acq = qKnowledgeGradient(model, num_fantasies=num_fantasies)
                    elif acquisition_name == "q-EI":
                        assert mc_samples is not None and mc_samples > 0
                        qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)
                        acq = qExpectedImprovement(model=model, best_f=tau,
                                                   sampler=qmc_sampler)

                    # optimize acquisition function
                    X_batch, b = optimize_acqf(acq_function=acq, bounds=bounds,
                                               q=batch_size,
                                               num_restarts=num_restarts,
                                               raw_samples=raw_samples,
                                               options=dict(batch_limit=5,
                                                            maxiter=200))

                    state_dict = model.state_dict()

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

                    # eval begin time
                    eval_begin_t = datetime.now()

                    # evaluate blackbox objective
                    y_next = func(x_next)

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
                    features.append(x_next)
                    targets.append(y_next)

                    row = dict_from_tensor(x_next, cs=config_space)
                    row["loss"] = - y_next.item()
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
