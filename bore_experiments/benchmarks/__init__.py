from .synthetic import (Ackley, Branin, GoldsteinPrice, Rosenbrock,
                        SixHumpCamel, StyblinskiTang, Michalewicz, Hartmann3D,
                        Hartmann6D, Forrester)
from .tabular import FCNet, FCNetAlt
from .surrogate import BOHBSurrogate
from .racing import (UCBF110RacingLine, ETHZORCARacingLine,
                     ETHZMobilORCARacingLine)

benchmarks = dict(
    forrester=Forrester,
    branin=Branin,
    goldstein_price=GoldsteinPrice,
    six_hump_camel=SixHumpCamel,
    ackley=Ackley,
    rosenbrock=Rosenbrock,
    styblinski_tang=StyblinskiTang,
    michalewicz=Michalewicz,
    hartmann3d=Hartmann3D,
    hartmann6d=Hartmann6D,
    fcnet=FCNet,
    fcnet_alt=FCNetAlt,
    bohb_surrogate=BOHBSurrogate,
    ucb_f110_racing=UCBF110RacingLine,
    ethz_orca_racing=ETHZORCARacingLine,
    ethz_mobil_orca_racing=ETHZMobilORCARacingLine
)


def make_benchmark(benchmark_name, dimensions=None, dataset_name=None,
                   input_dir=None):

    Benchmark = benchmarks[benchmark_name]

    kws = {}
    if any(map(benchmark_name.startswith, ("fcnet", "bohb_surrogate"))):
        assert dataset_name is not None, "must specify dataset name"
        assert input_dir is not None, "must specify data directory"
        kws["dataset_name"] = dataset_name
        kws["input_dir"] = input_dir

    if benchmark_name in ("michalewicz", "styblinski_tang", "rosenbrock", "ackley"):
        assert dimensions is not None, "must specify dimensions"
        kws["dimensions"] = dimensions

    return Benchmark(**kws)
