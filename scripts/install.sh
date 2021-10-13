#!/usr/bin/env bash

set -v
set -e   # fail if any command fails

# git clone https://github.com/ltiao/bore-experiments.git
cd bore-experiments/

# mkdir src
# cd src/

# git clone https://github.com/ltiao/GPyOpt.git
# git clone https://github.com/ltiao/nas_benchmarks.git

# git clone https://github.com/ltiao/bore.git
# cd bore/
# git fetch origin develop
# git checkout develop
# cd ..

# python -m pip install ConfigSpace==0.4.18 GPy==1.10.0 cvxpy==1.1.14 hyperopt hpbandster h5py
python -m pip install --no-deps -e src/GPyOpt
python -m pip install --no-deps -e src/bore
python -m pip install --no-cache-dir --no-deps -e src/nas_benchmarks
python -m pip install --no-cache-dir --no-deps -e .
