#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = []

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Louis C. Tiao",
    author_email='louistiao@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Bayesian Optimization by Density-Ratio Estimation",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    name='bore-experiments',
    packages=find_packages(include=['bore_experiments', 'bore_experiments.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ltiao/bore-experiments',
    version='0.1.0',
    zip_safe=False,
)
