from setuptools import find_packages, setup

# Viam custom training script for the MicroDuck walk. Viam's launcher pip-installs this sdist and runs
# `python3 -m model.training`, which this package's model/training.py forwards to duck_train.train.main.
# Dependencies live in the image (container/Dockerfile.duck).
setup(name="microduck-train", version="0.2.0", packages=find_packages(include=["model", "duck_train"]), install_requires=[])
