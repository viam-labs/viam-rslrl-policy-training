from setuptools import find_packages, setup

# Viam custom training script package. Viam's launcher pip-installs this sdist and runs
# `python3 -m model.training`. Dependencies live in the training image
# (container/Dockerfile.train), so the job doesn't reinstall them on billed GPU time.
setup(
    name="rl-insert-train",
    version="0.1.0",
    packages=find_packages(include=["model", "rl_insert", "rl_insert.*"]),
    package_data={"rl_insert": ["assets/**/*"]},  # every asset; listing folders dropped assets/g2 once
    install_requires=[],
)
