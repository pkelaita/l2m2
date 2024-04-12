from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="l2m2",
    version="0.0.5",
    packages=find_packages(exclude=["tests", "integration_tests"]),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    long_description=readme,
    long_description_content_type="text/markdown",
)
