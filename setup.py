from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="algorithmic_ml_sandbox",
    version="0.3.0",
    install_requires=[
        "scann",
        "pandas==0.24.2",
        "tensorflow_datasets",
        "tensorflow",
        "matplotlib",
        "keras",
        "pymongo",
        "pillow",
        "numpy==1.16.3",
        "black",
        "scikit-learn==0.20.3",
        "scipy==1.2.1",
        "termcolor==1.1.0"
    ],
    author=["Matthew Siper", "Zehua Jiang"],
    author_email="siper.matthew@gmail.com",
    description="A project for evaluating artifical intelligence for the task of nearest neighbor search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matt-quant-heads-io/algorithmic_ml_sandbox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
