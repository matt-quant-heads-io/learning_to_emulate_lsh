from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="algorithmic_ml_sandbox",
    version="0.1.0",
    install_requires=[
        "keras",
        "pymongo",
        "pillow",
        "numpy",
        "black",
        "redis[hiredis]",
        "redis-om",
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
