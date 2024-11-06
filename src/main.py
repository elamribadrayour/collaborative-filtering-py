import os
import logging

import numpy
from typer import Typer, Option

import utils.save
import utils.matrix
import utils.logger

app = Typer()


@app.command()
def run(cache_dir: str = Option(default=".cache")) -> None:
    utils.logger.init()
    numpy.random.seed(42)
    matrix = utils.matrix.get_random(num_users=100, num_items=200)
    rankings = utils.matrix.get_predictions(matrix=matrix, k=10)
    utils.save.save(cache_dir=cache_dir, rankings=rankings)


@app.command()
def help() -> None:
    pass


if __name__ == "__main__":
    app()
