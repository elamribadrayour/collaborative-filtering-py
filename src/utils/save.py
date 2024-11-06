import os
import numpy


def save(cache_dir: str, rankings: numpy.ndarray) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    numpy.save(os.path.join(cache_dir, "rankings.npy"), rankings)
