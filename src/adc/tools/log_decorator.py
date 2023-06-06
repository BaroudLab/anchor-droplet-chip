import functools
import logging

import numpy as np


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [
            (
                f"{repr(a)}"
                if not isinstance(a, np.ndarray)
                else f"numpy array of shape {a.shape} `{a[0]}`"
            )
            for a in args
        ]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logging.debug(
            f"function `{func.__name__}` called with args `{signature}`"
        )
        try:
            result = func(*args, **kwargs)
            logging.debug(f"function `{func.__name__}` returns `{result}`")
            return result
        except Exception as e:
            logging.exception(
                f"Exception raised in `{func.__name__}`. exception: `{str(e)}`"
            )
            raise e

    return wrapper
