import datetime
import logging
from functools import wraps


def wrap_with_logging(logger=logging):
    def make_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            curtime = datetime.datetime.now().isoformat()
            logger.debug(f"Calling ({curtime}) {func.__name__}(*{args}, **{kwargs})")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"({curtime}) => {result}")
                return result
            except Exception as e:
                logger.warning(f"({curtime}) => {e}")
                raise

        wrapper.__logger__ = logger

        return wrapper

    return make_wrapper
