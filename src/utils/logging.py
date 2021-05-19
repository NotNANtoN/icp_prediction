import time


def log_time(func):
    """Decorator that (on verbose logging level) prints the elapsed time to the console."""

    def wrapper(v=0, verbose=0, *args, **kwargs):
        if v or verbose:
            print(f'Starting {func.__name__}...')
            start = time.time()
        result = func(*args, **kwargs)
        if v or verbose:
            end = time.time()
            print(f'{func.__name__} completed. Elapsed time: {(end - start):0.2f}s\n')
        return result

    return wrapper

# TODO: add log_time Class with __enter__ and __exit__ method for didactic purposes
