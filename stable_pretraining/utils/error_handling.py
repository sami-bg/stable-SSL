import time
from huggingface_hub.utils import HfHubHTTPError
import requests
from loguru import logger as logging
import sys
import os
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any


try:
    import wandb
except ImportError:
    wandb = None


def get_rank():
    """Get distributed training rank."""
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0


@contextmanager
def catch_errors():
    """Catch and log errors from all ranks before re-raising.

    Ensures errors appear in Slurm logs, wandb, and everywhere else.
    """
    try:
        yield

    except Exception as e:
        rank = get_rank()
        rank_prefix = f"[Rank {rank}] " if rank > 0 else ""

        error_msg = (
            f"\n{'=' * 80}\n"
            f"{rank_prefix}💥 EXCEPTION CAUGHT\n"
            f"{'=' * 80}\n"
            f"Type: {type(e).__name__}\n"
            f"Message: {str(e)}\n"
            f"{'=' * 80}\n"
            f"TRACEBACK:\n"
            f"{traceback.format_exc()}"
            f"{'=' * 80}\n"
        )

        # Log from ALL ranks (important for debugging distributed issues)
        logging.opt(depth=1).error(error_msg)

        # Direct prints to stderr/stdout (backup for Slurm logs)
        print(error_msg, file=sys.stderr, flush=True)
        print(error_msg, file=sys.stdout, flush=True)

        # Wandb logging (only from main process, with error handling)
        if is_main_process() and wandb is not None:
            try:
                if getattr(wandb, "run", None) is not None:
                    wandb.log(
                        {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    wandb.finish(exit_code=1)
            except Exception as wandb_error:
                # Don't let wandb errors hide the original error
                print(
                    f"Warning: Failed to log to wandb: {wandb_error}",
                    file=sys.stderr,
                    flush=True,
                )

        # Always re-raise
        raise


def catch_errors_decorator():
    """Decorator version of catch_errors.

    Usage:
        @catch_errors_decorator()
        def train():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with catch_errors():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def with_hf_retry_ratelimit(func, *args, delay=10, max_attempts=100, **kwargs):
    """Calls the given function with retry logic for HTTP 429 (Too Many Requests) errors.

    This function attempts to call ``func(*args, **kwargs)``. If a rate-limiting error (HTTP 429)
    is encountered—detected via exception type, status code, or error message—it will wait
    for the duration specified by the HTTP ``Retry-After`` header (if present), or fall back to
    the ``delay`` parameter, and then retry. Retries continue up to ``max_attempts`` times.
    Non-429 errors are immediately re-raised. If all attempts fail due to 429, the last
    exception is raised.

    Exceptions handled:
        - huggingface_hub.utils.HfHubHTTPError
        - requests.exceptions.HTTPError
        - OSError

    429 detection is performed by checking the exception's ``response.status_code`` (if available)
    or by searching for '429' or 'Too Many Requests' in the exception message.

    Args:
        func (callable): The function to call.
        *args: Positional arguments to pass to ``func``.
        delay (int, optional): Default wait time (in seconds) between retries if ``Retry-After``
            is not provided. Defaults to 10.
        max_attempts (int, optional): Maximum number of attempts before giving up. Defaults to 100.
        **kwargs: Keyword arguments to pass to ``func``.

    Returns:
        The return value of ``func(*args, **kwargs)`` if successful.

    Raises:
        Exception: The original exception if a non-429 error occurs, or if all attempts fail.

    Example:
        >>> from transformers import AutoModel
        >>> model = with_hf_retry_ratelimit(
        ...     AutoModel.from_pretrained,
        ...     "facebook/ijepa_vith14_1k",
        ...     delay=10,
        ...     max_attempts=5,
        ... )
    """
    attempts = 0
    while True:
        try:
            return func(*args, **kwargs)
        except (HfHubHTTPError, requests.exceptions.HTTPError, OSError) as e:
            # Try to extract status code and Retry-After
            status_code = None
            retry_after = delay
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", None)
                retry_after = int(e.response.headers.get("Retry-After", delay))
            # Fallback: parse error message for 429
            if status_code == 429 or "429" in str(e) or "Too Many Requests" in str(e):
                attempts += 1
                if attempts >= max_attempts:
                    raise
                logging.warning(
                    f"429 received. Waiting {retry_after}s before retrying (attempt {attempts}/{max_attempts})..."
                )
                time.sleep(retry_after)
            else:
                raise


def catch_errors_class(exclude_methods=None):
    """Class decorator that wraps all methods with catch_errors.

    Usage:
        @catch_errors_class()
        class Manager(submitit.helpers.Checkpointable):
            def train(self):
                ...

    Args:
        exclude_methods: List of method names to exclude from wrapping
                        (default: excludes __init__, __new__, __del__, checkpoint)
    """
    if exclude_methods is None:
        # Default exclusions - dunder methods and checkpoint (for Submitit)
        exclude_methods = {
            "__init__",
            "__new__",
            "__del__",
            "__repr__",
            "__str__",
            "checkpoint",
            "__call__",
        }

    def decorator(cls):
        # Iterate over all attributes
        for attr_name in dir(cls):
            # Skip excluded methods
            if attr_name in exclude_methods:
                continue

            # Skip private methods (starting with _)
            if attr_name.startswith("_"):
                continue

            attr = getattr(cls, attr_name)

            # Only wrap callable methods
            if callable(attr):
                # Wrap the method
                wrapped = catch_errors_decorator()(attr)
                setattr(cls, attr_name, wrapped)

        return cls

    return decorator
