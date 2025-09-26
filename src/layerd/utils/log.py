import logging
import os
import sys
import warnings

from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler compatible with tqdm progress bars.

    This handler uses tqdm.write() to emit log messages, preventing
    interference with active progress bars.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using tqdm.write().

        Args:
            record: The log record to emit.
        """
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str | int = "INFO",
    use_tqdm_handler: bool = False,
    format: str = "%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Setup logging configuration for the application.

    This function should be called once at the application's entry point.

    Args:
        level: Logging level (can be string like 'INFO' or int like logging.INFO).
               Can also be set via LOG_LEVEL environment variable.
        use_tqdm_handler: Whether to use TqdmLoggingHandler for tqdm compatibility.
        format: Log message format.
        datefmt: Date format for timestamps.
    """
    # Check environment variable if not explicitly set
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        level = env_level

    # Convert string level to logging constant
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        level = numeric_level

    # Configure root logger
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    root_logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(fmt=format, datefmt=datefmt)

    # Create and configure handler
    if use_tqdm_handler:
        handler = TqdmLoggingHandler(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Prevent propagation to avoid duplicate messages
    root_logger.propagate = False
