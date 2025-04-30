import logging


def get_logger(name: str) -> logging.Logger:
    """
    Logger creation.
    """
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    return logging.getLogger(name)
