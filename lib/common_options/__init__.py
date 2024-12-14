import click

DEFAULT_LIBRARY_PATH = "/Volumes/T9/Pictures/Photos Library.photoslibrary"
DEFAULT_CONFIDENCE_THRESHOLD = 0.8


def verbose_mode(func):
    return click.option(
        "--verbose",
        "-v",
        "verbose_mode",
        is_flag=True,
        help="Print verbose output.",
    )(func)

def dry_run(func):
    return click.option(
        "--dry-run",
        "-n",
        is_flag=True,
        help="Dry run mode: don't actually update keywords in Photos library.",
    )(func)

def library_path(func):
    return click.option(
        "--library",
        "-l",
        "library_path",
        default=DEFAULT_LIBRARY_PATH,
        help="Path to the library to process.",
    )(func)

def common_options(func):
    """
    A decorator to add common options to a Click command.
    """
    verbose_mode(func)
    dry_run(func)
    func = click.option(
        "--reset",
        "-R",
        is_flag=True,
        help="Reset the database of previously processed photos.",
    )(func)
    library_path(func)
    func = click.option(
        "--selected",
        "-s",
        is_flag=True,
        help="Only process selected photos.",
    )(func)
    func = click.option(
        "--confidence_threshold",
        "-C",
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for models.",
    )(func)
    return func
