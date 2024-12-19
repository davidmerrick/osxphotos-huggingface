import click

DEFAULT_CONFIG_PATH = "~/.config/harmonia/config.yml"
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


def env(func):
    return click.option(
        "--env",
        "-E",
        default="dev",
        help="Environment to use for keystore prefix",
    )(func)

def reset(func):
    return click.option(
        "--reset",
        "-R",
        is_flag=True,
        help="Reset the database of previously processed photos.",
    )(func)

def selected(func):
    return click.option(
        "--selected",
        "-s",
        is_flag=True,
        help="Only process selected photos.",
    )(func)

def confidence(func):
    return click.option(
        "--confidence_threshold",
        "-C",
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for models.",
    )(func)

def config_path(func):
    return click.option(
        "--config_path",
        "-y", # for yaml
        default=DEFAULT_CONFIG_PATH,
        help="Confidence threshold for models.",
    )(func)

def common_options(func):
    """
    A decorator to add common options to a Click command.
    """
    verbose_mode(func)
    dry_run(func)
    reset(func)
    library_path(func)
    selected(func)
    confidence(func)
    return func
