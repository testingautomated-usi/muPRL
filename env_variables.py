import os
import sys


def parse_at_prefixed_params(args, param_name):
    try:
        index = args.index("--" + param_name)
        return args[index + 1]
    except ValueError:
        return None


logging_level = parse_at_prefixed_params(args=sys.argv, param_name="logging_level")

# Get the root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

LOGGING_LEVEL = logging_level.upper() if logging_level else "DEBUG"
assert LOGGING_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
