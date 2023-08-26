import click
import yaml

import sys

from .v2 import v2


@click.group()
def main(args=None):
    """Console script for care."""
    return None

main.add_command(v2)

if __name__ == "__main__":
    sys.exit(main())
