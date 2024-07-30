"""Command Line Interface for seq2ge."""

import click

from ._seq2buffer import seq2buffer
from ._seq2files import seq2files


@click.command()
@click.option(
    "--output-name", required=True, help="Name of output .tar / .dat sequence file"
)
@click.option(
    "--input-file",
    required=True,
    help="Path on disk to input Pulseq file (.seq) to be converted",
)
@click.option(
    "--binary-format",
    default=False,
    show_default=True,
    help="Export sequence in binary format (True) or TOPPE files (False)",
)
@click.option(
    "--sequence-path",
    default="/srv/nfs/psd/usr/psd/pulseq/seq2ge/",
    show_default=True,
    help="Location of sequence files on scanner",
)
@click.option(
    "--nviews",
    default=600,
    show_default=True,
    help="Number of views, i.e., frequency encodings for a single k-space volume (e.g., phase encoding lines).",
)
@click.option(
    "--nslices",
    default=2048,
    show_default=True,
    help="Number of slices / slab-encodings.",
)
@click.option(
    "--verbose",
    default=True,
    show_default=True,
    help="Display information.",
)
@click.option(
    "--ignore-trigger",
    default=False,
    show_default=True,
    help="Ignore TTL pulses in sequence.",
)
@click.option(
    "--ignore-segments",
    default=False,
    show_default=True,
    help="Assign each parent block to individual segment.",
)
def cli(
    output_name,
    input_file,
    binary_format,
    sequence_path,
    nviews,
    nslices,
    verbose,
    ignore_trigger,
    ignore_segments,
):
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a set of files that
    can be executed on GE scanners using the TOPPE interpreter (v6) or use a binary representation (v7).
    """
    if binary_format:
        seq2buffer(
            input_file,
            verbose=verbose,
            sequence_name=output_name,
            ignore_trigger=ignore_trigger,
            ignore_segments=ignore_segments,
        )
    else:
        seq2files(
            output_name,
            input_file,
            nviews=nviews,
            nslices=nslices,
            verbose=verbose,
            sequence_path=sequence_path,
            ignore_trigger=ignore_trigger,
            ignore_segments=ignore_segments,
        )
