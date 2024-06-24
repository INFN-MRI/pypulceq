"""Command Line Interface for seq2ge."""

import click

from ._seq2ge import seq2ge


@click.command()
@click.option("--output-name", required=True, help="Name of output .tar sequence file")
@click.option(
    "--input-file",
    required=True,
    help="Path on disk to input Pulseq file (.seq) to be converted",
)
@click.option(
    "--sequence-path",
    default="/usr/g/research/pulseq/v6/seq2ge/",
    show_default=True,
    help="location of sequence files on scanner",
)
@click.option(
    "--nviews",
    default=600,
    show_default=True,
    help="number of views, i.e., frequency encodings for a single k-space volume (e.g., phase encoding lines).",
)
@click.option(
    "--nslices",
    default=2048,
    show_default=True,
    help="number of slices / slab-encodings.",
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
    help="ignore TTL pulses in sequence",
)
@click.option(
    "--ignore-segments",
    default=False,
    show_default=True,
    help="assign each parent block to individual segment",
)
def cli(
    output_name,
    input_file,
    sequence_path,
    nviews,
    nslices,
    verbose,
    ignore_trigger,
    ignore_segments,
):
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a set of files that
    can be executed on GE scanners using the TOPPE interpreter (v6).
    """
    seq2ge(
        output_name,
        input_file,
        nviews=nviews,
        nslices=nslices,
        verbose=verbose,
        sequence_path=sequence_path,
        ignore_trigger=ignore_trigger,
        ignore_segments=ignore_segments,
    )
