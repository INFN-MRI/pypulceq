"""Command Line Interface for seq2ge."""

import click

from ._seq2ge import seq2ge

@click.command()
@click.option("--output-name", required=True, help="Name of output .tar sequence file")
@click.option("--sequence-file", required=True, help="Path on disk to input Pulseq file (.seq) to be converted")
@click.option("--sequence-path", default="/usr/g/research/pulseq/v6/seq2ge/", show_default=True, help="location of sequence files on scanner")
@click.option("--ignore-trigger", default=False, show_default=True, help="ignore TTL pulses in sequence")
@click.option("--ignore-segments", default=False, show_default=True, help="assign each parent block to individual segment")
def cli(output_name, sequence_file, sequence_path, ignore_trigger, ignore_segments):
    """Main seq2ge CLI."""
    seq2ge(output_name, sequence_file, sequence_path=sequence_path, ignore_trigger=ignore_trigger, ignore_segments=ignore_segments)