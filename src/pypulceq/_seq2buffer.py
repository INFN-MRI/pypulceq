"""Pulseq to TOPPE converter."""

__all__ = ["seq2buffer"]

from typing import Union

import pypulseq as pp

from . import _toppe

from ._seq2ceq import seq2ceq
from ._ceq2buffer import ceq2buffer


def seq2buffer(
    seqarg: Union[pp.Sequence, str],
    sys: _toppe.SystemSpecs = None,
    verbose: bool = False,
    sequence_name: str = None,
    **kwargs,
):
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a set of files that
    can be executed on GE scanners using the TOPPE interpreter (v6).

    Parameters
    ----------
    seqarg : pypulseq.Sequence | str
        A seq object, or name of a .seq file
    sys : _toppe.SystemSpecs, optional
        TOPPE SystemSpecs object with GE scanner specifications.
        The default is ``None`` (infer from ``pypulseq.Opts`` object used to create ``seqarg``).
    verbose : bool, optional
        Display info. The default is ``False`` (silent mode).
    sequence_name : str, optional
        Sequence name (i.e., output ``.dat`` file name).
        If not provided, do not save to disk (e.g., for streaming).
        The default is ``None``.

    Keyword Arguments
    -----------------
    n_max : int, optional
        Maximum number of sequence events to be read. The default is ``None``
        (read all blocks).
    ignore_segments : bool, optional
        Ignore segment labels and assign each parent block to an individual segment.
        The default is ``False`` (use segment label or attempt to determine automatically).
    ignore_trigger : bool, optional
        If True, ignore all TTL pulses. The default is ``False``.

    """
    # Parse keyworded arguments
    n_max = kwargs.get("n_max", None)
    ignore_segments = kwargs.get("ignore_segments", False)
    ignore_trigger = kwargs.get("ignore_trigger", False)

    # Get sequence
    if isinstance(seqarg, str):
        if verbose:
            print("Reading .seq file...", end="\t")
        seq = pp.Sequence()
        seq.read(seqarg)
        if verbose:
            print("done!\n")
    else:
        if verbose:
            print("Removing duplicates from seq object...", end="\t")
        seq = seqarg.remove_duplicates()
        if verbose:
            print("done!\n")

    # If toppe SystemSpecs is not specified, convert pulseq
    if sys is None:
        sys = _toppe.SystemSpecs(
            maxGrad=seq.system.max_grad / seq.system.gamma * 100,
            maxSlew=seq.system.max_slew / seq.system.gamma / 10,
            rfDeadTime=seq.system.rf_dead_time * 1e6,
            rfRingdownTime=seq.system.rf_ringdown_time * 1e6,
            adcDeadTime=seq.system.adc_dead_time * 1e6,
            B0=seq.system.B0,
            rfUnit="G",
            gradUnit="G/cm",
            slewUnit="G/cm/msec",
        )

    # Convert Pulseq sequence to PulCeq structure
    ceqstruct = seq2ceq(seq, n_max, ignore_segments, verbose)

    # Create CEQ structure
    ceq = ceq2buffer(ceqstruct, sys, ignore_trigger, verbose)
    ceqbuffer = ceq.to_bytes()

    # (Optional): write to disk
    if sequence_name is not None:
        with open(sequence_name + ".dat", "wb") as f:
            f.write(ceqbuffer)
        if verbose:
            print(f"Sequence {sequence_name} ready for execution on GE scanners\n")
    else:
        if verbose:
            print("Sequence ready for execution on GE scanners\n")

    return ceqbuffer
