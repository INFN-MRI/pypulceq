"""Pulseq to TOPPE converter."""

__all__ = ["seq2ge"]

from types import SimpleNamespace
from typing import Union

import pypulseq as pp

from . import _toppe

from ._seq2ceq import seq2ceq
from ._ceq2ge import ceq2ge


def seq2ge(sequence_name : str, seqarg : Union[pp.Sequence, str], sys : _toppe.SystemSpecs = None, verbose : bool = False, **kwargs):
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a set of files that 
    can be executed on GE scanners using the TOPPE interpreter (v6).

    Parameters
    ----------
    sequence_name : str
        Sequence name (i.e., output ``.tar`` file name).
    seqarg : pypulseq.Sequence | str
        A seq object, or name of a .seq file
    sys : _toppe.SystemSpecs, optional
        TOPPE SystemSpecs object with GE scanner specifications. 
        The default is ``None`` (infer from ``pypulseq.Opts`` object used to create ``seqarg``).
    verbose : bool, optional
        Display info. The default is ``False`` (silent mode).
        
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
    sequence_path : str, optional
        Sequence files on the scanner. The default is
        ``"/usr/g/research/pulseq/v6/seq2ge/{sequence_name}"``.

    """
    # Parse keyworded arguments
    n_max = kwargs.get("n_max", None)
    ignore_segments = kwargs.get("ignore_segments", False)
    ignore_trigger = kwargs.get("ignore_trigger", False)
    sequence_path = kwargs.get("sequence_path", None)
    
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
            maxGrad=seq.system.maxGrad / seq.system.gamma * 100,
            maxGrad=seq.system.maxSlew / seq.system.gamma / 10,
            rfDeadTime=seq.system.rf_dead_time * 1e6,
            rfRingdownTime=seq.system.rf_ringdown_time * 1e6,
            adcDeadTime=seq.system.adc_dead_time * 1e6,
            )

    # Convert Pulseq sequence to PulCeq structure
    ceq = seq2ceq(seqarg, n_max, ignore_segments, verbose)
    
    # Write to TOPPE files
    ceq2ge(sequence_name, ceq, sys, ignore_trigger, sequence_path, verbose)