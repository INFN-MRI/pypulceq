"""Main API."""

__all__ = ["seq2ge"]

import pypulseq as _pp

from . import _toppe

from . import _seq2ceq
from . import _ceq2ge


def seq2ge(seqarg, ofname, ignoreSegmentLabels=False, sysGE=None):
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a .tar file for GE scanners.

    This is a porting of https://github.com/HarmonizedMRI/PulCeq/blob/main/matlab/seq2ge.m

    Parameters
    ----------
    seqarg : str or Sequence
        A seq object, or the name of a .seq file.
    ofname : str
        Output file name.
    sysGE : SystemSpecs, optional
        Sys struct representing GE scanner parameters. 
        Provide ``None`` to trigger default behavior.
        The default is ``None``.

    Effects
    -------
    Writes a .tar file containing all the required scan files to run on GE scanners.

    """
    # Get Pulseq system struct
    if isinstance(seqarg, str):
        print("Getting Pulseq system struct from .seq file...")
        seq = _pp.Sequence()
        seq.read(seqarg)
        print(" done\n")
    else:
        seq = seqarg
    
    # Read in system specs defined from .seq file
    sys = seq.system

    # Default behavior for sysGE
    if not sysGE:

        # Define sysGE
        sysGE = _toppe.SystemSpecs(
            psd_rf_wait=200,  # us
            psd_grd_wait=200,  # us
            maxGrad=sys.max_grad / sys.gamma * 100,  # G/cm
            maxSlew=1.01
            * sys.max_slew
            / sys.gamma
            / 10,  # G/cm/ms. Factor > 1 is fudge factor to avoid exceeding limit after interpolating to GE raster time.
            rfDeadTime=sys.rf_dead_time * 1e6,  # us
            rfRingdownTime=sys.rf_ringdown_time * 1e6,  # us
            adcDeadTime=sys.adc_dead_time * 1e6,  # us
        )

    # Convert the .seq file/object to the PulCeq representation
    ceq = _seq2ceq.seq2ceq(seqarg, ignoreSegmentLabels=ignoreSegmentLabels)

    # Write to TOPPE files
    _ceq2ge.ceq2ge(ceq, sysGE, ofname, "seqGradRasterTime", sys.grad_raster_time)
