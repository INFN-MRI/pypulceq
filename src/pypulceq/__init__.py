"""Main API."""

# read version from installed package
from importlib.metadata import version

__version__ = version("pypulceq")

__all__ = ["seq2ge"]

import pypulseq as _pp

from . import _toppe

from . import _seq2ceq
from . import _ceq2ge


def seq2ge(seqarg, sysGE, ofname):
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a .tar file for GE scanners.

    This is a porting of https://github.com/HarmonizedMRI/PulCeq/blob/main/matlab/seq2ge.m

    Parameters
    ----------
    seqarg : str or Sequence
        A seq object, or the name of a .seq file.
    sysGE : dict
        Sys struct representing GE scanner parameters. Provide an empty dictionary {} to trigger default behavior.
    ofname : str
        Output file name.

    Effects
    -------
    Writes a .tar file containing all the required scan files to run on GE scanners.

    """
    # Get Pulseq system struct
    print("Getting Pulseq system struct from .seq file...")
    seq = _pp.Sequence()
    seq.read(seqarg)
    sys = seq.sys
    print(" done\n")

    # Default behavior for sysGE
    if not sysGE:
        # Read in system specs defined from .seq file
        seq = _pp.Sequence()
        seq.read(seqarg)
        sys = seq.sys

        # Define sysGE
        sysGE = _toppe.SystemSpecs(
            psd_rf_wait=200,  # us
            psd_grd_wait=200,  # us
            maxGrad=sys.maxGrad / sys.gamma * 100,  # G/cm
            maxSlew=1.01
            * sys.maxSlew
            / sys.gamma
            / 10,  # G/cm/ms. Factor > 1 is fudge factor to avoid exceeding limit after interpolating to GE raster time.
            rfDeadTime=sys.rfDeadTime * 1e6,  # us
            rfRingdownTime=sys.rfRingdownTime * 1e6,  # us
            adcDeadTime=sys.adcDeadTime * 1e6,  # us
        )

    # Convert the .seq file/object to the PulCeq representation
    ceq = _seq2ceq.seq2ceq(seqarg)

    # Write to TOPPE files
    _ceq2ge.ceq2ge(ceq, sysGE, ofname, "seqGradRasterTime", sys.gradRasterTime)
