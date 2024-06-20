"""Ceq to TOPPE converter."""

__all__ = ["ceq2ge"]

import copy
import math

from types import SimpleNamespace

import numpy as np

import pypulseq as pp


def ceq2ge():
    # Determine b1 scaling file by searching higher rf peak amongst parent blocks

    # Determine readout.mod as the last parent block containing ADC (to mimick ceq2ge.m)

    # Loop over parent blocks and convert blocks to modules
    # {"rf": None, "gx": None, "gy: None, "gz": None, "nChop": None}
    # names = module{n}.mod

    pass


# %% local utils
def _find_b1_scaling(ceq):
    peakb1 = np.zeros(ceq.n_parent_blocks)
    for n in range(ceq.n_parent_blocks):
        block = ceq.parent_blocks[n]
        if block is not None and block.rf is not None:
            peakb1[n] = block.rf.amplitude
    b1scaling = np.argmax(peakb1)
    return b1scaling


def _find_readout(ceq):
    readout = 0
    for n in range(ceq.n_parent_blocks):
        block = ceq.parent_blocks[n]
        if block is not None and block.adc is not None:
            readout = n
    return readout


def _block2mod(p, block, sys):
    if block.rf is not None:
        assert block.adc is None, "A block can either have RF event or ADC, not both"
    if block.adc is not None:
        assert block.rf is None, "A block can either have RF event or ADC, not both"

    if block.rf is not None:
        rf, npre = _rf2ge(p, block.rf, sys)


def _rf2ge(p, block, sys):
    # check timing
    if block.rf.delay + np.finfo(float).eps < sys.rfDeadTime * 1e-6:
        raise ValueError(f"Parent block {p}: RF delay must be >= sysGE.rfDeadTime")
    if (
        block.rf.delay + block.rf.shape_dur + sys.rfRingdownTime * 1e-6
        > block.blockDuration + np.finfo(float).eps
    ):
        raise ValueError(f"Parent block {p}: RF ringdown extends past end of block")

    # calculate time grid of rf pulse
    tge = np.arange(
        0.5 * sys.raster * 1e-6,
        block.rf.shape_dur + 0.5 * sys.raster * 1e-6,
        sys.raster * 1e-6,
    )

    # if time grid is different from native, interpolate
    if np.allclose(tge, block.rf.tt):
        rf = block.rf.amplitude * block.rf.signal / sys.gamma * 1e4  # Gauss
    else:
        rf = (
            np.interp(
                tge, block.rf.tt, block.rf.amplitude * block.rf.signal, left=0, right=0
            )
            / sys.gamma
            * 1e4
        )  # Gauss

    # append dummy points for delay
    npre = round(block.rf.delay / (sys.raster * 1e-6))
    rfres = len(rf)
    rf = np.concatenate([np.zeros(npre), rf])
    return rf, rfres


def _grad2ge():
    pass


def _adc2ge(p, block, sys):
    if block.adc.delay + np.finfo(float).eps < sys.adcDeadTime * 1e-6:
        print(f"Warning: Parent block {p}: ADC delay is < sys.adcDeadTime")
    if (
        block.adc.delay + block.adc.numSamples * block.adc.dwell
        > block.blockDuration + np.finfo(float).eps
    ):
        raise ValueError(f"Parent block {p}: ADC window extends past end of block")
    npre = round(block.adc.delay / (sys.raster * 1e-6))
    rfres = round(block.adc.numSamples * block.adc.dwell / (sys.raster * 1e-6))
    return npre, rfres
