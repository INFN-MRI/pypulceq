"""PulCeq to buffer converter."""

__all__ = ["ceq2buffer"]

from types import SimpleNamespace

import numpy as np

from ._ceq import Ceq

from . import _interp


def ceq2buffer(
    ceqstruct: SimpleNamespace,
    sys: SimpleNamespace,
    ignore_trigger: bool = False,
    verbose: bool = False,
):
    """
    Convert a PulCeq struct to a binary buffer that can be executed
    on GE scanners using the TOPPE interpreter (v7).

    Parameters
    ----------
    ceqstruct : SimpleNamespace
        PulCeq struct, based on github/HarmonizedMRI/PulCeq/src/pulCeq.h
    sys : _toppe.SystemSpecs
        TOPPE SystemSpecs object with GE scanner specifications.
    ignore_trigger : bool, optional
        If True, ignore all TTL pulses. The default is ``False``.
    verbose : bool, optional
        Display info. The default is ``False`` (silent mode).

    """
    # Determine B1 scaling file by searching higher rf peak amongst parent blocks
    max_b1 = _find_b1_max(ceqstruct)
    ceqstruct.max_b1 = max_b1

    # Loop over parent blocks and interpolate blocks
    if verbose:
        print("Interpolating Pulseq blocks to GE specs...", end="\t")
    pblocks = []  # dummy wait module
    for p in range(1, ceqstruct.n_parent_blocks):
        pblocks.append(
            _block_interp(
                p, ceqstruct.parent_blocks[p], sys, ceqstruct.sys.grad_raster_time
            )
        )
    ceqstruct.parent_blocks = pblocks
    if verbose:
        print("done!\n")

    # Build core lut
    segments = []
    for n in range(ceqstruct.n_segments):
        tmp = {}
        tmp["segment_id"] = n
        tmp["n_blocks_in_segment"] = len(ceqstruct.blocks_in_segment[n])
        tmp["block_ids"] = ceqstruct.blocks_in_segment[n]
        segments.append(tmp)
    ceqstruct.segments = segments

    # fix Loop
    ceqstruct.loop.modname = np.arange(ceqstruct.n_parent_blocks)[
        ceqstruct.parent_blocks_idx
    ]
    ceqstruct.loop.core = ceqstruct.segments_idx
    ceqstruct.loop.Gamplitude = np.ascontiguousarray(ceqstruct.loop.Gamplitude.T)
    ceqstruct.loop.rotmat = ceqstruct.loop.rotmat.reshape(-1, 9)
    ceqstruct.loop.rotmat = np.ascontiguousarray(ceqstruct.loop.rotmat.T)

    # put loop together
    loop = [
        ceqstruct.loop.core,
        ceqstruct.loop.modname,
        ceqstruct.loop.RFamplitude,
        ceqstruct.loop.RFphase,
        ceqstruct.loop.RFoffset,
        ceqstruct.loop.Gamplitude[0],
        ceqstruct.loop.Gamplitude[1],
        ceqstruct.loop.Gamplitude[2],
        ceqstruct.loop.DAQphase,
        ceqstruct.loop.textra * 1e3,
        *ceqstruct.loop.rotmat,
    ]

    loop = np.stack(loop, axis=-1)
    loop = np.ascontiguousarray(loop)
    ceqstruct.loop = loop.astype(np.float32)

    # get non-delay durations
    block_durations = [b["block_duration"] for b in ceqstruct.parent_blocks]
    block_durations = np.asarray([0] + block_durations)
    tot_block_durations = block_durations[ceqstruct.loop[:, 1].astype(int)].sum()

    # get delay durations
    delay_durations = ceqstruct.loop[:, 9] * 1e-3
    tot_delay_durations = delay_durations[ceqstruct.loop[:, 1] == 0].sum()

    # total segment ringdown
    n_seg_boundaries = (np.diff(ceqstruct.loop[:, 0]) != 0).sum()
    tot_seg_ringdown = sys.segmentRingdownTime * 1e-6 * n_seg_boundaries
    ceqstruct.duration = tot_block_durations + tot_seg_ringdown + tot_delay_durations

    # Build Ceq structure
    ceq = Ceq.from_struct(ceqstruct)

    return ceq


# %% local utils
def _find_b1_max(ceqstruct):
    peakb1 = np.zeros(ceqstruct.n_parent_blocks)
    for n in range(ceqstruct.n_parent_blocks):
        block = ceqstruct.parent_blocks[n]
        if block is not None and block.rf is not None:
            peakb1[n] = block.rf.amplitude
    return np.max(peakb1)


def _block_interp(p, block, sys, dt):
    if block.rf is not None:
        assert block.adc is None, "A block can either have RF event or ADC, not both"
    if block.adc is not None:
        assert block.rf is None, "A block can either have RF event or ADC, not both"

    # compute duration
    block_duration = (
        np.ceil(block.block_duration / (sys.raster * 1e-6)) * sys.raster * 1e-6
    )

    # initialize output module
    pblock = {"ID": int(p), "block_duration": block_duration}

    if block.rf is not None:
        block = _interp.rf2ge(p, block, sys)
        pblock["rf"] = block.rf
    if block.gx is not None:
        gx = _interp.grad2ge(p, block.gx, sys, dt)
        pblock["gx"] = gx
    if block.gy is not None:
        gy = _interp.grad2ge(p, block.gy, sys, dt)
        pblock["gy"] = gy
    if block.gz is not None:
        gz = _interp.grad2ge(p, block.gz, sys, dt)
        pblock["gz"] = gz
    if block.adc is not None:
        adc = _interp.adc2ge(p, block, sys)
        pblock["adc"] = adc
    if hasattr(block, "trig"):
        pblock["trig"] = _interp.trig2ge(p, block, sys)

    return pblock
