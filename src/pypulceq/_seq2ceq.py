"""Pulseq to PulCeq converter."""

__all__ = ["seq2ceq"]

import copy
import math

from typing import Union
from types import SimpleNamespace

import numpy as np

import pypulseq as pp

from . import _autosegment


def seq2ceq(
    seq: pp.Sequence,
    n_max: int = None,
    ignore_segments: bool = False,
    verbose: bool = False,
) -> SimpleNamespace:
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a PulCeq struct.

    See github/HarmonizedMRI/PulCeq/src/pulCeq.h and
    https://github.com/HarmonizedMRI/PulCeq/blob/main/matlab/seq2ceq.m

    Parameters
    ----------
    seqarg : pypulseq.Sequence | str
        A seq object, or name of a .seq file
    n_max : int, optional
        Maximum number of sequence events to be read. The default is ``None``
        (read all blocks).
    ignore_segments : bool, optional
        Ignore segment labels and assign each parent block to an individual segment.
        The default is ``False`` (use segment label or attempt to determine automatically).
    verbose : bool
        Display info. The default is ``False`` (silent mode).

    Returns
    -------
    ceq  : SimpleNamespace
        PulCeq struct, based on github/HarmonizedMRI/PulCeq/src/pulCeq.h

    """
    # Get blocks
    if verbose:
        if n_max is None:
            print("Selecting all blocks...", end="\t")
        else:
            print(
                f"Selecting first {n_max} blocks out of {len(seq.block_events)}...",
                end="\t",
            )
    blocks = np.stack(list(seq.block_events.values()), axis=1)
    if verbose:
        print("done!\n")

    # Restrict to first n_max
    if n_max is not None:
        blocks = blocks[:, :n_max]

    # Loop over [BLOCKS], extract scan loop
    if verbose:
        print("Getting scan dynamics...", end="\t")
    dur, loop, trid, hasrot = _get_dynamics(seq, n_max)
    if verbose:
        print("done!\n")

    if verbose:
        print("Getting parent blocks...", end="\t")
    # Loop over [RF], extract (amp, mag_id, shape_id) -> row 0, 1, 2 (0-index based)
    rf = np.stack(list(seq.rf_library.data.values()), axis=-1)
    rf_events = rf[:3].T
    rf_events = np.pad(rf_events, ((1, 0), (0, 0)))

    # Split [GRADIENTS] and [TRAP]
    gradients_and_traps = np.array(list(seq.grad_library.data.values()), dtype=object)
    istrap = np.asarray([len(g) == 5 for g in gradients_and_traps])
    isgradient = np.logical_not(istrap)

    # Actual split
    if isgradient.any():
        gradients = np.stack(list(gradients_and_traps[isgradient]), axis=-1)

        # Loop over [GRADIENTS], extract (amp, shape_id) -> row 0-1 (0-index based)
        gradient_events = gradients[:2]
        gradient_events = np.pad(gradient_events, ((0, 0), (0, 3)))
    else:
        gradient_events = None

    if istrap.any():
        traps = np.stack(list(gradients_and_traps[istrap]), axis=-1)

        # Loop over [TRAP], extract (amp, rise, flat, fall, delay) _> all row
        trap_events = traps.T
    else:
        trap_events = None

    # Put back together
    if gradient_events is not None and trap_events is not None:
        gradient_and_traps_events = np.zeros((len(gradients_and_traps), 5), dtype=float)
        gradient_and_traps_events[isgradient] = gradient_and_traps_events
        gradient_and_traps_events[istrap] = trap_events
    elif gradient_events is not None:
        gradient_and_traps_events = gradient_events.astype(float)
    elif trap_events is not None:
        gradient_and_traps_events = trap_events.astype(float)

    # Pad to account for 1-based index
    gradient_and_traps_events = np.pad(gradient_and_traps_events, ((1, 0), (0, 0)))

    # Loop over [ADC], extract (num, dwell, delay) -> -> row 1-3 (0-index based)
    adc = np.stack(list(seq.adc_library.data.values()), axis=-1)
    adc_events = adc[:3].T
    adc_events = np.pad(adc_events, ((1, 0), (0, 0)))

    # # Loop over [BLOCKS], extract trigger flag (0: off; 1: on)
    hastrig = loop.trigout[:, None]

    # Get rf, gx, gy, gz and adc indexes
    rf_idx = blocks[1]
    gx_idx = blocks[2]
    gy_idx = blocks[3]
    gz_idx = blocks[4]
    adc_idx = blocks[5]

    # Build block identifiers matrix
    rf_uid = rf_events[rf_idx, 1:]  # (nblocks, 3); row = [mag_id, phase_id]
    # (nblocks, 4); row = [shape_id, 0, 0, 0] (grad) or [rise, flat, fall, delay] (trap)
    gx_uid = gradient_and_traps_events[gx_idx, 1:]
    # (nblocks, 4); row = [shape_id, 0, 0, 0] (grad) or [rise, flat, fall, delay] (trap)
    gy_uid = gradient_and_traps_events[gy_idx, 1:]
    # (nblocks, 4); row = [shape_id, 0, 0, 0,] (grad) or [rise, flat, fall, delay] (trap)
    gz_uid = gradient_and_traps_events[gz_idx, 1:]
    adc_uid = adc_events[adc_idx]  # (nblocks, 3); row = [num, dwell, delay]
    block_uid = [dur[:, None], rf_uid, gx_uid, gy_uid, gz_uid, adc_uid, hastrig]
    block_uid = np.concatenate(block_uid, axis=1)

    # Find pure delay blocks
    # exclude duration, adc and trigger columns (isdelay.m only checks for rf, gx, gy and gz waveforms)
    notdelay = block_uid[:, 1:16].any(axis=1)
    isdelay = np.logical_not(notdelay)
    event_uid = block_uid[notdelay]

    # Find unique non-delay events in parent blocks
    parent_blocks_ids = np.unique(event_uid, return_index=True, axis=0)[1]
    parent_blocks_ids = sorted(parent_blocks_ids)
    parent_block_uid = block_uid[parent_blocks_ids]

    # generate parent block list and module index for each row in loop
    parent_blocks, parent_blocks_idx = _get_parent_blocks(
        seq, block_uid, parent_block_uid, parent_blocks_ids
    )
    if verbose:
        print(f"done! Found {len(parent_blocks)} parent blocks.\n")

    # Build block amplitudes matrix
    if verbose:
        print("Setting parent blocks amplitudes to max...", end="\t")
    rf_amp = rf_events[rf_idx, 0]
    gx_amp = gradient_and_traps_events[gx_idx, 0]
    gy_amp = gradient_and_traps_events[gy_idx, 0]
    gz_amp = gradient_and_traps_events[gz_idx, 0]
    amplitudes = np.stack((rf_amp, gx_amp, gy_amp, gz_amp), axis=1)

    # Set parent blocks to maximum amplitude
    parent_blocks, amplitudes = _set_max_amplitudes(
        amplitudes, parent_blocks, parent_blocks_idx
    )

    # Compute delay
    textra = loop.textra
    textra[isdelay] = np.round(dur[isdelay] * 1e6) / 1e3

    # Fill scan dynamics with missing info
    loop.RFamplitude = amplitudes[:, 0]
    loop.Gamplitude = amplitudes[:, 1:]
    loop.textra = textra
    if verbose:
        print("done!\n")

    # Build Ceq structure
    ceq = SimpleNamespace(n_max=blocks.shape[1])

    # Set parent blocks
    ceq.n_parent_blocks = len(parent_blocks)
    ceq.parent_blocks = parent_blocks
    ceq.parent_blocks_idx = parent_blocks_idx
    ceq.loop = loop

    # Determine segments
    segments_idx = np.zeros(ceq.n_max)
    if ignore_segments:
        if verbose:
            print(
                "Ignoring segment labels; Put each parent block in a separate segment...",
                end="\t",
            )
        segments_idx = parent_blocks_idx
        blocks_in_segment = [n for n in range(len(parent_blocks))]
        if verbose:
            print("done!\n")
    elif (trid != -1).any():
        if verbose:
            print(
                "Found segment labels; building segment definitions and segment ids...",
                end="\t",
            )
        trid_idx = np.where(trid != -1)[0]
        trid_val = trid[trid_idx]
        for n in range(len(trid_idx) - 1):
            segments_idx[trid_idx[n] : trid_idx[n + 1]] = trid_val[n]
        segments_idx[trid_idx[-1] :] = trid_val[-1]
        if verbose:
            print("done!\n")

        # Create segment definition (already squashed and re-indexed in appearence order)
        blocks_in_segment = []
        signed_parent_blocks_idx = parent_blocks_idx * hasrot
        for n in trid_val:
            tmp_val, tmp_idx = np.unique(
                signed_parent_blocks_idx[segments_idx == n], return_index=True
            )
            tmp_val = tmp_val[np.argsort(tmp_idx)]
            blocks_in_segment.append(tmp_val)
        blocks_in_segment = _autosegment.split_rotated_segments(blocks_in_segment)

        # Rename segment_idx
        tmp = -np.ones_like(segments_idx)
        for n in range(len(trid_val)):
            tmp[segments_idx == trid_val[n]] = n
        segments_idx = tmp + 1
    else:  # attempt automatic search
        if verbose:
            print(
                "Segment labels not found; attempt to determine it automatically...",
                end="\t",
            )
        blocks_in_segment = _autosegment.find_segment_definitions(parent_blocks_idx * hasrot) # rotated events will have positive sign
        blocks_in_segment = _autosegment.split_rotated_segments(blocks_in_segment)
        for n in range(len(blocks_in_segment)):
            tmp = _autosegment.find_segments(parent_blocks_idx, blocks_in_segment[n])
            segments_idx[tmp] = n
        segments_idx += 1
        if verbose:
            print(f"done! Found {len(blocks_in_segment)} segments.\n")

    # Assign
    ceq.n_segments = len(blocks_in_segment)
    ceq.segments_idx = segments_idx.astype(int)
    ceq.blocks_in_segment = blocks_in_segment
    ceq.sys = seq.system

    return ceq


# %% local utils
def _get_dynamics(seq, nevents):
    # get number of events
    if nevents is None:
        nevents = len(seq.block_events)

    # initialize quantities (here struct of array instead of array of struct)
    RFoffset = np.zeros(nevents)
    RFphase = np.zeros(nevents)
    DAQphase = np.zeros(nevents)
    trigout = np.zeros(nevents)
    rotmat = np.eye(3, 3)
    rotmat = np.repeat(rotmat[None, ...], nevents, axis=0)
    textra = np.zeros(nevents)
    adc_idx = np.zeros(nevents)

    # non-loop variables (duration and TRID labels)
    dur = np.zeros(nevents)
    trid = -np.ones(nevents)
    hasrot = -np.ones(nevents)

    # init variable
    adc_count = 0

    # loop over evens
    for n in range(nevents):
        b = seq.get_block(n + 1)
        dur[n] = b.block_duration
        if b.rf is not None:
            RFoffset[n] = b.rf.freq_offset
            RFphase[n] = b.rf.phase_offset
        if b.adc is not None:
            DAQphase[n] = b.adc.phase_offset
            adc_count += 1
        if hasattr(b, "trig"):
            trigout[n] = 1.0
        if hasattr(b, "rotation"):
            hasrot[n] = 1.0
            rotmat[n] = b.rotation.rot_matrix
        if b.label is not None and b.label.label == "TRID":
            trid[n] = b.label.value
        adc_idx[n] = adc_count

    # prepare loop dict
    loop = SimpleNamespace()
    loop.modname = None
    loop.Gamplitude = None
    loop.RFoffset = RFoffset
    loop.RFphase = RFphase
    loop.DAQphase = DAQphase
    loop.trigout = trigout
    loop.textra = textra
    loop.rotmat = rotmat
    loop.adc_idx = adc_idx.astype(int)
    loop.view = None
    loop.slice = None
    loop.echo = None
    loop.core = None

    return dur, loop, trid, hasrot


def _get_parent_blocks(seq, block_uid, parent_block_uid, parent_blocks_ids):
    # Number of unique blocks
    n_parent_blocks = parent_block_uid.shape[0]
    parent_blocks_idx = -1 * np.ones(block_uid.shape[0], dtype=int)

    # Find indexes
    for n in range(n_parent_blocks):
        # Compare each row in the matrix with the given row
        tmp = np.all(block_uid == parent_block_uid[n], axis=1)

        # Find the index of the first matching row
        parent_blocks_idx[tmp] = n

    # Convert to array
    parent_blocks_idx = np.asarray(parent_blocks_idx)
    parent_blocks_idx += 1

    # Get blocks
    parent_blocks = []

    # Get shapes library
    shape_lib = list(seq.shape_library.data.values())

    for n in range(n_parent_blocks):
        idx = parent_blocks_ids[n] + 1
        parent_block = copy.deepcopy(seq.get_block(idx))
        if parent_block.rf is not None:
            mag_idx, phase_idx = int(block_uid[idx][1]), int(block_uid[idx][2])
            mag_shape = _decompress(shape_lib[mag_idx])
            phase_shape = _decompress(shape_lib[phase_idx])

            parent_block.rf.signal = mag_shape * np.exp(1j * 2 * math.pi * phase_shape)

        if parent_block.gx is not None and parent_block.gx.type == "grad":
            shape_idx = int(block_uid[idx][3])
            grad_shape = _decompress(shape_lib[shape_idx])

            parent_block.gx.waveform = grad_shape

        if parent_block.gy is not None and parent_block.gy.type == "grad":
            shape_idx = int(block_uid[idx][7])
            grad_shape = _decompress(shape_lib[shape_idx])

            parent_block.gy.waveform = grad_shape

        if parent_block.gz is not None and parent_block.gz.type == "grad":
            shape_idx = int(block_uid[idx][11])
            grad_shape = _decompress(shape_lib[shape_idx])

            parent_block.gz.waveform = grad_shape

        parent_blocks.append(parent_block)

    # return parent_blocks (add empty delay block at idx=0)
    return [None] + parent_blocks, parent_blocks_idx


def _decompress(shape_data):
    compressed = SimpleNamespace()
    compressed.num_samples = shape_data[0]
    compressed.data = shape_data[1:]

    return pp.decompress_shape.decompress_shape(compressed)


def _set_max_amplitudes(amplitudes, parent_blocks, parent_blocks_idx):
    n_parent_blocks = len(parent_blocks)
    for n in range(1, n_parent_blocks):
        idx = parent_blocks_idx == n
        tmp = abs(amplitudes[idx])
        amp = tmp.max(axis=0)

        # now set to maximum
        if parent_blocks[n].rf is not None:
            amplitudes[idx, 0] /= amp[0]
            parent_blocks[n].rf.amplitude = amp[0]
        if parent_blocks[n].gx is not None:
            amplitudes[idx, 1] /= amp[1]
            if parent_blocks[n].gx.type == "grad":
                parent_blocks[n].gx.waveforms *= amp[1]
            else:
                parent_blocks[n].gx.amplitude = amp[1]
                parent_blocks[n].gx.area = parent_blocks[n].gx.amplitude * (
                    parent_blocks[n].gx.flat_time
                    + parent_blocks[n].gx.rise_time / 2
                    + parent_blocks[n].gx.fall_time / 2
                )
                parent_blocks[n].gx.flat_area = (
                    parent_blocks[n].gx.amplitude * parent_blocks[n].gx.flat_time
                )
        if parent_blocks[n].gy is not None:
            amplitudes[idx, 2] /= amp[2]
            if parent_blocks[n].gy.type == "grad":
                parent_blocks[n].gy.waveforms *= amp[2]
            else:
                parent_blocks[n].gy.amplitude = amp[2]
                parent_blocks[n].gy.area = parent_blocks[n].gy.amplitude * (
                    parent_blocks[n].gy.flat_time
                    + parent_blocks[n].gy.rise_time / 2
                    + parent_blocks[n].gy.fall_time / 2
                )
                parent_blocks[n].gy.flat_area = (
                    parent_blocks[n].gy.amplitude * parent_blocks[n].gy.flat_time
                )
        if parent_blocks[n].gz is not None:
            amplitudes[idx, 3] /= amp[3]
            if parent_blocks[n].gz.type == "grad":
                parent_blocks[n].gz.waveforms *= amp[3]
            else:
                parent_blocks[n].gz.amplitude = amp[3]
                parent_blocks[n].gz.area = parent_blocks[n].gz.amplitude * (
                    parent_blocks[n].gz.flat_time
                    + parent_blocks[n].gz.rise_time / 2
                    + parent_blocks[n].gz.fall_time / 2
                )
                parent_blocks[n].gz.flat_area = (
                    parent_blocks[n].gz.amplitude * parent_blocks[n].gz.flat_time
                )

    return parent_blocks, amplitudes


