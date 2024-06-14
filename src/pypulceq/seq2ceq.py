"""Pulseq to ceq converter."""

__all__ = ["seq2ceq"]

from types import SimpleNamespace

import numpy as np

import pypulseq as pp


def seq2ceq(seqarg):
    """
    Convert a Pulseq file (http://pulseq.github.io/) to a PulCeq struct.

    See github/HarmonizedMRI/PulCeq/src/pulCeq.h and
    https://github.com/HarmonizedMRI/PulCeq/blob/main/matlab/seq2ceq.m

    Parameters
    ----------
    seqarg : pypulseq.Sequence | str
        A seq object, or name of a .seq file

    Returns
    -------
    ceq  : SimpleNamespace
        Struct, based on github/HarmonizedMRI/PulCeq/src/pulCeq.h

    """
    # Parse
    if isinstance(seqarg, str):
        seq = pp.Sequence()
        seq.read(seqarg)
    else:
        seq = seqarg.remove_duplicates()

    # Get blocks
    blocks = np.stack(list(seq.block_events.values()), axis=1)

    # Get number of blocks
    n_blocks = blocks.shape[1]

    # Loop over [BLOCKS], extract durations (rounded to 1e-6)
    dur = _get_blocks_durations(n_blocks, seq)[:, None]

    # Loop over [RF], extract (amp, mag_id, shape_id) -> row 1, 2 (0-index based)
    rf = np.stack(list(seq.rf_library.data.values()), axis=-1)
    rf_shapes = rf[:3].T
    rf_shapes = np.pad(rf_shapes, ((1, 0), (0, 0)))

    # Split [GRADIENTS] and [TRAP]
    gradients_and_traps = np.array(list(seq.grad_library.data.values()), dtype=object)
    istrap = np.asarray([len(g) == 5 for g in gradients_and_traps])
    isgradient = np.logical_not(istrap)

    # Actual split
    gradients = np.stack(list(gradients_and_traps[isgradient]), axis=-1)
    traps = np.stack(list(gradients_and_traps[istrap]), axis=-1)

    # Loop over [GRADIENTS], extract (amp, shape_id) -> row 1 (0-index based)
    gradient_shapes = gradients[:2]
    gradient_shapes = np.pad(gradient_shapes, ((0, 0), (0, 3)))

    # Loop over [TRAP], extract (amp, rise, flat, fall, delay) -> -> row 1-end (0-index based)
    trap_shapes = traps.T

    # Put back together
    gradient_and_traps_shapes = np.zeros((len(gradients_and_traps), 5), dtype=float)
    gradient_and_traps_shapes[isgradient] = gradient_shapes
    gradient_and_traps_shapes[istrap] = trap_shapes
    gradient_and_traps_shapes = np.pad(gradient_and_traps_shapes, ((1, 0), (0, 0)))

    # Loop over [ADC], extract (num, dwell, delay) -> -> row 1-3 (0-index based)
    adc = np.stack(list(seq.adc_library.data.values()), axis=-1)
    adc_shapes = adc[:3].T
    adc_shapes = np.pad(adc_shapes, ((1, 0), (0, 0)))

    # Loop over [BLOCKS], extract trigger flag (0: off; 1: on)
    hastrig = _check_trig_events(n_blocks, seq)[:, None]

    # Get rf, gx, gy, gz and adc indexes
    rf_idx = blocks[1]
    gx_idx = blocks[2]
    gy_idx = blocks[3]
    gz_idx = blocks[4]
    adc_idx = blocks[5]

    # Build block identifiers matrix
    rf = rf_shapes[rf_idx, 1:]  # (nblocks, 2); row = [mag_id, phase_id]
    # (nblocks, 4); row = [shape_id, 0, 0, 0] (grad) or [rise, flat, fall, delay] (trap)
    gx = gradient_and_traps_shapes[gx_idx, 1:]
    # (nblocks, 4); row = [shape_id, 0, 0, 0] (grad) or [rise, flat, fall, delay] (trap)
    gy = gradient_and_traps_shapes[gy_idx, 1:]
    # (nblocks, 4); row = [shape_id, 0, 0, 0] (grad) or [rise, flat, fall, delay] (trap)
    gz = gradient_and_traps_shapes[gz_idx, 1:]
    adc = adc_shapes[adc_idx]  # (nblocks, 3); row = [num, dwell, delay]
    block_shapes = [dur, rf, gx, gy, gz, adc, hastrig]
    block_shapes = np.concatenate(block_shapes, axis=1)

    # Find pure delay blocks
    # exclude duration, adc and trigger columns (isdelay.m only checks for rf, gx, gy and gz waveforms)
    tmp = block_shapes[:, 1:15]
    event_shapes = block_shapes[tmp.any(axis=1)]

    # Find unique events in parent blocks
    parent_block_shapes, parent_blocks_ids = np.unique(
        event_shapes, return_inverse=True, axis=0
    )
    parent_blocks = _get_parent_blocks(seq, block_shapes, parent_block_shapes)
    # parent_blocks_ids += 1 # ID = 0 is reserved to pure delay segment

    # Build block amplitudes matrix
    rf_amp = rf_shapes[rf_idx, 0][:, None]
    gx_amp = gradient_and_traps_shapes[gx_idx, 0][:, None]
    gy_amp = gradient_and_traps_shapes[gy_idx, 0][:, None]
    gz_amp = gradient_and_traps_shapes[gz_idx, 0][:, None]
    block_amps = [rf_amp, gx_amp, gy_amp, gz_amp]
    block_amps = np.concatenate(block_amps, axis=1)

    # Build Ceq structure
    ceq = SimpleNamespace(n_max=n_blocks)

    #  Set parent blocks
    # ceq.n_parent_blocks =


# %% local utils
def _get_blocks_durations(nevents, seq):
    dur = [seq.get_block(n + 1).block_duration for n in range(nevents)]
    dur = np.asarray(dur, dtype=float)
    dur = np.round(dur, 6)
    return dur


def _check_trig_events(nevents, seq):
    hastrig = []
    for n in range(nevents):
        b = seq.get_block(n + 1)
        if hasattr(b, "trig"):
            hastrig.append(1.0)
        else:
            hastrig.append(0.0)

    return np.asarray(hastrig, dtype=float)


def _get_parent_blocks(seq, blocks, parent_blocks_shapes):
    # Number of unique blocks
    n_parent_blocks = parent_blocks_shapes.shape[0]
    parent_blocks_idx = []

    # Find indexes
    for n in range(n_parent_blocks):
        # Compare each row in the matrix with the given row
        tmp = np.all(blocks == parent_blocks_shapes[n], axis=1)

        # Find the index of the first matching row
        parent_blocks_idx.append(np.where(tmp)[0][0])

    # Convert to array
    parent_blocks_idx = np.asarray(parent_blocks_idx)
    parent_blocks_idx += 1

    # Get blocks
    parent_blocks = [seq.get_block(n) for n in parent_blocks_idx]
    return parent_blocks
