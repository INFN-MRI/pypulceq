"""Pulseq to ceq converter."""

__all__ = ["seq2ceq"]

import copy
import math
from types import SimpleNamespace

import numpy as np

import pypulseq as pp


def seq2ceq(seqarg, max_view, max_slices, max_echo):
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

    # Loop over [BLOCKS], extract durations (rounded to 1e-6)
    dur, loop, trid = _get_dynamics(seq, max_view, max_slices, max_echo)

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
    hastrig = loop["trigout"][:, None]

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
    tmp = block_uid[:, 1:16]
    event_uid = block_uid[tmp.any(axis=1)]
    
    # Find unique non-delay events in parent blocks
    parent_blocks_ids = np.unique(
        event_uid, return_index=True, axis=0
    )[1]
    parent_blocks_ids = sorted(parent_blocks_ids)
    parent_block_uid = block_uid[parent_blocks_ids]
    
    # generate parent block list and module index for each row in loop
    parent_blocks, parent_blocks_idx = _get_parent_blocks(seq, block_uid, parent_block_uid, parent_blocks_ids)

    # Build block amplitudes matrix
    rf_amp = rf_events[rf_idx, 0]
    gx_amp = gradient_and_traps_events[gx_idx, 0]
    gy_amp = gradient_and_traps_events[gy_idx, 0]
    gz_amp = gradient_and_traps_events[gz_idx, 0]
    amplitudes = np.stack((rf_amp, gx_amp, gy_amp, gz_amp), axis=1)
    
    # Set parent blocks to maximum amplitude
    parent_blocks, amplitudes = _set_max_amplitudes(amplitudes, parent_blocks, parent_blocks_idx)
    
    # Compute delay
    textra = np.zeros_like(dur)
    textra[np.logical_not(tmp.any(axis=1))] = np.round(dur[np.logical_not(tmp.any(axis=1))] * 1e6) / 1e3
    
    # Fill scan dynamics with missing info
    loop["RFamplitude"] = amplitudes[:, 0]
    loop["Gamplitude"] = amplitudes[:, 1:]
    loop["textra"] = textra

    # Build Ceq structure
    # ceq = SimpleNamespace(n_max=n_blocks)

    #  Set parent blocks
    # ceq.n_parent_blocks =


# %% local utils
# Delay block
# 'textra', round(ceq.loop(n, 10)*1e6)/1e3, ... % msec
# 'core', i);

# Normal block
# 'Gamplitude',  [amp.gx amp.gy amp.gz]', ...
# 'RFoffset',    RFoffset, ...
# 'RFamplitude', RFamplitude, ...
# 'RFphase',     RFphase, ...
# 'DAQphase',    DAQphase, ...
# 'slice',       sl, ...
# 'echo',        echo+1, ...  % write2loop starts indexing at 1
# 'view',        view, ...
# 'textra',      0, ...  
# 'trigout',     trigout, ...
# 'rotmat',      rotmat, ...
# 'core', i);

def _get_dynamics(seq, max_view, max_slice, max_echo):
    
    sl = 1
    view = 1
    echo = 0
    adc_count = 0
    
    # get number of events
    nevents = len(seq.block_events)
    
    # initialize quantities (here struct of array instead of array of struct)
    RFoffset = np.zeros(nevents)
    RFphase = np.zeros(nevents)
    DAQphase = np.zeros(nevents)
    trigout = np.zeros(nevents)
    rotmat = np.eye(3, 3)
    rotmat = np.repeat(rotmat[None, ...], nevents, axis=0)
    views = np.ones(nevents)
    slices = np.ones(nevents)
    echoes = np.zeros(nevents)
    
    # non-loop variables (duration and TRID labels)
    dur = np.zeros(nevents)
    trid = -np.ones(nevents)
    
    # loop over evens
    for n in range(nevents):
        b = seq.get_block(n + 1)
        dur[n] = b.block_duration
        if b.rf is not None:
            RFoffset[n] = b.rf.freq_offset
            RFphase[n] = b.rf.phase_offset
        if b.adc is not None:
            DAQphase[n] = b.adc.phase_offset
            
            view = (adc_count % max_view) + 1
            sl = (adc_count // max_view) + 1
            assert sl <= max_slice, f"max number of slices (={max_slice}) ecxeeded"
            
            echo = adc_count // (max_view * max_slice)
            assert echo <= max_echo, f"max number of echoes (={max_echo}) ecxeeded"    
            adc_count += 1
        
        views[n] = view
        slices[n] = sl
        echoes[n] = echo
        
        if hasattr(b, "trig"):
            trigout[n] = 1.0
        if hasattr(b, "rotation"):
            rotmat[n] = b.rotation.rot_matrix
        if b.label is not None and b.label.label == "TRID":
            trid[n] = b.label.value
            
    # prepare loop dict
    loop = {
        "core": None,
        "modname": None,
        "Gamplitude": None, 
        "RFoffset": RFoffset,
        "RFamplitude": None,
        "RFphase": RFoffset,
        "DAQphase": DAQphase,
        "trigout": trigout,
        "textra": None,
        "rotmat": rotmat,
        "slice": slices,
        "echo": echoes,
        "view": views,
        }
    
    return dur, loop, trid
            

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
                    parent_blocks[n].gx.flat_time + parent_blocks[n].gx.rise_time / 2 + parent_blocks[n].gx.fall_time / 2
                )
                parent_blocks[n].gx.flat_area = parent_blocks[n].gx.amplitude * parent_blocks[n].gx.flat_time
        if parent_blocks[n].gy is not None:
            amplitudes[idx, 2] /= amp[2]
            if parent_blocks[n].gy.type == "grad":
                parent_blocks[n].gy.waveforms *= amp[2]
            else:
                parent_blocks[n].gy.amplitude = amp[2]
                parent_blocks[n].gy.area = parent_blocks[n].gy.amplitude * (
                    parent_blocks[n].gy.flat_time + parent_blocks[n].gy.rise_time / 2 + parent_blocks[n].gy.fall_time / 2
                )
                parent_blocks[n].gy.flat_area = parent_blocks[n].gy.amplitude * parent_blocks[n].gy.flat_time
        if parent_blocks[n].gz is not None:
            amplitudes[idx, 3] /= amp[3]
            if parent_blocks[n].gz.type == "grad":
                parent_blocks[n].gz.waveforms *= amp[3]
            else:
                parent_blocks[n].gz.amplitude = amp[3]
                parent_blocks[n].gz.area = parent_blocks[n].gz.amplitude * (
                    parent_blocks[n].gz.flat_time + parent_blocks[n].gz.rise_time / 2 + parent_blocks[n].gz.fall_time / 2
                )
                parent_blocks[n].gz.flat_area = parent_blocks[n].gz.amplitude * parent_blocks[n].gz.flat_time
                    
    return parent_blocks, amplitudes
    