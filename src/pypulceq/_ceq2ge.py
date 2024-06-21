"""PulCeq to TOPPE converter."""

__all__ = ["ceq2ge"]

from types import SimpleNamespace

import numpy as np

from . import _interp
from . import _toppe

def ceq2ge(sequence_name : str, ceq : SimpleNamespace, sys : SimpleNamespace, ignore_trigger : bool = False, sequence_path : str = None, verbose : bool = False):
    """
    Write a PulCeq struct to a set of files that can be executed
    on GE scanners using the TOPPE interpreter (v6).   

    Parameters
    ----------
    sequence_name : str
        Sequence name.
    ceq : SimpleNamespace
        PulCeq struct, based on github/HarmonizedMRI/PulCeq/src/pulCeq.h
    sys : _toppe.SystemSpecs
        TOPPE SystemSpecs object with GE scanner specifications.
    ignore_trigger : bool, optional
        If True, ignore all TTL pulses. The default is ``False``.
    sequence_path : str, optional
        Sequence files on the scanner. The default is
        ``"/usr/g/research/pulseq/v6/seq2ge/{sequence_name}"``.
    verbose : bool, optional
        Display info. The default is ``False`` (silent mode).

    """
    # Determine B1 scaling file by searching higher rf peak amongst parent blocks
    b1scaling_idx = _find_b1_scaling(ceq)
    
    # Determine readout.mod as the last parent block containing ADC (to mimick ceq2ge.m)
    readout_idx = _find_readout(ceq)
    
    # Loop over parent blocks and convert blocks to modules
    if verbose:
        print("Converting Pulseq blocks to TOPPE modules...", end="\tab")
    modules = {"delay": None} # dummy wait module
    for p in range(1, ceq.n_parent_blocks):
        modules[f"module{p}"] = _block2mod(p, ceq.parent_blocks[p], sys, ceq.sys.grad_raster_time)
    if verbose:
        print("done!\n")
        
    # Build core lut
    modnames = list(modules.keys())
    cores = {}
    for c in range(ceq.n_segments):
        cores[f"core{c}"] = [modnames[b] for b in ceq.blocks_in_segment]
        
    # Determine B1 scaling filename and readout filename
    b1scaling_name = modnames[b1scaling_idx]
    readout_name = modnames[readout_idx]

    # fix Loop
    ceq.loop.modname = modnames[ceq.parent_blocks_idx]
    ceq.loop.core = ceq.segments_idx
    view = (ceq.loop.adc_idx % sys.maxView) + 1
    sl = (ceq.loop.adc_idx // sys.maxView) + 1
    assert sl > sys.maxSlice, f"max number of slices (={sys.maxSlice}) ecxeeded (requesting slice_idx = {sl})"
    echo = (ceq.loop.adc_idx // (sys.maxView * sys.maxSlice)) + 1
    assert echo > sys.maxEcho, f"max number of echoes (={sys.maxEcho}) ecxeeded (requesting echo_idx = {echo})"
    ceq.loop.view = view.astype(int)
    ceq.loop.slice = sl.astype(int)
    ceq.loop.echo = echo.astype(int)
    loop = _soa2aos(ceq.n_max, ceq.loop)
        
    # Build seqdict
    seqdict = {"sys": sys.__dict__, "modules": modules, "cores": cores, "loop": loop, "b1scaling_name": b1scaling_name,
               "readout_name": readout_name}
    
    # Export
    _toppe.write_sequence(sequence_name, seqdict, ignore_trigger, sequence_path, verbose)
    if verbose:
        print(f"Sequence file {sequence_name} ready for execution on GE scanners\n")


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


def _block2mod(p, block, sys, dt):
    if block.rf is not None:
        assert block.adc is None, "A block can either have RF event or ADC, not both"
    if block.adc is not None:
        assert block.rf is None, "A block can either have RF event or ADC, not both"
        
    # initialize output module
    mod = {"ofname": f"module{p}.mod", "block_duration": block.block_duration}
    npre = 0
    rfres = None
    rf = []
    gx = []
    gy = []
    gz = []
    
    if block.rf is not None:
        rf, rfres, npre = _interp.rf2ge(p, block.rf, sys)
        mod["rf"] = rf
    if block.gx is not None:
        gx = _interp.grad2ge(p, block.gx, sys, dt)
        mod["gx"] = gx
    if block.gy is not None:
        gy = _interp.grad2ge(p, block.gy, sys, dt)
        mod["gy"] = gy
    if block.gz is not None:
        gz = _interp.grad2ge(p, block.gz, sys, dt)
        mod["gz"] = gz
    if block.adc is not None:
        rfres, npre = _interp.adc2ge(p, block, sys)
    if hasattr(block, "trig"):
        mod["trig"] = {"delay": block.trig.delay}

    n = max(len(rf), len(gx), len(gy), len(gz)) # number of 4us samples in module
    if rfres is None:
        mod["nChop"] = (0, 0)
    else:
        mod["nChop"] = (npre, n-npre-rfres) # trim this many samples from beginning and end of RF/ADC window

    return mod

def _soa2aos(numel, input):
    output = []
    for n in range(numel):
        el = {
            "modname": input.modname[n],
            "Gamplitude": input.Gamplitude[n],
            "RFoffset": input.RFoffset[n],
            "RFphase": input.RFphase[n],
            "DAQphase": input.DAQphase[n],
            "trigout": input.trigout[n],
            "textra": input.textra[n],
            "rotmat": input.rotmat[n],
            "view": input.view[n],
            "slice": input.slice[n],
            "echo": input.echo[n],
            "core": input.core[n],
            }
        output.append(el)
    return output
        