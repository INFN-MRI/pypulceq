"""Pulseq block conversion subroutines."""

__all__ = ["getdynamics", "compareblocks", "isdelayblock"]

from types import SimpleNamespace

import numpy as np

def getdynamics(block, segmentID, parentBlockID, parentBlock):
    """
    Return vector containing waveform amplitudes, RF/ADC phase, etc,
    for a Pulseq block, in physical (Pulseq) units.

    Parameters
    ----------
    block : SimpleNamespace
        Pulseq block.
    segmentID : int
        Segment ID.
    parentBlockID : int
        Parent block ID.
    parentBlock : SimpleNamespace
        Parent block.

    Returns
    -------
    numpy.ndarray
        Array containing segmentID, parentBlockID, rfamp, rfphs, rffreq, amp.gx, amp.gy, amp.gz, recphs, blockDuration
        
    """
    # Defaults
    rfamp = 0
    rfphs = 0
    rffreq = 0
    amp = SimpleNamespace(gx=0, gy=0, gz=0)
    recphs = 0

    if block.rf is not None:
        rfamp = max(abs(block.rf.signal))
        rfphs = block.rf.phaseOffset
        rffreq = block.rf.freqOffset

    for ax in ['gx', 'gy', 'gz']:
        g = getattr(block, ax, None)
        if g is not None:
            if g.type == 'trap':
                setattr(amp, ax, g.amplitude)
            else:
                # Need to check polarity (sign) with respect to parent block
                mx = max(g.waveform)
                mn = min(g.waveform)
                pbmx = max(getattr(parentBlock, ax).waveform)
                pbmn = min(getattr(parentBlock, ax).waveform)
                setattr(amp, ax, max(abs(g.waveform)))
                if (mx > abs(mn)) ^ (pbmx > abs(pbmn)):
                    setattr(amp, ax, -getattr(amp, ax))

    if block.adc is not None:
        recphs = block.adc.phaseOffset

    loop = [segmentID, parentBlockID, rfamp, rfphs, rffreq, amp.gx, amp.gy, amp.gz, recphs, block.blockDuration]
    return np.asarray(loop)


def compareblocks(seq, b1Events, b2Events, n1, n2):
    """
    Compare two Pulseq blocks.

    Parameters
    ----------
    seq : Pulseq object
        Pulseq object.
    b1Events : list
        Events of block 1.
    b2Events : list
        Events of block 2.
    n1 : int
        Block ID of block 1.
    n2 : int
        Block ID of block 2.

    Returns
    -------
    bool
        True if the blocks are the same, False otherwise.
        
    """
    issame = True

    b1 = seq.getBlock(n1)
    b2 = seq.getBlock(n2)

    # Compare duration
    tol = 1e-6  # s
    if abs(b1.blockDuration - b2.blockDuration) > tol:
        issame = False
        return issame

    # Is a trigger present/absent in both blocks
    if hasattr(b1, 'trig') != hasattr(b2, 'trig'):
        issame = False
        return issame

    # Are gradients non-unique (same shapes)
    for ax in ['gx', 'gy', 'gz']:
        if not _comparegradients(b1.__dict__[ax], b2.__dict__[ax]):
            issame = False
            return issame

    # Are ADC events same and consistent
    if not _compareadc(seq, n1, n2):
        issame = False
        return issame

    # Are RF events non-unique (same shapes)
    if not _comparerf(b1, b2, b1Events, b2Events, seq):
        issame = False
        return issame

    return issame


def isdelayblock(block):
    """
    Returns True if block contains no waveforms.

    Parameters
    ----------
    block : SimpleNamespace
        Pulseq block.

    Returns
    -------
    bool
        True if the block contains no waveforms, False otherwise.
    """
    return (not block.rf and
            not block.adc and
            not block.gx and
            not block.gy and
            not block.gz)

# %% local utils
def _comparerf(b1, b2, b1Events, b2Events, seq):
    """
    Compare RF events.

    Parameters
    ----------
    b1 : SimpleNamespace
        Block 1.
    b2 : SimpleNamespace
        Block 2.
    b1Events : list
        Events of block 1.
    b2Events : list
        Events of block 2.
    seq : Pulseq object
        Pulseq object.

    Returns
    -------
    bool
        True if RF events are the same, False otherwise.
        
    """
    if not b1.rf and not b2.rf:
        return True

    if bool(b1.rf) != bool(b2.rf):
        return False

    RFid1 = b1Events[1]
    RFid2 = b2Events[1]
    RFevent1 = seq.rfLibrary.data[RFid1]
    RFevent2 = seq.rfLibrary.data[RFid2]
    magShapeID1 = RFevent1.array[2]
    magShapeID2 = RFevent2.array[2]
    phaseShapeID1 = RFevent1.array[3]
    phaseShapeID2 = RFevent2.array[3]

    return magShapeID1 == magShapeID2 and phaseShapeID1 == phaseShapeID2


def _comparegradients(g1, g2):
    """
    Compare gradient events.

    Parameters
    ----------
    g1 : SimpleNamespace
        Gradient event 1.
    g2 : SimpleNamespace
        Gradient event 2.

    Returns
    -------
    bool
        True if gradient events are the same, False otherwise.
        
    """
    if not g1 and not g2:
        return True

    if bool(g1) != bool(g2):
        return False

    if g1.type != g2.type:
        return False

    if g1.type == 'trap':
        return (g1.riseTime == g2.riseTime and
                g1.flatTime == g2.flatTime and
                g1.fallTime == g2.fallTime and
                g1.delay == g2.delay)
    else:
        return g1.shape_id == g2.shape_id


def _compareadc(seq, n1, n2):
    """
    Compare ADC events.

    Parameters
    ----------
    seq : Pulseq object
        Pulseq object.
    n1 : int
        Block ID of block 1.
    n2 : int
        Block ID of block 2.

    Returns
    -------
    bool
        True if ADC events are the same, False otherwise.
        
    """
    adc1 = seq.getBlock(n1).adc
    adc2 = seq.getBlock(n2).adc

    if not adc1 and not adc2:
        return True

    if bool(adc1) != bool(adc2):
        return False

    return (adc1.numSamples == adc2.numSamples and
            adc1.dwell == adc2.dwell and
            adc1.delay == adc2.delay)
