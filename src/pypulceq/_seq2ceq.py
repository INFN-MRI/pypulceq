"""Seq2Ceq subroutines."""

__all__ = ["seq2ceq"]

import warnings
import numpy as np

import pypulse as pp

from . import _blocks

from types import SimpleNamespace

def seq2ceq(seqarg, verbose=False, nMax=None, ignoreSegmentLabels=False):
    """
    Convert a Pulseq file to a PulCeq struct.

    Parameters
    ----------
    seqarg : str or mr.Sequence
        A seq object, or name of a .seq file.
    verbose : bool, optional
        Print some info to the terminal. Default is False.
    nMax : int, optional
        Only parse the first nMax blocks in the .seq file. If None, parse all blocks.
    ignoreSegmentLabels : bool, optional
        Treat each block as a segment. Use with caution! Default is False.

    Returns
    -------
    SimpleNamespace
        A struct representing the PulCeq struct.
        
    """
    # Get seq object
    ceq, seq, blockEvents = _get_seq_object(seqarg, nMax)

    # Get parent blocks
    ceq, parentBlockIDs = _get_parent_blocks(ceq, seq, blockEvents, verbose)

    # Get segment (block group) definitions
    ceq, segmentID2Ind, segmentIDs = _get_segment_definitions(ceq, seq, parentBlockIDs, ignoreSegmentLabels)

    # Get dynamic scan information
    ceq = _get_dynamic_scan_info(ceq, seq, parentBlockIDs, segmentID2Ind, segmentIDs)

    # Check that the execution of blocks throughout the sequence
    # is consistent with the segment definitions
    _check_consistent_segment_definitions(ceq)

    return ceq

# %% local utils
def _get_seq_object(seqarg, nMax):
    if isinstance(seqarg, str):
        seq = pp.Sequence()
        seq.read(seqarg)
    else:
        if not isinstance(seqarg, pp.Sequence):
            raise ValueError('First argument is not an mr.Sequence object')
        seq = seqarg

    nEvents = 7 # Pulseq 1.4.0
    blockEvents = np.array(seq.blockEvents)
    blockEvents = np.reshape(blockEvents, (len(seq.blockEvents), nEvents))
    
    # initialize ceq
    ceq = SimpleNamespace()

    # Number of blocks (rows in .seq file) to step through
    if nMax is None:
        ceq.nMax = blockEvents.shape[0]
    else:
        ceq.nMax = nMax

    return ceq, seq, blockEvents


def _get_parent_blocks(ceq, seq, blockEvents, verbose):
    parentBlockIDs = []  # Initialize as an empty list
    parentBlockIndex = [0] # First block is unique by definition

    print(f'seq2ceq: Getting block 1/{ceq.nMax}', end='', flush=True)
    prev_n = 1  # Progress update trackers
    for n in range(ceq.nMax):
        if n % 500 == 0 or n == ceq.nMax:
            print('\b' * len(f'seq2ceq: Getting block {prev_n}/{ceq.nMax}'), end='', flush=True)
            prev_n = n
            print(f'seq2ceq: Getting block {n}/{ceq.nMax}', end='', flush=True)
            if n == ceq.nMax:
                print()

        # Pure delay blocks are handled separately
        b = seq.getBlock(n)
        if _blocks.isdelayblock(b):
            parentBlockIDs.append(0)
            continue

        IsSame = np.zeros(len(parentBlockIndex))
        for p in range(len(parentBlockIndex)):
            n2 = parentBlockIndex[p]
            IsSame[p] = _blocks.compareblocks(seq, blockEvents[n], blockEvents[n2], n, n2)

        if np.sum(IsSame) == 0:
            if verbose:
                print(f'\nFound new block on line {n}\n')
            parentBlockIndex.append(n)  # Append new block index to the list
            parentBlockIDs.append(len(parentBlockIndex))
        else:
            I = np.where(IsSame)[0]
            parentBlockIDs.append(I[0])

    ceq.nParentBlocks = len(parentBlockIndex)
    ceq.parentBlocks = [seq.getBlock(parentBlockIndex[p]) for p in parentBlockIndex]

    # Determine max amplitude across blocks
    for p in range(len(parentBlockIndex)):
        ceq.parentBlocks[p].amp = SimpleNamespace(rf=0.0, gx=0.0, gy=0.0, gz=0.0)
        for n in range(ceq.nMax):
            if parentBlockIDs[n] != p + 1:
                continue
            block = seq.getBlock(n)
            if block.rf is not None:
                ceq.parentBlocks[p].amp.rf = max(ceq.parentBlocks[p].amp.rf, np.max(np.abs(block.rf.signal)))
            for ax in ['gx', 'gy', 'gz']:
                g = block[ax]
                if g is not None:
                    if g.type == 'trap':
                        gamp = np.abs(g.amplitude)
                    else:
                        gamp = np.max(np.abs(g.waveform))
                    tmp = getattr(ceq.parentBlocks[p].amp, ax)
                    tmp = max(tmp, gamp)
                    setattr(ceq.parentBlocks[p].amp, ax, tmp)

    # Set parent block waveform amplitudes to max
    for p in range(ceq.nParentBlocks):
        b = ceq.parentBlocks[p]  # shorthand
        if b.rf is not None:
            ceq.parentBlocks[p].rf.signal = b.rf.signal / np.max(np.abs(b.rf.signal)) * b.amp.rf
        for ax in ['gx', 'gy', 'gz']:
            g = b[ax]
            if g is not None:
                block = getattr(ceq.parentBlocks[p], ax)
                if g.type == 'trap':
                    block.amplitude = getattr(b.amp, ax)
                else:
                    tmp = getattr(b, ax)
                    if np.max(np.abs(tmp.waveform)) > 0:
                        block.waveform = tmp.waveform / np.max(np.abs(tmp.waveform)) * getattr(b.amp, ax)
                setattr(ceq.parentBlocks[p], ax, block)
                    
    return ceq, parentBlockIDs
                        
                        
def _get_segment_definitions(ceq, seq, parentBlockIDs, ignoreSegmentLabels):
    previouslyDefinedSegmentIDs = []

    if not ignoreSegmentLabels:
        segmentIDs = np.zeros(ceq.nMax, dtype=int)  # Keep track of which segment each block belongs to
        Segments = {}  # Dictionary to store segments

        for n in range(ceq.nMax):
            b = seq.getBlock(n)

            if 'label' in b.__dict__:
                if b.label.label == 'TRID':  # Marks start of segment
                    activeSegmentID = b.label.value

                    if activeSegmentID not in previouslyDefinedSegmentIDs:
                        # Start new segment
                        firstOccurrence = True
                        previouslyDefinedSegmentIDs.append(activeSegmentID)
                        Segments[activeSegmentID] = []
                    else:
                        firstOccurrence = False

            if 'firstOccurrence' not in locals():
                raise ValueError('First block must contain a segment ID')

            # Add block to segment
            if firstOccurrence:
                Segments[activeSegmentID].append(parentBlockIDs[n])

            segmentIDs[n] = activeSegmentID

    else:
        # Each block becomes its own segment
        ceq.groups = [SimpleNamespace(groupID=p, nBlocksInGroup=1, blockIDs=[p]) for p in range(ceq.nParentBlocks)]

        # Add delay segment (dedicated segment that's always defined)
        ceq.groups.append(SimpleNamespace(groupID=ceq.nParentBlocks, nBlocksInGroup=1, blockIDs=[0]))

        segmentIDs = parentBlockIDs.copy()
        segmentIDs[parentBlockIDs == 0] = ceq.nParentBlocks

        segmentID2Ind = np.arange(ceq.nParentBlocks)

    # Squash the Segments array and redefine the Segment IDs accordingly
    # This is needed since the interpreter assumes that segment ID = index into segment array.
    if not ignoreSegmentLabels:
        iSeg = 0  # Segment array index
        for segmentID in range(len(Segments)):
            if Segments[segmentID]:
                segmentID2Ind[segmentID] = iSeg
                ceq.groups[iSeg].groupID = iSeg
                ceq.groups[iSeg].nBlocksInGroup = len(Segments[segmentID])
                ceq.groups[iSeg].blockIDs = Segments[segmentID]
                iSeg += 1

    ceq.nGroups = len(ceq.groups)
    
    return ceq, segmentID2Ind, segmentIDs
    
    
def _get_dynamic_scan_info(ceq, seq, parentBlockIDs, segmentID2Ind, segmentIDs):
    ceq.loop = np.zeros((ceq.nMax, 10), dtype=float)

    for n in range(ceq.nMax):
        b = seq.getBlock(n)
        p = parentBlockIDs[n]

        if p == 0:  # Delay block
            ceq.loop[n] = _blocks.getdynamics(b, segmentID2Ind[segmentIDs[n]], p)
        else:
            ceq.loop[n] = _blocks.getdynamics(b, segmentID2Ind[segmentIDs[n]], p, ceq.parentBlocks[p])
            
    return ceq

def _check_consistent_segment_definitions(ceq):
    n = 0
    while n < ceq.nMax:
        i = int(ceq.loop[n, 0])  # Segment ID
        for j in range(ceq.groups[i].nBlocksInGroup):
            p = int(ceq.loop[n, 1])  # Parent block ID
            p_ij = ceq.groups[i].blockIDs[j]
            if p != p_ij:
                # Warning about inconsistent segment definitions
                warning_msg = ('Sequence contains inconsistent segment definitions. '
                               'This may occur due to programming error (possibly fatal), '
                               'or if an arbitrary gradient resembles that from another block '
                               'except with opposite sign or scaled by zero (which is probably ok). '
                               'Expected parent block ID {}, found {} (block {})').format(p_ij, p, n)
                warnings.warn(warning_msg, category=UserWarning)
            n += 1