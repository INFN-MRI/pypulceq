"""Dataclass representation of CEQ structure for C-code interfacing."""


from dataclasses import dataclass
from typing import Union

from numpy.typing import NDArray


@dataclass
class PulseqShapeArbitrary: # noqa
    n_samples: int # number of waveform samples
    raster: float # sample duration (sec)
    magnitude: NDArray[float] # magnitude waveform (normalized)
    phase: NDArray[float] # phase waveform (rad)
    
    def __post_init__(self): # noqa
        self.magnitude = self.magnitude.astype(float)
        self.phase = self.phase.astype(float)

@dataclass
class PulseqShapeTrap: # noqa
    rise_time: float # sec
    flat_time: float # sec
    fall_time: float # sec

@dataclass
class PulseqRF: # noqa
    ptype: int # NULL or ARBITRARY
    delay: float # sec
    dead_time: float # sec
    ringdown_time: float # sec
    wav: PulseqShapeArbitrary # arbitrary waveform, normalized amplitude

@dataclass
class PulseqGrad: # noqa
    ptype: int # NULL, TRAP or ARBITRARY
    delay: float # sec
    shape: Union[PulseqShapeArbitrary, PulseqShapeTrap]

@dataclass
class PulseqADC: # noqa
    ptype: int # NULL or ADC
    num_samples: int # number of ADC samples
    dwell: float # sec
    delay: float # sec
    dead_time: float # sec

@dataclass
class PulseqTrig: # noqa
    ptype: int # NULL or OUTPUT or ?
    channel: int # EXT1 or ?n
    delay: float # sec
    duration: float # sec

@dataclass
class PulseqBlock: # noqa
    idx: int # unique block ID
    duration: float # sec
    rf: PulseqRF
    gx: PulseqGrad
    gy: PulseqGrad
    gz: PulseqGrad
    adc: PulseqADC
    trig: PulseqTrig
    
    # vectors available for use as needed by the client program
    n_val_1: int # number of int values. Must be defined.
    val_1: NDArray[int] # to be allocated dynamically by the client program
    n_val_2: int # number of float values. Must be defined.
    val_2: NDArray[float] # to be allocated dynamically by the client program
    
    def __post_init__(self): # noqa
        self.val_1 = self.val_1.astype(float)
        self.val_2 = self.val_2.astype(float)

@dataclass
class BlockGroup: # noqa
    group_idx: int
    n_blocks_in_group: int # number of blocks in group
    block_idxs: NDArray[int] # block id's in this group
    
    def __post_init__(self): # noqa
        self.block_idxs = self.block_idxs.astype(float)

@dataclass
class Ceq: # noqa
    n_parent_blocks: int
    parent_blocks: NDArray[PulseqBlock]
    n_groups: int
    groups: int
    parent_blocks: NDArray[BlockGroup]
    loop: NDArray[float] # Dynamic scan settings: waveform amplitudes, phase offsets, etc (n_max, nLC)
    n_max: int
    
    def __post_init__(self): # noqa
        self.loop = self.loop.astype(float)
