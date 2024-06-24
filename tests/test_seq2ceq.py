"""Test pp.Sequence to Ceq structure conversion subroutines."""

import numpy as np
import numpy.testing as npt

from pypulceq import demo
from pypulceq import _seq2ceq


def test_cartesian():
    # design sequence
    seq = demo.design_gre((32, 32), (32, 32))

    # convert to ceq
    ceq = _seq2ceq.seq2ceq(seq)

    # we expect 7 parent blocks
    # (0=delay, 1=rf, 2=slab rephaser, 3=pre-/re-phasers, 4=read, 5=spoil, 6=read+adc)
    assert len(ceq.parent_blocks) == 7
    assert ceq.n_parent_blocks == 7

    # RF block
    assert ceq.parent_blocks[1].rf is not None
    assert ceq.parent_blocks[1].gx is None
    assert ceq.parent_blocks[1].gy is None
    assert ceq.parent_blocks[1].gz is not None
    assert ceq.parent_blocks[1].adc is None

    # slice rephaser
    assert ceq.parent_blocks[2].rf is None
    assert ceq.parent_blocks[2].gx is None
    assert ceq.parent_blocks[2].gy is None
    assert ceq.parent_blocks[2].gz is not None
    assert ceq.parent_blocks[2].adc is None

    # pre- / re- winders
    assert ceq.parent_blocks[3].rf is None
    assert ceq.parent_blocks[3].gx is not None
    assert ceq.parent_blocks[3].gy is not None
    assert ceq.parent_blocks[3].gz is not None
    assert ceq.parent_blocks[3].adc is None

    # readout (dummy block)
    assert ceq.parent_blocks[4].rf is None
    assert ceq.parent_blocks[4].gx is not None
    assert ceq.parent_blocks[4].gy is None
    assert ceq.parent_blocks[4].gz is None
    assert ceq.parent_blocks[4].adc is None

    # spoiler
    assert ceq.parent_blocks[5].rf is None
    assert ceq.parent_blocks[5].gx is None
    assert ceq.parent_blocks[5].gy is None
    assert ceq.parent_blocks[5].gz is not None
    assert ceq.parent_blocks[5].adc is None

    # readout (imaging block)
    assert ceq.parent_blocks[6].rf is None
    assert ceq.parent_blocks[6].gx is not None
    assert ceq.parent_blocks[6].gy is None
    assert ceq.parent_blocks[6].gz is None
    assert ceq.parent_blocks[6].adc is not None

    # check there are no missing blocks in loop
    npt.assert_array_equal(
        np.sort(np.unique(ceq.parent_blocks_idx)), [1, 2, 3, 4, 5, 6]
    )  # no delays

    # check segments
    assert ceq.n_segments == 2
    assert (len(ceq.blocks_in_segment)) == 2
    npt.assert_array_equal(ceq.blocks_in_segment[0], [1, 2, 3, 4, 3, 5])
    npt.assert_array_equal(ceq.blocks_in_segment[1], [1, 2, 3, 6, 3, 5])
