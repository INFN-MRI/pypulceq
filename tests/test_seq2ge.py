"""Test Ceq structure to TOPPE dictionary conversion subroutines."""

from pypulceq import demo
from pypulceq import _seq2ceq
from pypulceq import _ceq2files
from pypulceq import _toppe


def test_cartesian():
    # design sequence
    seq = demo.design_gre((32, 32), (32, 32))

    # convert to ceq
    ceq = _seq2ceq.seq2ceq(seq)

    # get system specs
    sys = _toppe.SystemSpecs(
        maxGrad=seq.system.max_grad / seq.system.gamma * 100,
        maxSlew=seq.system.max_slew / seq.system.gamma / 10,
        maxView=32,
        maxSlice=32,
        rfDeadTime=seq.system.rf_dead_time * 1e6,
        rfRingdownTime=seq.system.rf_ringdown_time * 1e6,
        adcDeadTime=seq.system.adc_dead_time * 1e6,
        B0=seq.system.B0,
    )

    # get sequence dictionary
    toppe_dict = _ceq2files.ceq2files("test", ceq, sys)

    assert toppe_dict["b1scaling_name"] == "module1.mod"
    assert toppe_dict["readout_name"] == "module6.mod"
    assert (len(toppe_dict["loop"])) == 6 * 32 * (
        32 + 1
    )  # blocks_per_TR * Ny * (Nz + 1), where 1 is the dummy block
