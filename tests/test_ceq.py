"""Test Ceq structure."""

import struct

from types import SimpleNamespace
import numpy as np

from pypulceq._ceq import PulseqShapeArbitrary
from pypulceq._ceq import PulseqShapeTrap
from pypulceq._ceq import PulseqRF
from pypulceq._ceq import PulseqGrad
from pypulceq._ceq import PulseqADC
from pypulceq._ceq import PulseqTrig
from pypulceq._ceq import PulseqBlock
from pypulceq._ceq import Segment
from pypulceq._ceq import Ceq


def test_pulseq_shape_arbitrary():
    data = {
        "n_samples": 100,
        "raster": 0.01,
        "waveform": [0.1, 0.2, 0.3],
    }
    obj = PulseqShapeArbitrary(**data)
    assert obj.n_samples == 100
    assert obj.raster == 0.01
    assert np.array_equal(obj.waveform, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">i", 100)
        + struct.pack(">f", 0.01)
        + np.array([0.1, 0.2, 0.3], dtype=np.float32).astype(">f4").tobytes()
    )
    assert serialized == expected


def test_pulseq_shape_trap():
    data = {"rise_time": 0.1, "flat_time": 0.2, "fall_time": 0.3}
    obj = PulseqShapeTrap(**data)
    assert obj.rise_time == 0.1
    assert obj.flat_time == 0.2
    assert obj.fall_time == 0.3

    # Test serialization
    serialized = obj.to_bytes()
    expected = struct.pack(">f", 0.1) + struct.pack(">f", 0.2) + struct.pack(">f", 0.3)
    assert serialized == expected


def test_pulseq_rf():
    data = SimpleNamespace(
        signal=np.asarray([0.1, 0.2]) * np.exp(1j * np.asarray([0.01, 0.02])),
        t=[0.001, 0.002],
        shape_dur=0.002,
        delay=0.1,
        freq_offset=100.0,
        phase_offset=0.5,
    )
    obj = PulseqRF.from_struct(data)
    assert obj.type == 1
    assert obj.n_samples == 2
    assert np.array_equal(obj.rho, np.array([0.1, 0.2], dtype=np.float32))
    assert np.array_equal(obj.theta, np.array([0.01, 0.02], dtype=np.float32))
    assert np.array_equal(obj.t, np.array([0.001, 0.002], dtype=np.float32))
    assert obj.shape_dur == 0.002
    assert obj.delay == 0.1
    assert obj.freq_offset == 100.0
    assert obj.phase_offset == 0.5
    assert obj.max_b1 == 0.2

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">h", 1)
        + struct.pack(">h", 2)
        + np.array([0.1, 0.2], dtype=np.float32).astype(">f4").tobytes()
        + np.array([0.01, 0.02], dtype=np.float32).astype(">f4").tobytes()
        + np.array([0.001, 0.002], dtype=np.float32).astype(">f4").tobytes()
        + struct.pack(">f", 0.002)
        + struct.pack(">f", 0.1)
        + struct.pack(">f", 100.0)
        + struct.pack(">f", 0.5)
        + struct.pack(">f", 0.2)
    )
    assert serialized == expected


def test_pulseq_grad():
    data = SimpleNamespace(
        type="trap",
        amplitude=0.8,
        delay=0.05,
        rise_time=0.1,
        flat_time=0.2,
        fall_time=0.3,
    )
    obj = PulseqGrad.from_struct(data)
    assert obj.type == 1
    assert obj.amplitude == 0.8
    assert obj.delay == 0.05
    assert isinstance(obj.shape, PulseqShapeTrap)
    assert obj.shape.rise_time == 0.1
    assert obj.shape.flat_time == 0.2
    assert obj.shape.fall_time == 0.3

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">h", 1)
        + struct.pack(">f", 0.8)
        + struct.pack(">f", 0.05)
        + obj.shape.to_bytes()
    )
    assert serialized == expected


def test_pulseq_adc():
    data = SimpleNamespace(
        num_samples=1000, dwell=0.0001, delay=0.02, freq_offset=50.0, phase_offset=0.3
    )
    obj = PulseqADC.from_struct(data)
    assert obj.type == 1
    assert obj.num_samples == 1000
    assert obj.dwell == 0.0001
    assert obj.delay == 0.02
    assert obj.freq_offset == 50.0
    assert obj.phase_offset == 0.3

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">h", 1)
        + struct.pack(">i", 1000)
        + struct.pack(">f", 0.0001)
        + struct.pack(">f", 0.02)
        + struct.pack(">f", 50.0)
        + struct.pack(">f", 0.3)
    )
    assert serialized == expected


def test_pulseq_trig():
    data = SimpleNamespace(channel="ext1", delay=0.01, duration=0.05)
    obj = PulseqTrig.from_struct(data)
    assert obj.type == 1
    assert obj.channel == 2
    assert obj.delay == 0.01
    assert obj.duration == 0.05

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">i", 1)
        + struct.pack(">i", 2)
        + struct.pack(">f", 0.01)
        + struct.pack(">f", 0.05)
    )
    assert serialized == expected


def test_pulseq_block():
    data = {
        "ID": 1,
        "block_duration": 0.5,
        "rf": SimpleNamespace(
            signal=np.asarray([0.1, 0.2]) * np.exp(1j * np.asarray([0.01, 0.02])),
            t=[0.001, 0.002],
            shape_dur=0.002,
            delay=0.1,
            freq_offset=100.0,
            phase_offset=0.5,
        ),
        "gx": SimpleNamespace(
            type="trap",
            amplitude=0.8,
            delay=0.05,
            rise_time=0.1,
            flat_time=0.2,
            fall_time=0.3,
        ),
        "gy": SimpleNamespace(
            type="trap",
            amplitude=0.8,
            delay=0.05,
            rise_time=0.1,
            flat_time=0.2,
            fall_time=0.3,
        ),
        "gz": SimpleNamespace(
            type="trap",
            amplitude=0.8,
            delay=0.05,
            rise_time=0.1,
            flat_time=0.2,
            fall_time=0.3,
        ),
        "adc": SimpleNamespace(
            num_samples=1000,
            dwell=0.0001,
            delay=0.02,
            freq_offset=50.0,
            phase_offset=0.3,
        ),
        "trig": SimpleNamespace(channel="ext1", delay=0.01, duration=0.05),
    }
    obj = PulseqBlock.from_dict(data)
    assert obj.ID == 1
    assert obj.block_duration == 0.5
    assert obj.rf.type == 1
    assert obj.gx.type == 1
    assert obj.gy.type == 1
    assert obj.gz.type == 1
    assert obj.adc.type == 1
    assert obj.trig.type == 1

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">i", 1)
        + struct.pack(">f", 0.5)
        + obj.rf.to_bytes()
        + obj.gx.to_bytes()
        + obj.gy.to_bytes()
        + obj.gz.to_bytes()
        + obj.adc.to_bytes()
        + obj.trig.to_bytes()
    )
    assert serialized == expected


def test_segment():
    data = {"segment_id": 1, "n_blocks_in_segment": 2, "block_ids": [1, 2]}
    obj = Segment.from_dict(data)
    assert obj.segment_id == 1
    assert obj.n_blocks_in_segment == 2
    assert np.array_equal(obj.block_ids, np.array([1, 2], dtype=np.int16))

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">h", 1)
        + struct.pack(">h", 2)
        + np.array([1, 2], dtype=np.int16).astype(">i2").tobytes()
    )
    assert serialized == expected


def test_ceq():
    data = SimpleNamespace(
        n_max=10,
        n_parent_blocks=2,
        n_segments=1,
        parent_blocks=[
            {
                "ID": 1,
                "block_duration": 0.5,
                "rf": SimpleNamespace(
                    signal=np.asarray([0.1, 0.2])
                    * np.exp(1j * np.asarray([0.01, 0.02])),
                    t=[0.001, 0.002],
                    shape_dur=0.002,
                    delay=0.1,
                    freq_offset=100.0,
                    phase_offset=0.5,
                ),
                "gx": SimpleNamespace(
                    type="trap",
                    amplitude=0.8,
                    delay=0.05,
                    rise_time=0.1,
                    flat_time=0.2,
                    fall_time=0.3,
                ),
                "gy": SimpleNamespace(
                    type="trap",
                    amplitude=0.8,
                    delay=0.05,
                    rise_time=0.1,
                    flat_time=0.2,
                    fall_time=0.3,
                ),
                "gz": SimpleNamespace(
                    type="trap",
                    amplitude=0.8,
                    delay=0.05,
                    rise_time=0.1,
                    flat_time=0.2,
                    fall_time=0.3,
                ),
                "adc": SimpleNamespace(
                    num_samples=1000,
                    dwell=0.0001,
                    delay=0.02,
                    freq_offset=50.0,
                    phase_offset=0.3,
                ),
                "trig": SimpleNamespace(channel="ext1", delay=0.01, duration=0.05),
            },
            {
                "ID": 2,
                "block_duration": 1.0,
                "rf": None,
                "gx": None,
                "gy": None,
                "gz": None,
                "adc": None,
                "trig": None,
            },
        ],
        segments=[{"segment_id": 1, "n_blocks_in_segment": 2, "block_ids": [1, 2]}],
        n_columns_in_loop_array=10,
        loop=np.asarray(
            [
                [1, 1, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.5],
                [2, 1, 0.6, 0.1, 0.1, 0.6, 0.6, 0.6, 0.1, 1.6],
            ]
        ),
        max_b1=1.0,
        duration=10.0,
    )
    obj = Ceq.from_struct(data)
    assert obj.n_max == 10
    assert obj.n_parent_blocks == 2
    assert obj.n_segments == 1
    assert len(obj.parent_blocks) == 2
    assert len(obj.segments) == 1
    assert obj.n_columns_in_loop_array == 10
    assert np.array_equal(
        obj.loop,
        np.array(
            [
                [1, 1, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.5],
                [2, 1, 0.6, 0.1, 0.1, 0.6, 0.6, 0.6, 0.1, 1.6],
            ],
            dtype=np.float32,
        ),
    )
    assert obj.max_b1 == 1.0
    assert obj.duration == 10.0

    # Test serialization
    serialized = obj.to_bytes()
    expected = (
        struct.pack(">i", 10)
        + struct.pack(">h", 2)
        + struct.pack(">h", 1)
        + b"".join([block.to_bytes() for block in obj.parent_blocks])
        + b"".join([segment.to_bytes() for segment in obj.segments])
        + struct.pack(">h", 10)
        + obj.loop.astype(">f4").tobytes()
        + struct.pack(">f", 1.0)
        + struct.pack(">f", 10.0)
    )
    assert serialized == expected
