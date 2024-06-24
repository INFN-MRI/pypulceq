"""Cartesian 3D GRE example."""

__all__ = ["design_gre"]

import math

import numpy as np
from tqdm import tqdm

import pypulseq as pp


def design_gre(
    fov=(256, 180),
    mtx=(256, 150),
    write_seq: bool = False,
    seq_filename: str = "cart_pypulseq.seq",
):
    """
    Design 3D GRE with Cartesian k-space encoding.

    Parameters
    ----------
    fov : tuple, optional
        Acquisition field of view specified as ``(in-plane, slab)``.
        The default is ``(256, 180)``.
    mtx : tuple, optional
        Image grid specified as ``(nx=ny, nz)``.
        The default is ``(256, 150)``.
    write_seq : bool, optional
        Save sequence to disk as ``.seq``.
        The default is ``False``.
    seq_filename : str, optional
        Sequence filename.
        The default is ``"cart_pypulseq.seq"``.

    Returns
    -------
    seq : pp.Sequence
        Pulseq Sequence structure describing the acqusition.

    """
    # ======
    # SETUP
    # ======
    # Create a new sequence object
    seq = pp.Sequence()
    fov, slab_thickness = fov[0] * 1e-3, fov[1] * 1e-3  # in-plane FOV, slab thickness
    Nx, Ny, Nz = mtx[0], mtx[0], mtx[1]  # in-plane resolution, slice thickness

    # RF specs
    alpha = 10  # flip angle
    rf_spoiling_inc = 117  # RF spoiling increment

    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    # ======
    # CREATE EVENTS
    # ======
    rf, gss, _ = pp.make_sinc_pulse(
        flip_angle=alpha * math.pi / 180,
        duration=3e-3,
        slice_thickness=slab_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=True,
    )
    gss_reph = pp.make_trapezoid(
        channel="z", area=-gss.area / 2, duration=1e-3, system=system
    )

    # Define other gradients and ADC events
    delta_kx, delta_ky, delta_kz = 1 / fov, 1 / fov, 1 / slab_thickness
    gread = pp.make_trapezoid(
        channel="x", flat_area=Nx * delta_kx, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(
        num_samples=Nx, duration=gread.flat_time, delay=gread.rise_time, system=system
    )
    gxpre = pp.make_trapezoid(
        channel="x", area=-gread.area / 2, duration=1e-3, system=system
    )
    gxrew = pp.scale_grad(grad=gxpre, scale=-1)
    gxrew.id = seq.register_grad_event(gxpre)

    gyphase = pp.make_trapezoid(channel="y", area=-delta_ky * Ny, system=system)
    gzphase = pp.make_trapezoid(channel="z", area=-delta_kz * Nz, system=system)

    # Phase encoding plan and rotation
    pey_steps = ((np.arange(Ny)) - Ny / 2) / Ny * 2
    pez_steps = ((np.arange(Nz)) - Nz / 2) / Nz * 2

    # Gradient spoiling
    gz_spoil = pp.make_trapezoid(channel="z", area=4 / slab_thickness, system=system)

    # Initialize RF phase and increment
    rf_phase = 0
    rf_inc = 0

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Loop over phase encodes and define sequence blocks
    for z in tqdm(range(-1, Nz)):
        # Pre-register PE events that repeat in the inner loop
        gzpre = pp.scale_grad(grad=gzphase, scale=pez_steps[z])
        gzpre.id = seq.register_grad_event(gzpre)
        gzrew = pp.scale_grad(grad=gzphase, scale=-pez_steps[z])
        gzrew.id = seq.register_grad_event(gzrew)

        for y in range(Ny):
            # Compute PE events
            if z < 0:  # dummy for pre-scane and steady state prep
                gypre = pp.scale_grad(grad=gyphase, scale=0.0)
                gyrew = pp.scale_grad(grad=gyphase, scale=0.0)
            else:
                gypre = pp.scale_grad(grad=gyphase, scale=pey_steps[y])
                gyrew = pp.scale_grad(grad=gyphase, scale=-pey_steps[y])

            # Compute RF and ADC phase for spoiling and signal demodulation
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi

            # Add slab-selective excitation
            seq.add_block(rf, gss)

            # Slab refocusing gradient
            seq.add_block(gss_reph)

            # Read-prewinding and phase encoding gradients
            seq.add_block(gxpre, gypre, gzpre)

            # Add readout
            if z < 0:
                seq.add_block(gread)
            else:
                seq.add_block(gread, adc)

            # Rewind
            seq.add_block(gxrew, gyrew, gzrew)

            # Spoil
            seq.add_block(gz_spoil)

            # Update RF phase and increment
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    # Check whether the timing of the sequence is correct
    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare the sequence output for the scanner
        seq.set_definition(key="FOV", value=[fov, fov, slab_thickness / Nz])
        seq.set_definition(key="Name", value="cart_gre")

        seq.write(seq_filename)
    else:
        seq = seq.remove_duplicates()

    return seq
