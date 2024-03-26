"""Waveform interpolation subroutines."""

import warnings

import numpy as np
import pypulseq as pp


def gradinterp(g, sysGE, seqGradRasterTime=10e-6, outRasterTime=4e-6):
    """
    Interpolate gradient waveforms and convert to Gauss/cm.

    Parameters
    ----------
    g : SimpleNamespace
        Pulseq gradient event.
    sysGE : _toppe.SystemSpecs
        GE hardware settings, see toppe.systemspecs().
    seqGradRasterTime : float, optional
        Sequence gradient raster time in [s], by default 10e-6.
    outRasterTime : float, optional
        Output raster time in [s], by default 4e-6.

    Returns
    -------
    numpy.ndarray
        Gradient waveform interpolated to 4us raster time, Gauss/cm

    """
    # Parse input options
    gamma = sysGE.gamma  # Hz/T
    raster = sysGE.raster * 1e-6  # sec

    if g is None:
        return np.array([])

    if g.type == "grad":
        # Arbitrary gradient
        tt_rast = g.tt / seqGradRasterTime + 0.5
        if np.all(np.abs(tt_rast - np.arange(1, len(tt_rast) + 1)) < 1e-6):
            # Samples assumed to be on center of raster intervals
            # Arbitrary gradient on a regular raster
            g.first = 0.0
            g.last = 0.0
            areaIn = np.sum(g.waveform) * seqGradRasterTime
            wavtmp = np.concatenate(([g.first], g.waveform, [g.last]))
            tttmp = g.delay + np.concatenate(
                ([0], g.tt, [g.tt[-1] + seqGradRasterTime / 2])
            )
        else:
            # Extended trapezoid: shape specified on "corners" of waveform
            areaIn = np.sum((g.waveform[:-1] + g.waveform[1:]) / 2 * np.diff(g.tt))
            wavtmp = g.waveform
            tttmp = g.delay + g.tt
    else:
        # Convert trapezoid to arbitrary gradient
        areaIn = (g.rise_time / 2 + g.flat_time + g.fall_time / 2) * g.amplitude
        tttmp, wavtmp = trap2arb(g)

    # Add delay
    if g.delay > 0:
        wavtmp = np.concatenate(([0], wavtmp))
        tttmp = np.concatenate(([0], tttmp))

    # Interpolate (includes the delay)
    tt = np.arange(raster / 2, tttmp[-1], raster)
    tmp = np.interp(tt, tttmp, wavtmp)

    areaOut = np.sum(tmp) * raster

    if np.any(np.isnan(tmp)):
        raise ValueError("NaN in gradient trapezoid waveform after interpolation")

    # If areas don't match to better than 0.01%, throw warning
    if np.abs(areaIn) > 1e-6:
        if np.abs(areaIn - areaOut) / np.abs(areaIn) > 1e-4:
            warning_msg = (
                "Gradient area not preserved after interpolating to GE raster time"
            )
            warning_msg += f" (in: {areaIn:.3f} 1/m, out: {areaOut:.3f})"
            warning_msg += "Did you wrap all gradient events in trap4ge()?"
            warnings.warn(warning_msg, category=UserWarning)

    # Convert to Gauss/cm
    wav = tmp / gamma * 100  # Gauss/cm
    return wav


def trap2arb(g):
    """
    Convert trapezoid to arbitrary gradient samples at corner points.
    First sample time is g.delay.

    Parameters
    ----------
    g : SimpleNamespace
        Pulseq gradient event.

    Returns
    -------
    tt : numpy.ndarray
        Sample time points ('corner' points), with tt[0] = g.delay.
    wav : numpy.ndarray
        Gradient samples at tt.

    """
    if g.flat_time > 0:
        wav = np.array([0, 1, 1, 0]) * g.amplitude
        tt = np.array(
            [
                g.delay,
                g.delay + g.rise_time,
                g.delay + g.rise_time + g.flat_time,
                g.delay + g.rise_time + g.flat_time + g.fall_time,
            ]
        )
    else:
        wav = np.array([0, 1, 0]) * g.amplitude
        tt = np.array(
            [g.delay, g.delay + g.rise_time, g.delay + g.rise_time + g.fall_time]
        )

    return tt, wav


def trap4ge(gin, commonRasterTime, sys):
    """
    Extend trapezoid rise/flat/fall times to be on commonRasterTime boundary.
    This ensures that sample points are on both 10us and 4us boundary, so that the interpolation
    from Siemens (10us) to GE (4us) raster time is accurate.

    Parameters
    ----------
    gin : dict
        Pulseq trapezoid gradient struct, created with mr.makeTrapezoid.
    commonRasterTime : float
        Should be 20e-6 s (common divisor of 10us and 4us).
    sys : dict
        Pulseq system struct.

    Returns
    -------
    dict
        Extended trapezoid gradient struct.

    """
    gout = pp.make_trapezoid(
        gin.channel,
        system=sys,
        amplitude=gin.amplitude,
        rise_time=np.ceil(gin.rise_time / commonRasterTime) * commonRasterTime,
        flat_time=np.ceil(gin.flat_time / commonRasterTime) * commonRasterTime,
        fall_time=np.ceil(gin.fall_time / commonRasterTime) * commonRasterTime,
    )

    # Scale to preserve area
    if abs(gin.area) > 1e-6:
        gout.amplitude = gout.amplitude * gin.area / gout.area

    gout.area = gin.area
    return gout
