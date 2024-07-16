"""Waveform interpolation subroutines."""

__all__ = ["rf2ge", "grad2ge", "adc2ge"]

import warnings

import numpy as np


def rf2ge(p, block, sys):
    """
    Interpolate RF waveform to GE format.

    Parameters
    ----------
    p : int
        Parent block index.
    block : pp.Sequence
        Pypulseq sequence block containing RF event.
    sys : SimpleNamespace
        Structure containing GE hardware specification.

    Returns
    -------
    rf : np.ndarray
        Interpolated RF waveform.
    rfres : int
        Actual RF length (excluding dummy points representing delay).

    """
    # check timing
    if block.rf.delay + np.finfo(float).eps < sys.rfDeadTime * 1e-6:
        raise ValueError(f"Parent block {p}: RF delay must be >= sysGE.rfDeadTime")
    if (
        block.rf.delay + block.rf.shape_dur + sys.rfRingdownTime * 1e-6
        > block.block_duration + np.finfo(float).eps
    ):
        raise ValueError(f"Parent block {p}: RF ringdown extends past end of block")

    # calculate time grid of rf pulse
    tge = arange(0.5 * sys.raster * 1e-6, block.rf.shape_dur, sys.raster * 1e-6)

    # if time grid is different from native, interpolate
    if len(tge) == len(block.rf.t) and np.allclose(tge, block.rf.t):
        rf = block.rf.amplitude * block.rf.signal / sys.gamma * 1e4  # Gauss
    else:
        rf = (
            np.interp(
                tge, block.rf.t, block.rf.amplitude * block.rf.signal, left=0, right=0
            )
            / sys.gamma
            * 1e4
        )  # Gauss

    # append dummy points for delay
    npre = round(block.rf.delay / (sys.raster * 1e-6))
    rfres = len(rf)
    rf = np.concatenate([np.zeros(npre), rf])
    return rf, rfres, npre


def grad2ge(p, grad, sys, dt):
    # check if grad is arbitrary or trapezoid
    if grad.type == "grad":  # arbitrary or extended trap
        tt_rast = grad.tt / dt + 0.5
        if np.all(np.abs(tt_rast - np.arange(1, len(tt_rast) + 1)) < 1e-6):
            grad.first = 0
            grad.last = 0
            area_in = np.sum(grad.waveform) * dt
            wavtmp = np.concatenate(([grad.first], grad.waveform, [grad.last]))
            tttmp = grad.delay + np.concatenate(([0], grad.tt, [grad.tt[-1] + dt / 2]))
        else:
            area_in = np.sum(
                (grad.waveform[:-1] + grad.waveform[1:]) / 2 * np.diff(grad.tt)
            )
            wavtmp = grad.waveform
            tttmp = grad.delay + grad.tt
    else:  # trap
        area_in = (
            grad.rise_time / 2 + grad.flat_time + grad.fall_time / 2
        ) * grad.amplitude
        tttmp, wavtmp = _trap2arb(grad)

    if grad.delay > 0:
        wavtmp = np.concatenate(([0], wavtmp))
        tttmp = np.concatenate(([0], tttmp))

    tt = arange(0.5 * sys.raster * 1e-6, tttmp[-1], sys.raster * 1e-6)
    tmp = np.interp(tt, tttmp, wavtmp, left=0, right=0)

    area_out = np.sum(tmp) * sys.raster * 1e-6

    if np.any(np.isnan(tmp)):
        raise ValueError(
            f"NaN in gradient trapezoid waveform after interpolation (parent block {p})"
        )

    if abs(area_in) > 1e-6:
        if abs(area_in - area_out) / abs(area_in) > 1e-4:
            warning_msg = (
                f"Gradient area for parent block {p} were not preserved after interpolating to GE raster time (in: {area_in:.3f} 1/m, out: {area_out:.3f}). "
                "Did you wrap all gradient events in trap4ge?"
            )
            warnings.warn(f"{warning_msg}")

    wav = tmp / sys.gamma * 100  # Gauss/cm

    return wav, tt


def adc2ge(p, block, sys):
    if block.adc.delay + np.finfo(float).eps < sys.adcDeadTime * 1e-6:
        print(f"Warning: Parent block {p}: ADC delay is < sys.adcDeadTime")
    if (
        block.adc.delay + block.adc.num_samples * block.adc.dwell
        > block.block_duration + np.finfo(float).eps
    ):
        raise ValueError(f"Parent block {p}: ADC window extends past end of block")
    npre = round(block.adc.delay / (sys.raster * 1e-6))
    rfres = round(block.adc.num_samples * block.adc.dwell / (sys.raster * 1e-6))

    return rfres, npre


# %% local subroutines
def _trap2arb(grad):
    if grad.flat_time > 0:
        wav = np.asarray([0, 1, 1, 0]) * grad.amplitude
        tt = grad.delay + np.asarray(
            [
                0,
                grad.rise_time,
                grad.rise_time + grad.flat_time,
                grad.rise_time + grad.flat_time + grad.fall_time,
            ]
        )
    else:
        wav = np.asarray([0, 1, 0]) * grad.amplitude
        tt = grad.delay + np.asarray(
            [0, grad.rise_time, grad.rise_time + grad.fall_time]
        )

    return tt, wav


def arange(start, stop, step=1):
    """
    Demonstrate how the built-in a:d:b is constructed.

    Parameters:
    start : float
        Start of the range.
    step : float, optional
        Step size. If not provided, default is 1.
    stop : float, optional
        End of the range. If not provided, the function works as if step is stop and step is 1.

    Returns:
    numpy.ndarray
        Generated range of values.
    """
    if stop is None:
        stop = step
        step = 1

    tol = 2.0 * np.finfo(float).eps * max(abs(start), abs(stop))
    sig = np.sign(step)

    # Exceptional cases
    if not np.isfinite(start) or not np.isfinite(step) or not np.isfinite(stop):
        return np.array([np.nan])
    elif step == 0 or (start < stop and step < 0) or (stop < start and step > 0):
        # Result is empty
        return np.zeros(0)

    # n = number of intervals = length(v) - 1
    if start == np.floor(start) and step == 1:
        # Consecutive integers
        n = int(np.floor(stop) - start)
    elif start == np.floor(start) and step == np.floor(step):
        # Integers with spacing > 1
        q = np.floor(start / step)
        r = start - q * step
        n = int(np.floor((stop - r) / step) - q)
    else:
        # General case
        n = round((stop - start) / step)
        if sig * (start + n * step - stop) > tol:
            n -= 1

    # last = right hand end point
    last = start + n * step
    if sig * (last - stop) > -tol:
        last = stop

    # out should be symmetric about the mid-point
    out = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1)
    out[k] = start + k * step
    out[n - k] = last - k * step
    if n % 2 == 0:
        out[n // 2 + 1] = (start + last) / 2

    return out
