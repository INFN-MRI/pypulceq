"""Waveform interpolation subroutines."""

__all__ = ["rf2mod", "rf2ge", "grad2mod", "grad2ge", "adc2mod", "adc2ge", "trig2ge"]

import warnings

import numpy as np
import pypulseq as pp


def rf2mod(p, block, sys):
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
    block : pp.Sequence
        Pypulseq sequence block with updated (interpolated)
        RF waveforms.

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
    t = arange(0.5 * sys.raster * 1e-6, block.rf.shape_dur, sys.raster * 1e-6)

    # if time grid is different from native, interpolate
    if len(t) == len(block.rf.t) and np.allclose(t, block.rf.t):
        block.rf.signal = block.rf.amplitude * block.rf.signal
        return block

    # update block and timing
    signal = np.interp(
        t, block.rf.t, block.rf.amplitude * block.rf.signal, left=0, right=0
    )
    block.rf.signal = signal
    block.rf.t = t
    block.rf.shape_dur = t[-1]

    return block


def grad2mod(p, grad, sys, dt):
    """
    Interpolate Gradient event to GE format.

    Parameters
    ----------
    p : int
        Parent block index.
    grad : SimpleNamespace
        Pypulseq structure containing grad event.
    sys : SimpleNamespace
        Structure containing GE hardware specification.
    dt : float
        Pypulseq Opts raster time [s].

    Returns
    -------
    wav : np.ndarray
        Interpolated Gradient waveform.
    tt : np.ndarray
        Interpolated timepoints for Gradient waveform.

    """
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


def grad2ge(p, grad, sys, dt):
    """
    Interpolate Gradient event to GE format.

    Parameters
    ----------
    p : int
        Parent block index.
    grad : SimpleNamespace
        Pypulseq structure containing grad event.
    sys : SimpleNamespace
        Structure containing GE hardware specification.
    dt : float
        Pulseq Opts raster time [s].

    Returns
    -------
    grad : SimpleNamespace
        Pypulseq structure containing updated (interpolated) grad event.

    """
    # check if grad is trapezoid
    if grad.type == "trap":
        if sys.raster * 1e-6 != dt:
            grad = _trap2ge(grad, sys.raster * 1e-6)
        return grad

    # arbitrary or extended trap
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

    if grad.delay > 0:
        wavtmp = np.concatenate(([0], wavtmp))
        tttmp = np.concatenate(([0], tttmp))

    tt = arange(0.5 * sys.raster * 1e-6, tttmp[-1], sys.raster * 1e-6)
    waveform = np.interp(tt, tttmp, wavtmp, left=0, right=0)

    # check area
    area_out = np.sum(waveform) * sys.raster * 1e-6
    if np.any(np.isnan(waveform)):
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

    # update
    grad.waveform = waveform
    grad.tt = tt

    return grad


def adc2mod(p, block, sys):
    """
    Convert ADC event to GE format.

    Parameters
    ----------
    p : int
        Parent block index.
    block : pp.Sequence
        Pypulseq sequence block containing ADC event.
    sys : SimpleNamespace
        Structure containing GE hardware specification.

    Returns
    -------
    rfres : int
        Actual ADC length (excluding dummy points representing delay).
    npre : np.ndarray
        Number of dummy points representing delay.


    """
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


def adc2ge(p, block, sys):
    """
    Convert ADC event to GE format.

    Parameters
    ----------
    p : int
        Parent block index.
    block : pp.Sequence
        Pypulseq sequence block containing ADC event.
    sys : SimpleNamespace
        Structure containing GE hardware specification.

    Returns
    -------
    block : pp.Sequence
        Pypulseq sequence block containing ADC event.

    """
    num_samples, npre = adc2mod(p, block, sys)
    bout = pp.make_adc(
        num_samples,
        dwell=sys.raster * 1e-6,
        delay=npre * sys.raster * 1e-6,
        freq_offset=block.adc.freq_offset,
        phase_offset=block.adc.phase_offset,
    )

    return bout


def trig2ge(p, block, sys):
    """
    Convert Trigger event to GE format.

    Parameters
    ----------
    p : int
        Parent block index.
    block : pp.Sequence
        Pypulseq sequence block containing Trigger event.
    sys : SimpleNamespace
        Structure containing GE hardware specification.

    Returns
    -------
    block : pp.Sequence
        Pypulseq sequence block containing Trigger event.

    """
    dt = sys.raster * 1e-6
    trigout = pp.make_digital_output_pulse(
        channel=block.trig.channel,
        delay=np.ceil(block.trig.delay / dt) * dt,
        duration=np.ceil(block.trig.duration / dt) * dt,
    )

    return trigout


# %% local subroutines
def _trap2ge(gin, dt):
    gout = pp.make_trapezoid(
        gin.channel,
        area=gin.area,
        flat_area=gin.flat_area,
        rise_time=np.ceil(gin.rise_time / dt) * dt,
        flat_time=np.ceil(gin.flat_time / dt) * dt,
        delay=np.ceil(gin.delay / dt) * dt,
    )

    return gout


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
