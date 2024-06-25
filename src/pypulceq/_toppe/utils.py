"""TOPPE utils"""

__all__ = ["padwaveforms", "checkwaveforms", "plotseq"]

import warnings

import numpy as np

# import numba as nb

from ._read import _readmodulelistfile
from ._read import _readloop
from ._read import _readcoresfile


def padwaveforms(rf=(), gx=(), gy=(), gz=()):
    """
    Pad input waveform(s) with zero

    Return arrays rf / gx / gy / gz of common size [ndat npulses], where ndat is multiple of 4.

    Parameters
    ----------
    rf : np.ndarray, optional
        Complex RF waveform of shape ``(nrf, nrfpulses)``.
        The default is ``()``.
    gx : np.ndarray, optional
        X-axis gradient waveform of shape ``(ngx, ngxpulses)``.
        The default is ``()``.
    gy : np.ndarray, optional
        Y-axis gradient waveform of shape ``(ngy, ngypulses)``.
        The default is ``()``.
    gz : np.ndarray, optional
        Z-axis gradient waveform of shape ``(ngz, ngzpulses)``.
        The default is ``()``.

    Returns
    -------
    rf : np.ndarray, optional
        Complex RF waveform of shape ``(ndat, nrfpulses)``, where ``ndat = max waveform length``,
        and npulses = max number of waveforms (pulses) per axis.
    gx : np.ndarray, optional
        X-axis gradient waveform of shape ``(ndat, ngxpulses)``.
    gy : np.ndarray, optional
        Y-axis gradient waveform of shape ``(ndat, ngypulses)``.
    gz : np.ndarray, optional
        Z-axis gradient waveform of shape ``(ndat, ngzpulses)``.

    """
    # Convert inputs to 2D numpy arrays
    rf, gx, gy, gz = _to_2d_array(rf, gx, gy, gz)

    # Detect all-zero RF waveform, treat as empty, and give warning to user
    if rf.shape[0] != 0:
        if int((rf != 0).sum()) == 0:
            warnings.warn(
                "(First) RF waveform contains all zeros -- ignored", UserWarning
            )

    # Force all waveform arrays to have the same dimensions,
    # and make length multiple of 4.
    ndat = max([rf.shape[0], gx.shape[0], gy.shape[0], gz.shape[0]])
    npulses = max([rf.shape[1], gx.shape[1], gy.shape[1], gz.shape[1]])

    # Check validity of input
    assert ndat != 0, "At least one waveform must be non-empty"
    assert (
        ndat <= 2**15
    ), f"Max waveform length is 32768 (samples) -- found {ndat} samples"

    # make length divisible by 4 (EPIC seems to behave best this way)
    # if ndat % 4 != 0:
    #     # warnings.warn('Waveform duration will be padded to 4 sample boundary.', UserWarning)
    #     ndat = ndat - (ndat % 4) + 4

    # keep minimal changes compared to MATLAB toppe:
    fields = {"rf": rf, "gx": gx, "gy": gy, "gz": gz}
    for wavtype in fields.keys():  # 'rf', 'gx', 'gy', or 'gz'
        wav = fields[wavtype].copy()  # [ndat npulses]

        if wavtype == "rf" and sum(abs(wav.flatten())) == 0:
            # Must have non-zero RF waveform to make toppe happy (even if it's not actually played out)
            wav = 0.01 * np.ones((ndat, npulses), dtype=np.float32)
            wav[0] = 0.0
            wav[-1] = 0.0

        # enforce equal number of rows and columns
        nrows, n2 = wav.shape

        # if nrows !=0 and nrows < ndat:
        #     warnings.warn(f'Padding {wavtype} with zero rows', UserWarning)

        # if n2 != 0 and n2 < npulses:
        #     warnings.warn('Padding {wavtype} with zero columns', UserWarning)

        # actual pad
        wav = np.pad(wav, ((0, ndat - nrows), (0, npulses - n2)))

        # copy to corresponding wav type (rf, gx, gy, or gz)
        fields[wavtype] = wav.copy()

    # unpack
    rf, gx, gy, gz = fields["rf"], fields["gx"], fields["gy"], fields["gz"]

    return rf, gx, gy, gz


def checkwaveforms(
    system, rf=(), gx=(), gy=(), gz=(), rfUnit="G", gradUnit="G/cm", slewUnit="G/cm/msec"
):
    """
    Check rf/gradient waveforms against system limits.

    Parameters
    ----------
    system : SystemSpecs
        Struct containing hardware specs. See systemspecs.py.
    rf : np.ndarray, optional
        Complex rf waveform of shape ``(nrf, nrfpulses)``. The default is ``()``.
    gx : np.ndarray, optional
        X-axis gradient waveform of shape ``(ngx, ngxpulses)``.
        The default is ``()``.
    gy : np.ndarray, optional
        Y-axis gradient waveform of shape ``(ngy, ngypulses)``.
        The default is ``()``.
    gz : np.ndarray, optional
        Z-axis gradient waveform of shape ``(ngz, ngzpulses)``.
        The default is ``()``.
    rfUnit : str, optional
        ``Gauss`` or ``uT``. The default is ``G``.
    gradUnit : str, optional
        ``Gauss/cm`` or ``mT/m``. The default is ``G/cm``.
    slewUnit : str, optional
        ``Gauss/cm/ms`` or ``T/m/s``. The default is ``G/cm/msec``.

    Returns
    -------
    isValid : bool
        ``True`` / ``False``.
    gmax : tuple
        Max gradient amplitude on the three gradient axes ``(x, y, z)`` in ``[G/cm]``.
    slewmax : tuple
        Max slew rate on the three gradient axes ``(x, y, z)`` in ``[G/cm/ms]``.

    """
    # Convert input waveforms and system limits to Gauss and Gauss/cm
    if rfUnit == "uT" and len(rf) > 0:
        rf /= 100.0  # Gauss

    if gradUnit == "mT/m" and len(gx) > 0:
        gx /= 10.0  # Gauss/
    if gradUnit == "mT/m" and len(gy) > 0:
        gy /= 10.0
    if gradUnit == "mT/m" and len(gz) > 0:
        gz /= 10.0

    if system.rfUnit == "uT":
        maxRF = system.maxRF / 100.0  # Gauss
    else:
        maxRF = system.maxRF

    if system.gradUnit == "mT/m":
        maxGrad = system.maxGrad / 10.0  # Gauss/cm
    else:
        maxGrad = system.maxGrad

    if system.slewUnit == "T/m/s":
        maxSlew = system.maxSlew / 10.0  # Gauss/cm/msec
    else:
        maxSlew = system.maxSlew

    # Check against system hardware limits
    axes = ["x", "y", "z"]

    # Gradient amplitude and slew
    g = {"x": gx, "y": gy, "z": gz}
    gmax = []
    slewmax = []
    for ax in axes:
        gmax_tmp = max(abs(g[ax].flatten()))
        if gmax_tmp > maxGrad:
            print(
                f"Error: {ax} gradient amplitude exceeds system limit {gmax_tmp / maxGrad * 100}\n"
            )
            isValid = False
        gmax.append(gmax_tmp)

        smax_tmp = abs(np.diff(g[ax], axis=0) / (system.raster * 1e3)).max(axis=0).max()
        if smax_tmp > maxSlew:
            print(
                f"Error: {ax} gradient slew rate exceeds system limit {smax_tmp / maxSlew * 100}\n"
            )
            isValid = False
        slewmax.append(smax_tmp)

    # Peak rf
    if len(rf) > 0:
        if max(abs(rf.flatten())) > maxRF:
            print(
                f"Error: rf amplitude exceeds system limit {max(abs(rf)) / maxRF * 100}\n"
            )
            isValid = False
        else:
            isValid = True
    else:
        isValid = True

    # Check PNS. Warnings if > 80% of threshold.
    # for jj in range(gx.shape[1]): # loop through all waveforms (pulses)
    #     gtm = []
    #     gtm.append(gx[:, jj].transpose() * 1e-2) # T/m
    #     gtm.append(gy[:, jj].transpose() * 1e-2) # T/m
    #     gtm.append(gz[:, jj].transpose() * 1e-2) # T/m
    #     gtm = np.stack(gtm, axis=0)
    #     pThresh, _, _, _, _, _ = pns(gtm, system.gradient, gdt=system.raster, do_plt=False, do_print=False)
    #     if max(pThresh) > 80:
    #         if max(pThresh) > 100:
    #             warnings.warn(f'PNS ({round(max(pThresh))}%%) exceeds first controlled mode (100%%)!!! (waveform {jj})', UserWarning)
    #         else:
    #             warnings.warn(f'PNS({round(max(pThresh))}%%) exceeds normal mode (80%%)! (waveform {jj})', UserWarning)

    # # do all waveforms start and end at zero?
    # wav = np.concatenate((gx[0], gx[-1], gy[0], gy[-1], gz[0], gz[-1], rf[0].flatten(), rf[-1].flatten()))
    # if any(wav):
    # 	print('Error: all waveforms must begin and end with zero\n')
    # 	isValid = False

    return isValid, gmax, slewmax


def plotseq(
    sysGE,
    timeRange,
    loop=None,
    loopFile="scanloop.txt",
    moduleListFile="modules.txt",
    timeOnly=False,
):
    """
    Trimmed down version of TOPPE plotseq, here used only
    for preflightcheck and getscanduration.

    Parameters
    ----------
    sysGE : SystemSpecs
        Struct containing hardware specs. See systemspecs.py.
    timeRange : Iterable[float]
        Time range of the sequence examination.
    loop : np.ndarray
        çoop file for sequence.
    loopFile : str, optional
        File containing sequence loop. The default is ``scanloop.txt``
    moduleListFile : str, optional
        List of sequence modules. The default is ``modules.txt``.
    timeOnly : bool, optional
        If ``True``, only compute scan duration. The default is ``True``.

    Returns
    -------
    tRange : Iterable[float]
        List of duration of each plotting interval.
    rf : np.ndarray, optional
        Complex rf waveform of shape ``(nrf, nrfpulses)``.
        Only returned if ``timeOnly`` is ``False``.
    gx : np.ndarray, optional
        X-axis gradient waveform of shape ``(ngx, ngxpulses)``.
        Only returned if ``timeOnly`` is ``False``.
    gy : np.ndarray, optional
        Y-axis gradient waveform of shape ``(ngy, ngypulses)``.
        Only returned if ``timeOnly`` is ``False``.
    gz : np.ndarray, optional
        Z-axis gradient waveform of shape ``(ngz, ngzpulses)``.
        Only returned if ``timeOnly`` is ``False``.

    """
    # support the old way of calling plotseq: plotseq(nStart, nStep, sysGE, varargin)
    timeRange = [int(t * 1e6) for t in timeRange]

    # read scan files
    modules = _readmodulelistfile(moduleListFile)

    # segment definitions (here referred to as 'module groups' in accordance with tv6.e)
    modGroups = _readcoresfile("cores.txt")
    if loop is None:
        loop, _ = _readloop(loopFile)

    # Add end of segment label to loop so we know when to insert
    # the segmentRingdownTime (= 116 us)
    n = 0
    isLastBlockInSegment = np.zeros(loop.shape[0])
    while n < loop.shape[0]:
        i = loop[n, -1] - 1
        isLastBlockInSegment[n + len(modGroups[i]) - 1] = 1
        n = n + len(modGroups[i])

    # Build sequence
    rho = np.array([])
    th = np.array([])
    gx = np.array([])
    gy = np.array([])
    gz = np.array([])
    raster = sysGE.raster

    # Calculation
    tic = timeRange[0]
    toc = timeRange[1]
    n = 0
    t = 0
    dur = []
    dur0 = []
    while t <= tic:
        if n == loop.shape[0] - 1:
            raise ValueError("Invalid start time")
        p = loop[n, 0] - 1
        if p == -1:
            dur0 = loop[n, 13]
        else:
            minimumModuleDuration = modules[p]["res"] * raster
            dur0 = max(modules[p]["dur"], minimumModuleDuration)
        if isLastBlockInSegment[n]:
            dur0 += sysGE.segmentRingdownTime
        t += dur0
        n += 1

    if not dur0:
        raise ValueError("Invalid time range")
    else:
        dur.append(dur0)

    tStart = max(t - dur0, 0)
    nStart = max(n - 2, 0)

    while t < toc and n < loop.shape[0]:
        p = loop[n, 0] - 1
        if p == -1:
            dur0 = loop[n, 13]
        else:
            minimumModuleDuration = modules[p]["res"] * raster
            dur0 = max(modules[p]["dur"], minimumModuleDuration)
        if isLastBlockInSegment[n]:
            dur0 += sysGE.segmentRingdownTime
        dur.append(dur0)
        t += dur0
        n += 1

    # cut last
    dur = np.asarray(dur)[:-1]

    tStop = t
    nStop = max(n - 2, 0)

    # get time
    tRange = [tStart * 1e-6, tStop * 1e-6]

    # get waveforms
    if timeOnly:
        return tRange
    else:
        rho, th, gx, gy, gz = _sub_getwavs(nStart, nStop, loop, modules, sysGE, dur)
        rf = rho * np.exp(1j * th)

    return rf, gx, gy, gz, tRange


# %% subfunc
def _to_2d_array(*args):
    out = []
    for arg in args:
        arr = np.asarray(arg)
        if len(arr.shape) == 1:
            arr = arr[:, None]
        out.append(arr)
    return out


def _sub_getwavs(blockStart, blockStop, loop, modules, sysGE, dur):
    # constants
    max_pg_iamp = 2**15 - 2
    raster = sysGE.raster

    # module index
    p = loop[blockStart:blockStop, 0] - 1

    # waveform index
    w = loop[blockStart:blockStop, 15]

    # get scaling and rotations
    ia_rf = loop[blockStart:blockStop, 1] / max_pg_iamp
    dtheta = np.pi * loop[blockStart:blockStop, 11] / max_pg_iamp

    ia_gx = loop[blockStart:blockStop, 3] / max_pg_iamp
    ia_gy = loop[blockStart:blockStop, 4] / max_pg_iamp
    ia_gz = loop[blockStart:blockStop, 5] / max_pg_iamp

    Rv = loop[blockStart:blockStop, 16:25].reshape(-1, 3, 3) / max_pg_iamp

    # remove pure delay blocks
    idx = np.where(p != -1)
    dur = dur[idx]
    p = p[idx]
    w = w[idx] - 1

    ia_rf = ia_rf[idx]
    dtheta = dtheta[idx]

    ia_gx = ia_gx[idx]
    ia_gy = ia_gy[idx]
    ia_gz = ia_gz[idx]

    Rv = Rv[idx]  # (blocksize, 3, 3)

    # # get rf flag
    hasRF = np.asarray([modules[idx]["hasRF"] for idx in p], dtype=bool)
    ia_rf[np.logical_not(hasRF)] = 0.0

    # get waveforms
    irho = [np.abs(modules[idx]["rf"]) for idx in p]
    itheta = [np.angle(modules[idx]["rf"]) for idx in p]
    igx = [modules[idx]["gx"] for idx in p]
    igy = [modules[idx]["gy"] for idx in p]
    igz = [modules[idx]["gz"] for idx in p]

    # calculate actual length
    res = (dur / raster).astype(int)

    # loop over modules
    rho, theta, gx, gy, gz = _sub_getwavs_nb(
        irho, itheta, igx, igy, igz, ia_rf, dtheta, ia_gx, ia_gy, ia_gz, Rv, res, w
    )

    # flatten and return
    rho = np.concatenate(rho)
    theta = np.concatenate(theta)
    gx = np.concatenate(gx)
    gy = np.concatenate(gy)
    gz = np.concatenate(gz)

    return rho, theta, gx, gy, gz


# @nb.njit(cache=True)
def _sub_getwavs_nb(
    irho, itheta, igx, igy, igz, ia_rf, dtheta, ia_gx, ia_gy, ia_gz, Rv, res, w
):
    # preallocate
    rho = []
    theta = []
    gx = []
    gy = []
    gz = []

    for n in range(len(irho)):
        # current iteration
        rhoit = irho[n]
        thetait = itheta[n]
        gxit = igx[n]
        gyit = igy[n]
        gzit = igz[n]

        # select correct waveform
        if len(rhoit.shape) == 2:
            rhoit = rhoit[:, w[n]]
            thetait = thetait[:, w[n]]
            gxit = gxit[:, w[n]]
            gyit = gyit[:, w[n]]
            gzit = gzit[:, w[n]]

        # scale
        rhoit = ia_rf[n] * rhoit
        thetait = ia_rf[n] * thetait + dtheta[n]

        gxit = ia_gx[n] * gxit
        gyit = ia_gy[n] * gyit
        gzit = ia_gz[n] * gzit

        # rotate
        g = np.dot(Rv[n], np.stack((gxit, gyit, gzit), axis=0))
        gxit = g[0]
        gyit = g[1]
        gzit = g[2]

        # pad
        # rhoit = np.pad(rhoit, (0, res[n] - rhoit.shape[0]))
        # thetait = np.pad(thetait, (0, res[n] - thetait.shape[0]))
        # gxit = np.pad(gxit, (0, res[n] - gxit.shape[0]))
        # gyit = np.pad(gyit, (0, res[n] - gyit.shape[0]))
        # gzit = np.pad(gzit, (0, res[n] - gzit.shape[0]))

        # append to output
        rho.append(rhoit)
        theta.append(thetait)
        gx.append(gxit)
        gy.append(gyit)
        gz.append(gzit)

    return rho, theta, gx, gy, gz
