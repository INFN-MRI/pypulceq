"""Porting of TOPPE writemod.m"""

__all__ = ["writemod", "writemodfile"]

import copy
import inspect
import warnings

from datetime import datetime
from os.path import realpath, sep

import numpy as np

from .utils import checkwaveforms
from .utils import padwaveforms

from ._read import _readmod

eps = np.finfo(np.float32).eps
nChopDefault = (0, 0)


def writemod(
    system,
    rf=(),
    gx=(),
    gy=(),
    gz=(),
    ofname="out.mod",
    desc="TOPPE module",
    nomflip=90.0,
    hdrfloats=(),
    hdrints=(),
    nChop=(0, 0),
    rfUnit="uT",
    gradUnit="mT/m",
    slewUnit="T/m/s",
    **kwargs,
):
    """
    Write waveforms to .mod file, for use with toppe psd on GE scanners.

    Assumes raster time (sample duration) of 4e-6 sec for all waveforms.

    Examples
    -------

    >>> writemod(sys, rf=rho*np.exp(1i*theta), gz=gzwaveform, ofname='tipdown.mod')
    >>> writemod(sys, gz=gzwaveform, desc=my spoiler gradient', ofname='spoiler.mod')

    Parameters
    ----------
    system : SystemSpecs
        Struct specifying hardware system info, see systemspecs.py.
    rf : np.ndarray, optional
        Complex RF waveform of shape ``(nrf, nrfpulses)``.
        The default is ``None``.
    gx : np.ndarray, optional
        X-axis gradient waveform of shape ``(ngx, ngxpulses)``.
        The default is ``None``.
    gy : np.ndarray, optional
        Y-axis gradient waveform of shape ``(ngy, ngypulses)``.
        The default is ``None``.
    gz : np.ndarray, optional
        Z-axis gradient waveform of shape ``(ngz, ngzpulses)``.
        The default is ``None``.
    ofname : str, optional
        Output filename. Defaults to ``'out.mod'``.
    desc : str, optional
        Text string (ASCII) descriptor. Defaults to ``'TOPPE module'``.
    nomflip : float, optional
        Excitation flip angle (degrees). Stored in .mod float header,
        but has no influence on b1 scaling. Defaults to ``90.0``.
    hdrfloats : np.ndarray, optional
        Additional floats to put in header (max 12). Defaults to ``None``.
    hdrints : np.ndarray, optional
        Additional ints to put in header (max 29). Defaults to ``None``.
    nChop : np.ndarray, optional
        ``(2,)`` ``(int, multiple of 4)`` trim (chop) the start and end of
        the RF wavevorm (or ADC window) by this many 4us samples.
        Using non-zero nChop can reduce module duration on scanner.
        If converting to Pulseq using pulsegeq.ge2seq, nChop[1] must
        be non zero to allow time for RF ringdown.
        The default is``(0, 0)``.
    rfUnit : str, optional
        ``Gauss`` or ``uT``. The default is ``uT``.
    gradUnit : str, optional
        ``Gauss/cm`` or ``mT/m``. The default is ``mT/m``.
    slewUnit : str, optional
        ``Gauss/cm/ms`` or ``T/m/s``. The default is ``T/m/s``.

    """
    # nReservedInts = 2  # [nChop(1) rfres], rfres = number of samples in RF/ADC window
    # maxHdrInts = 32 - nReservedInts

    # convert to numpy
    rf, gx, gy, gz = _to_numpy(rf, gx, gy, gz)

    # Check nChop
    if nChop[0] < nChopDefault[0] or nChop[1] < nChopDefault[1]:
        msg = "Module duration (on scanner) will be extended to account for RF/ADC dead/ringdown times as needed."
        warnings.warn(msg, UserWarning)

    # if len(rf) == 0:
    #     msg = f'RF waveform must be zero during the first/last {nChop[0]}/{nChop[1]} (nChop) samples.'
    #     assert int((rf[:nChop[0]+1] != 0).sum()) + int((rf[-1-nChop[1]:] != 0).sum()) == 0, msg

    assert (
        nChop[1] % 4 == 0 and nChop[1] % 4 == 0
    ), "Each entry in nChop must be multiple of 4"

    # Force all waveform arrays to have the same dimensions
    rf, gx, gy, gz = padwaveforms(rf=rf, gx=gx, gy=gy, gz=gz)

    # Check waveforms against system hardware limits
    isValid, _, _ = checkwaveforms(system, rf=rf, gx=gx, gy=gy, gz=gz)
    assert isValid, "Waveforms failed system hardware checks -- exiting"

    # Convert to Gauss and Gauss/cm
    if rfUnit == "mT":
        rf = rf / 100.0  # Gauss

    if gradUnit == "mT/m":
        gx = gx / 10.0  # Gauss/cm
        gy = gy / 10.0
        gz = gz / 10.0

    # convert system
    system = copy.deepcopy(system)
    if system.rfUnit == "uT":
        system.maxRF = system.maxRF / 100.0  # Gauss
        system.rfUnit = "G"

    if system.gradUnit == "mT/m":
        system.maxGrad = system.maxGrad / 10.0  # Gauss/cm
        system.gradUnit = "G/cm"

    if system.slewUnit == "T/m/s":
        system.maxSlew = system.maxSlew / 10.0  # Gauss/cm/msec
        system.slewUnit = "G/cm/msec"

    # header arrays
    paramsfloat = _myrfstat(abs(rf[:, 0]), nomflip, system)
    paramsint16 = [
        nChop[0],
        rf.shape[0] - sum(nChop),
    ]  # these reserved ints are used by interpreter
    paramsint16 = np.asarray(paramsint16, dtype=np.int16)

    # Write to .mod file
    desc = f"Filename: {ofname} \n{desc}"
    _writemod(ofname, desc, rf, gx, gy, gz, paramsint16, paramsfloat, system)


def writemodfile(ceq, modFiles, sysGE, seqGradRasterTime, ignoreTrigger=False):
    """
    Write mod files for each parent block in ceq.

    Parameters
    ----------
    ceq : SimpleNamespace
        A namespace representing the Ceq struct.
    modFiles : list of str
        List to store output mod file names for each parent block.
    sysGE : SystemSpecs
        System parameters for the GE scanner.
    seqGradRasterTime : float, optional
        Gradient raster time in .seq file.
    ignoreTrigger : bool, optional
        Flag to ignore trigger or not. The default is ``False``.

    """
    # Write modules.txt
    # Do this after creating .mod files, so .mod file duration can be set exactly.
    with open("modules.txt", "w") as fid:
        fid.write("Total number of unique cores\n")
        fid.write(f"{ceq.nParentBlocks}\n")
        fid.write("fname dur(us) hasRF hasADC trigpos\n")

        for p in range(ceq.nParentBlocks):
            b = ceq.parentBlocks[p]

            # check if block contains RF, ADC or none (pure delay)
            hasRF = hasattr(b, "rf")
            hasADC = hasattr(b, "adc")

            if hasRF and hasADC:
                raise ValueError("Block cannot contain both RF and ADC events")

            # Determine trigger position
            if hasattr(b, "trig") and not ignoreTrigger:
                trigpos = (
                    np.round(b.trig.delay * 1e6)
                    if b.trig.delay + eps >= 100e-6
                    else 100
                )
            else:
                trigpos = -1  # no trigger

            rf = _readmod(modFiles[p])[0] if hasRF else []
            dur = max(
                len(rf) * seqGradRasterTime * 1e6,
                np.round(
                    np.floor(b.blockDuration / seqGradRasterTime)
                    * seqGradRasterTime
                    * 1e6
                ),
            )

            fid.write(
                f"{modFiles[p]}\t{int(dur)}\t{int(hasRF)}\t{int(hasADC)}\t{int(trigpos)}\n"
            )


# %% subfunc
def _to_numpy(*args):
    """
    Convert inputs to numpy.
    """
    out = []
    for arg in args:
        arg = copy.deepcopy(arg)
        try:
            out.append(arg.numpy())
        except:
            out.append(arg)

    return out


def _myrfstat(b1, nom_fa, system):
    """
    Calculate RF parameters needed for RFPULSE struct in .e file.

    Needed for B1 scaling, SAR calculations, and enforcing duty cycle limits.
    See also mat2signa_krishna.m

    Parameters
    ----------
    b1 : np.ndarray
        Real 1D vector containing B1 amplitude, size ``Nx1`` (units does not matter, automatically converted to ``[G]``).
    nom_fa  : float
        Nominal flip angle in ``[degrees]``.
    sys : SystemSpecs
        Struct specifying hardware system info, see systemspecs.py.

    Returns
    -------
    paramsfloat : np.ndarray
        RF parameters for module header.

    """
    g = 1  # legacy dummy value, ignore
    nom_bw = 2000

    dt = 4e-6  # use 4 us RF raster time
    # gamma = 4.2575e3  # Hz/Gauss
    # tbwdummy = 2

    hardpulse = max(abs(b1)) * np.ones_like(
        b1
    )  # hard pulse of equal duration and amplitude

    pw = len(b1) * dt * 1e3  # ms
    nom_pw = len(b1) * dt * 1e6  # us

    if max(abs(b1)) == 0:  # non-zero RF pulse
        raise ValueError("RF waveform cannot be zero")

    abswidth = sum(abs(b1)) / sum(abs(hardpulse))
    effwidth = sum(b1**2) / sum(hardpulse**2)

    # or equivalently:  effwidth = sum(b1 ** 2) / (max(abs(b1)) ** 2) / len(b1)
    area = abs(sum(b1)) / abs(sum(hardpulse))
    dtycyc = np.sum(np.abs(b1) > 0.2236 * max(abs(b1))) / len(b1)
    maxpw = dtycyc
    num = 1
    max_b1 = (
        system.maxRF
    )  # Gauss. Full instruction amplitude (32766) should produce max_b1 RF amplitude,
    # as long as other RF .mod files (if any) use the same max_b1.

    max_int_b1_sq = np.sum(np.abs(b1) ** 2) * dt * 1e3
    max_rms_b1 = np.sqrt(np.sum(np.abs(b1) ** 2)) / len(b1)
    nom_fa = nom_fa  # degrees

    # calculate equivalent number of standard pulses
    stdpw = 1  # duration of standard pulse (ms)
    # stdpulse = 0.117 * np.ones(int(stdpw // (dt * 1e3)))
    numstdpulses = num * effwidth * (pw / stdpw) * (max(abs(b1)) / 0.117) ** 2

    paramsfloat = [
        pw,
        abswidth,
        effwidth,
        area,
        dtycyc,
        maxpw,
        num,
        max_b1,
        max_int_b1_sq,
        max_rms_b1,
        90.0,
        nom_pw,
        nom_bw,
        g,
        numstdpulses,
        nom_fa,
    ]  # hardcode 'opflip' to 90

    return np.asarray(paramsfloat, dtype=np.float32)


def _writemod(fname, desc, rf, gx, gy, gz, paramsint16, paramsfloat, system):
    """
    Actual writing of waveforms to file.

    Parameters
    ----------
    fname : str
        File name.
    desc : str
        ASCII description.
    rf : np.ndarray
        ``size(rho) = N x 1``, where ``N`` is the number of waveform samples
        and ``Ncoils`` is the number of transmit channel.
    gx : np.ndarray
        ``(Gauss/cm)``. vector of length ``N``.
    gy : np.ndarray
        ``(Gauss/cm)``. vector of length ``N``.
    gz : np.ndarray
        ``(Gauss/cm)``. vector of length ``N``.
    paramsint16 : np.ndarray
        Vector containing various int parameters, e.g., ``[npre ndur [additional optional values]]``.
        Max length is ``32``.
    paramsfloat : np.ndarray
        Vector containing various RF pulse parameters needed for SAR and B1 scaling calculations -- just a placeholder for now.
        Max length is ``32``.
    system : SystemSpecs
        Struct specifying hardware system info, see systemspecs.py.

    Jon-Fredrik Nielsen, jfnielse@umich.edu
    Python porting: Matteo Cencini

    """
    # get magnitude and phase of pulse
    rho = np.abs(rf)
    theta = np.angle(rf)

    # get number of pulses
    npulses = rf.shape[1]

    # max length of params* vectors
    nparamsint16 = 32
    nparamsfloat = 32

    # RF waveform is scaled relative to system.maxRF.
    # This may be 0.25G/0.125G for quadrature/body RF coils (according to John Pauly RF class notes), but haven't verified...
    # if system.rfUnit == 'uT':
    # 	maxRF = system.maxRF / 100.0 # Gauss

    # checks
    assert len(paramsint16) <= nparamsint16, "too many int16 parameters"
    assert len(paramsfloat) <= nparamsfloat, "too many float parameters"

    # convert params* to row vectors, and pad to max length
    paramsint16 = np.pad(paramsint16, (0, nparamsint16 - len(paramsint16)))
    paramsfloat = np.pad(paramsfloat, (0, nparamsfloat - len(paramsfloat)))

    # get shape
    if len(rho.shape) == 2:
        rho = rho[:, :, None]
        theta = theta[:, :, None]
    res, nrfpulses, ncoils = rho.shape

    # write
    with open(fname, "wb") as f:
        # peak gradient amplitude across all pulses
        grad = np.concatenate((gx.flatten(), gy.flatten(), gz.flatten()))
        gmax = max(1.0, max(abs(grad)))  # to avoid division by zero

        # write header
        globaldesc = "RF waveform file for ssfpbanding project.\n"
        globaldesc = f"{globaldesc}Created by {realpath(__file__)} on {_datestr()}.\n"
        fs = inspect.stack()
        if len(fs) > 2:
            globaldesc = (
                f"{globaldesc}called by (dbstack(3)): {fs[2][1].split(sep)[-1]}\n"
            )

        globaldesc = f"{globaldesc}ncoils = {ncoils}, res = {res}\n"
        desc = f"{globaldesc}{desc}\n"
        np.array(len(desc)).astype(np.int16).byteswap().tofile(f)
        f.write(desc.encode("utf-8"))

        np.array(ncoils).astype(np.int16).byteswap().tofile(
            f
        )  # shorts must be written in binary -- otherwise it won't work on scanner
        np.array(res).astype(np.int16).byteswap().tofile(f)
        np.array(npulses).astype(np.int16).byteswap().tofile(f)
        f.write(
            f"b1max:  {float(system.maxRF)}\n".encode("utf-8")
        )  # (floats are OK in ASCII on scanner)
        f.write(f"gmax:   {float(system.maxGrad)}\n".encode("utf-8"))

        np.array(nparamsint16).astype(np.int16).byteswap().tofile(f)
        np.array(paramsint16).astype(np.int16).byteswap().tofile(f)
        np.array(nparamsfloat).astype(np.int16).byteswap().tofile(f)
        for n in range(nparamsfloat):
            f.write(f"{float(paramsfloat[n])}\n".encode("utf-8"))

        # write binary waveforms (*even* short integers -- the toppe driver/interpreter sets the EOS bit, so don't have to worry about it here)
        max_pg_iamp = (
            2**15 - 2
        )  # RF amp is flipped if setting to 2^15 (as observed on scope), so subtract 2
        rho = 2 * np.round(rho / system.maxRF * max_pg_iamp / 2)
        theta = 2 * np.round(theta / np.pi * max_pg_iamp / 2)
        gx = 2 * np.round(gx / gmax * max_pg_iamp / 2)
        gy = 2 * np.round(gy / gmax * max_pg_iamp / 2)
        gz = 2 * np.round(gz / gmax * max_pg_iamp / 2)

        # cast arrays
        for ip in range(npulses):
            for ic in range(ncoils):
                tmp = rho[:, ip, ic].copy()
                np.array(tmp).astype(np.int16).byteswap().tofile(f)
            for ic in range(ncoils):
                tmp = theta[:, ip, ic].copy()
                np.array(tmp).astype(np.int16).byteswap().tofile(f)

            tmp = gx[:, ip].copy()
            np.array(tmp).astype(np.int16).byteswap().tofile(f)
            tmp = gy[:, ip].copy()
            np.array(tmp).astype(np.int16).byteswap().tofile(f)
            tmp = gz[:, ip].copy()
            np.array(tmp).astype(np.int16).byteswap().tofile(f)


def _datestr() -> str:
    """
    Reproduce datesr(now) from MATLAB.

    Returns
    -------
    str
        Datestring.

    """
    return datetime.now().strftime("%m-%b-%Y %H:%M:%S")
