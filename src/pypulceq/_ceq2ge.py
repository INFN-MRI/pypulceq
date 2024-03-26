"""Ceq2GE subroutines."""

__all__ = ["ceq2ge"]

import tarfile
import os

import numpy as np

from . import _toppe as toppe
from ._interp import gradinterp


def ceq2ge(ceq, sysGE, ofname, ignoreTrigger=False, seqGradRasterTime=10e-6):
    """
    Write a Ceq struct to a set of files that can be executed on GE scanners using the TOPPE interpreter (v6).

    Parameters
    ----------
    ceq : SimpleNamespace
        A namespace representing the Ceq struct.
    sysGE : SystemSpecs
        System parameters for the GE scanner.
    ofname : str
        Output file name.
    ignoreTrigger : bool, optional
        Whether to ignore trigger events. Defaults to False.
    seqGradRasterTime : float, optional
        Gradient raster time in .seq file. Defaults to 10e-6.

    """
    # Define .mod file names
    modFiles = [f"module{p}.mod" for p in range(ceq.nParentBlocks)]

    # Write .mod files
    b1ScalingFile, readoutFile = _writemodules(ceq, modFiles, sysGE, seqGradRasterTime)

    # Write modules.txt
    toppe.writemodfile(ceq, modFiles, sysGE, seqGradRasterTime, ignoreTrigger)

    # Write segment definition file (cores.txt) and determine TOPPE version
    toppeVersion = _writecoresfile(ceq)

    # Write scanloop.txt
    _writescanloop(ceq, modFiles, sysGE, toppeVersion)

    # Write .entry file
    toppe.writeentryfile(
        "toppeN.entry",
        filePath="/usr/g/research/pulseq/v6/seq2ge/",
        b1ScalingFile=b1ScalingFile,
        readoutFile=readoutFile,
    )

    # Create 'sequence stamp' file for TOPPE
    toppe.preflightcheck("toppeN.entry", "seqstamp.txt", sysGE)

    # Put TOPPE files in a .tar file (for convenience)
    if toppeVersion > 5:
        _archive_and_cleanup(
            f"{ofname}.tar",
            ["toppeN.entry", "seqstamp.txt", "modules.txt", "scanloop.txt", "cores.txt"]
            + modFiles,
        )
    else:
        _archive_and_cleanup(
            f"{ofname}.tar",
            ["toppeN.entry", "seqstamp.txt", "modules.txt", "scanloop.txt"] + modFiles,
        )

    print(f"Sequence file {ofname} ready for execution on GE scanners")


# %%  local utils
def _writemodules(ceq, modFiles, sysGE, seqGradRasterTime):
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

    """
    # Initialize default arguments
    b1ScalingFileIsDefined = False
    peakB1InSequence = 0
    adcCount = 0
    toppeVersion = 6 if ceq.nGroups > 0 else 5

    # initialize
    hasRF = []
    hasADC = []

    for p in range(ceq.nParentBlocks):
        b = ceq.parentBlocks[p]

        # Initialize defaults
        rf, grad = None, {"x": None, "y": None, "z": None}
        isDelayBlock = True

        # check if block contains RF, ADC or none (pure delay)
        hasRF = hasattr(b, "rf")
        hasADC = hasattr(b, "adc")

        # Interpolate RF waveforms and convert to Gauss
        if hasRF:
            if not b1ScalingFileIsDefined or max(abs(b.rf.signal)) > peakB1InSequence:
                b1ScalingFile = modFiles[p]
                peakB1InSequence = max(abs(b.rf.signal))
                b1ScalingFileIsDefined = True

            if b.rf.delay + sysGE.rfDeadTime * 1e-6 < b.blockDuration:
                raise ValueError(
                    f"Parent block {p}: RF delay must be >= sysGE.rfDeadTime"
                )

            if (
                b.rf.delay + b.rf.shape_dur + sysGE.rfRingdownTime * 1e-6
                > b.blockDuration
            ):
                raise ValueError(
                    f"Parent block {p}: RF ringdown extends past end of block"
                )

            tge = np.arange(seqGradRasterTime / 2, b.rf.shape_dur, seqGradRasterTime)
            rf = np.interp(tge, b.rf.t, b.rf.signal) / sysGE.gamma * 1e4  # Gauss
            npre = int(np.round(b.rf.delay / seqGradRasterTime))
            rf = np.concatenate((np.zeros(npre), rf))

            isDelayBlock = False

        # Interpolate gradient waveforms and convert to Gauss/cm
        for ax in ["x", "y", "z"]:
            g = getattr(b, f"g{ax}", None)
            if g is not None:
                isDelayBlock = False
                grad[ax] = gradinterp(g, sysGE, seqGradRasterTime=seqGradRasterTime)

        # ADC
        if hasADC:
            if b.adc.delay + sysGE.adcDeadTime * 1e-6 < b.blockDuration:
                raise ValueError(f"Parent block {p}: ADC delay is < sysGE.adcDeadTime")

            if b.adc.delay + b.adc.numSamples * b.adc.dwell > b.blockDuration:
                raise ValueError(
                    f"Parent block {p}: ADC window extends past end of block"
                )

            npre = int(np.round(b.adc.delay / seqGradRasterTime))
            rfres = int(np.round(b.adc.numSamples * b.adc.dwell / seqGradRasterTime))
            readoutFile = modFiles[p]

        # Set nChop, which is the number of samples to trim from beginning and end of RF/ADC window
        n = max(len(rf) if rf is not None else 0, *[len(grad[ax]) for ax in grad])
        nChop = [npre, n - npre - rfres] if "rfres" in locals() else [0, 0]

        # Write .mod file
        if not isDelayBlock:
            toppe.writemod(
                sysGE,
                ofname=modFiles[p],
                rf=rf,
                gx=grad["x"],
                gy=grad["y"],
                gz=grad["z"],
                nChop=nChop,
            )

    return b1ScalingFile, readoutFile


def _writecoresfile(ceq):
    """
    Write core information to 'cores.txt' based on the block groups defined in ceq.

    Parameters
    ----------
    ceq : SimpleNamespace
        A namespace representing the Ceq struct.

    Returns
    -------
    int
        The TOPPE version used for writing cores file.

    """
    if ceq.nGroups > 0:
        toppeVersion = 6
        blockGroups = [[blockID for blockID in group.blockIDs] for group in ceq.groups]
        toppe.writecoresfile(blockGroups)
    else:
        toppeVersion = 5

    return toppeVersion


def _writescanloop(ceq, modFiles, sysGE, toppeVersion):
    """
    Write scanloop information to 'scanloop.txt'.

    Parameters
    ----------
    ceq : SimpleNamespace
        A namespace representing the Ceq struct.
    modFiles : list of str
        List of mod file names.
    sysGE : dict
        System parameters for the GE scanner.
    toppeVersion : str
        Version of the TOPPE interpreter.

    """
    sl = 1
    view = 1
    echo = 0
    adcCount = 0

    seq = toppe.Loop(sysGE, toppeVer=toppeVersion, modulelistfile=modFiles)

    for n in range(ceq.nMax):
        i, p = ceq.loop[n, :2]

        if p == 0:
            seq.write2loop("delay", textra=np.round(ceq.loop[n, 9] * 1e6) / 1e3, core=i)
            continue

        if ceq.parentBlocks[p].amp.rf > 0:
            RFamplitude = ceq.loop[n, 2] / ceq.parentBlocks[p].amp.rf
        else:
            RFamplitude = 0

        RFphase = ceq.loop[n, 3]
        RFoffset = np.round(ceq.loop[n, 4])

        if hasattr(ceq.parentBlocks[p], "adc"):
            view = (adcCount % sysGE.maxView) + 1
            sl = (adcCount // sysGE.maxView) + 1
            if sl > sysGE.maxSlice:
                raise ValueError(f"max number of slices exceeded ({sysGE.maxSlice})")
            echo = adcCount // (sysGE.maxView * sysGE.maxSlice)
            if echo > sysGE.maxEcho:
                raise ValueError(f"max number of echoes exceeded ({sysGE.maxEcho})")
            adcCount += 1

        amp = {
            ax: ceq.loop[n, 4 + i] / ceq.parentBlocks[p].amp[ax]
            if ceq.parentBlocks[p].amp[ax] > 0
            else 0
            for i, ax in enumerate(["gx", "gy", "gz"])
        }

        DAQphase = ceq.loop[n, 8]

        trigout = 1 if hasattr(ceq.parentBlocks[p], "trig") else 0

        seq.write2loop(
            modFiles[p],
            Gamplitude=[amp["gx"], amp["gy"], amp["gz"]],
            RFamplitude=RFamplitude,
            RFphase=RFphase,
            DAQphase=DAQphase,
            RFoffset=RFoffset,
            slice=sl,
            echo=echo + 1,
            view=view,
            dabmode="on",
            textra=0,
            waveform=1,
            trigout=trigout,
            core=i,
        )

    seq.finish()


def _archive_and_cleanup(archive_filename, files_to_delete):
    """
    Create a tar archive from a list of files and then delete those files.

    Parameters
    ----------
    archive_filename : str
        The name of the tar archive to be created.
    files_to_delete : list
        List of file paths to be added to the archive and deleted.

    """
    with tarfile.open(archive_filename, "w") as archive:
        for file_to_delete in files_to_delete:
            if os.path.exists(file_to_delete):
                archive.add(file_to_delete)

    # Delete the listed files
    for file_to_delete in files_to_delete:
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
