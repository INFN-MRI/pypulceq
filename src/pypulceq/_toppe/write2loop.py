"""Porting of TOPPE write2loop."""

__all__ = ["Loop"]

import warnings

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import numpy.typing as npt

from .utils import plotseq

from ._read import _readmodulelistfile


@dataclass
class Loop:
    """
    Represents a loop for a TOPPE sequence.

    Attributes
    ----------
    system : SystemSpecs
        System parameters.
    toppeVer : int
        Version of the TOPPE interpreter.
    loopFile : str, optional
        Output file name for the loop, defaults to "scanloop.txt".
    rf_spoil_seed_cnt : float, optional
        RF spoil seed count, defaults to 0.0.
    rf_spoil_phase : float, optional
        RF spoil phase, defaults to 0.0.
    irfphase : int, optional
        Initial RF phase, defaults to 0.
    idaqphase : int, optional
        Initial DAQ phase, defaults to 0.
    modules : List[str], optional
        List of module names, defaults to None.
    d : np.ndarray[float], optional
        Array representing some data, defaults to None.
    d_index : int, optional
        Index for data, defaults to 0.
    setupdone : bool, optional
        Whether setup is done, defaults to True.
    modulelistfile : str, optional
        File name for the module list, defaults to "modules.txt".

    """

    system: Any  # TODO: replace with correct struct
    toppeVer: int
    loopFile: str = "scanloop.txt"
    rf_spoil_seed_cnt: float = 0.0
    rf_spoil_phase: float = 0.0
    irfphase: int = 0
    idaqphase: int = 0
    modules: List[str] = None
    d: npt.NDArray[float] = None
    d_index: int = 0
    setupdone: bool = True
    modulelistfile: str = "modules.txt"

    def __post_init__(self):
        self.modules = _readmodulelistfile(self.modulelistfile)

        # prepare d
        if self.toppeVer == 6:
            d = np.zeros((1000000, 28), dtype=np.float32)  # 28th column is core id
        elif self.toppeVer == 5:
            self.d = np.zeros(
                (1000000, 27), dtype=np.float32
            )  # 27th column is trigger out flag (0 or 1)
        elif self.toppeVer == 4:
            self.d = np.zeros(
                (1000000, 26), dtype=np.float32
            )  # 26th column is trigger in mode (0 or 1)
        elif self.toppeVer == 3:
            self.d = np.zeros(
                (1000000, 25), dtype=np.float32
            )  # Columns 17-25 contain 3D rotation matrix
        else:
            self.d = np.zeros((1000000, 16), dtype=np.float32)

        # assign
        self.d = d
        self.d_index = 0
        self.setupdone = True

    def finish(self):
        # parse
        system = self.system
        toppeVer = self.toppeVer
        modules = self.modules
        loopFile = self.loopFile

        # Remove rows that are all zeros (preallocated but unused)
        d = self.d[: self.d_index, :].astype(int)

        # Check if all lines are integers
        if not np.all(np.isclose(d, np.round(d))):
            raise ValueError("Value in d is a non-integer, this should never happen!")

        # Calculate header values
        nt = d.shape[0]  # number of startseq() calls
        maxslice = np.max(d[:, 6])
        maxecho = np.max(d[:, 7])
        maxview = np.max(d[:, 8])

        # check if max 'slice', 'echo', and 'view' numbers in scanloop.txt exceed system limits
        if maxslice > system.maxSlice:
            warnings.warn(
                f"maxslice (={maxslice}) > system.maxSlice {(system.maxSlice)} -- scan may not run!"
            )
        if maxecho + 1 > system.maxEcho:  # +1 since 'echo' starts at 0
            warnings.warn(
                f"maxecho (={maxecho+1}) > system.maxEcho (={system.maxEcho}) -- scan may not run!"
            )
        if maxview > system.maxView:
            warnings.warn(
                f"maxview (={maxview}) > system.maxView (={system.maxView}) -- scan may not run!"
            )

        dur = _getscantime(system, loopArr=d, mods=modules)
        udur = np.round(dur * 1e6)

        if toppeVer > 2:
            newParams = np.asarray([nt, maxslice, maxecho, maxview, udur, toppeVer])
        else:
            newParams = np.asarray([nt, maxslice, maxecho, maxview, udur])

        # Write file header
        with open(loopFile, "w+") as fid:
            if toppeVer > 2:
                fid.write("nt\tmaxslice\tmaxecho\tmaxview\tscandur\tversion\n")
                lineformat = "%d\t%d\t%d\t%d\t%d\t%d\n"
            else:
                fid.write("nt\tmaxslice\tmaxecho\tmaxview\tscandur\n")
                lineformat = "%d\t%d\t%d\t%d\t%d\n"

            fid.write(lineformat % tuple(newParams))  # Updated line
            fid.write(headerline())  # Assuming headerline is defined somewhere
            np.savetxt(
                fid, d, delimiter="\t", fmt="%8d"
            )  # Write scanloop lines from d matrix.

        # Disable setup flag to conclude file
        self.modules = None  # Erase modules struct from memory for safety in case .mod files get updated

    def write2loop(
        self,
        modname,
        Gamplitude=[1, 1, 1],
        core=1,
        waveform=1,
        textra=0,
        RFamplitude=1,
        RFphase=0,
        DAQphase=0,
        RFspoil=False,
        RFoffset=0,
        slice=1,
        echo=1,
        view=1,
        dabmode="on",
        rot=0,
        rotmat=np.eye(3),
        trig=0,
        trigout=0,
    ):
        # parse
        toppeVer = self.toppeVer
        modules = self.modules
        d = self.d
        d_index = self.d_index

        # Apply in-plane rotation angle 'rot' to rotmat
        n = np.dot(rotmat, np.array([0, 0, 1]))
        R = _angleaxis2rotmat(rot, n)
        rotmat = np.dot(R, rotmat)

        # Find input module in persistent module struct
        # Find module in cell array and store the index in moduleno
        if modname == "delay":
            # Pure delay block
            iModule = -1
            module = {"hasRF": 0, "hasDAQ": 0}
        else:
            for iModule in range(len(modules)):
                if modname == modules[iModule]["fname"]:
                    break
            else:
                raise ValueError(f"Can't find {modname} in {self.modulelistfile}")
            module = modules[iModule]

        # Calculate gradient scalings
        ia_gx = int(2 * np.round(Gamplitude[0] * max_pg_iamp() / 2))
        ia_gy = int(2 * np.round(Gamplitude[1] * max_pg_iamp() / 2))
        ia_gz = int(2 * np.round(Gamplitude[2] * max_pg_iamp() / 2))

        # Calculate time at end of module
        textra_us = int(4 * round(1000 * textra / 4))

        # 3d rotation matrix
        if toppeVer > 2:
            if rotmat.ndim != 2 or rotmat.shape != (3, 3):
                raise ValueError("rotmat must be a 3x3 orthonormal matrix")
            drot = np.round(max_pg_iamp() * rotmat.flatten()).astype(
                int
            )  # vectorized, in row-major order
            phi = 0  # in-plane rotation (already applied to 3D rotation matrix)
        else:
            drot = []
            phi = np.angle(np.exp(1j * rot))  # in-plane rotation. wrap to [-pi pi]

        # 2D in-plane rotation ('iphi' only applies to v2)
        iphi = int(2 * np.round(phi / np.pi * max_pg_iamp() / 2))

        # Trigger
        if toppeVer <= 3:
            trig = []

        # Trigger (TTL pulse) out
        if toppeVer > 4:
            trigout = trigout

        if module["hasRF"]:  # Write RF module
            # Do RF amplitude stuff
            ia_rf = int(2 * np.round(RFamplitude * max_pg_iamp() / 2))
            ia_th = max_pg_iamp()

            # Do RF phase stuff
            if RFspoil:  # If spoil is called, replace RF phase with spoil phase
                rf_spoil_phase = self.updateSpoilPhase()
                irfphase = _phase2int(rf_spoil_phase)
                idaqphase = irfphase
            else:
                irfphase = _phase2int(RFphase)
                idaqphase = _phase2int(DAQphase)

            # Dummy slice/echo/view
            dabslice = 0
            dabecho = 0
            dabview = 1

            # Frequency offset in Hz
            if RFoffset != round(RFoffset):
                print("Rounding frequency offset to nearest value")
                RFoffset = round(RFoffset)
            f = int(RFoffset)

            # Write line values and increment
            if toppeVer == 4:
                d[d_index, :] = np.concatenate(
                    [
                        iModule + 1,
                        ia_rf,
                        ia_th,
                        ia_gx,
                        ia_gy,
                        ia_gz,
                        dabslice,
                        dabecho,
                        dabview,
                        0,
                        iphi,
                        irfphase,
                        irfphase,
                        textra_us,
                        f,
                        waveform,
                    ],
                    drot,
                    [
                        trig,
                    ],
                )
            elif toppeVer == 5:
                d[d_index, :] = np.concatenate(
                    [
                        [
                            iModule + 1,
                            ia_rf,
                            ia_th,
                            ia_gx,
                            ia_gy,
                            ia_gz,
                            dabslice,
                            dabecho,
                            dabview,
                            0,
                            iphi,
                            irfphase,
                            irfphase,
                            textra_us,
                            f,
                            waveform,
                        ],
                        drot,
                        [
                            trig,
                            trigout,
                        ],
                    ]
                )
            elif toppeVer == 6:
                d[d_index, :] = np.concatenate(
                    [
                        [
                            iModule + 1,
                            ia_rf,
                            ia_th,
                            ia_gx,
                            ia_gy,
                            ia_gz,
                            dabslice,
                            dabecho,
                            dabview,
                            0,
                            iphi,
                            irfphase,
                            irfphase,
                            textra_us,
                            f,
                            waveform,
                        ],
                        drot,
                        [
                            trig,
                            trigout,
                            core,
                        ],
                    ]
                )

            d_index += 1

        elif module["hasDAQ"]:  # Write DAQ module
            # Set slice/echo/view
            if slice == "dis":
                dabslice = 0
            elif slice > 0:
                dabslice = slice
            else:
                raise ValueError('Slice must be "dis" or > 0')

            dabecho = echo - 1  # Index echo from 0 to n-1
            dabview = view

            # Receive phase
            if RFspoil:  # If spoil is called, replace RF phase with spoil phase
                idaqphase = irfphase
            else:
                idaqphase = _phase2int(DAQphase)

            if toppeVer == 4:
                d[d_index, :] = np.concatenate(
                    [
                        iModule,
                        0,
                        0,
                        ia_gx,
                        ia_gy,
                        ia_gz,
                        dabslice,
                        dabecho,
                        dabview,
                        dabval(dabmode),
                        iphi,
                        idaqphase,
                        idaqphase,
                        textra_us,
                        0,
                        waveform,
                    ],
                    drot,
                    [
                        trig,
                    ],
                )
            elif toppeVer == 5:
                d[d_index, :] = np.concatenate(
                    [
                        [
                            iModule,
                            0,
                            0,
                            ia_gx,
                            ia_gy,
                            ia_gz,
                            dabslice,
                            dabecho,
                            dabview,
                            dabval(dabmode),
                            iphi,
                            idaqphase,
                            idaqphase,
                            textra_us,
                            0,
                            waveform,
                        ],
                        drot,
                        [
                            trig,
                            trigout,
                        ],
                    ]
                )
            elif toppeVer == 6:
                d[d_index, :] = np.concatenate(
                    [
                        [
                            iModule,
                            0,
                            0,
                            ia_gx,
                            ia_gy,
                            ia_gz,
                            dabslice,
                            dabecho,
                            dabview,
                            dabval(dabmode),
                            iphi,
                            idaqphase,
                            idaqphase,
                            textra_us,
                            0,
                            waveform,
                        ],
                        drot,
                        [
                            trig,
                            trigout,
                            core,
                        ],
                    ]
                )

            d_index += 1
        else:
            # Gradients only
            if toppeVer == 4:
                d[d_index, :] = np.concatenate(
                    [
                        iModule,
                        0,
                        0,
                        ia_gx,
                        ia_gy,
                        ia_gz,
                        0,
                        0,
                        0,
                        0,
                        iphi,
                        0,
                        0,
                        textra_us,
                        0,
                        waveform,
                    ],
                    drot,
                    [
                        trig,
                    ],
                )
            elif toppeVer == 5:
                d[d_index, :] = np.concatenate(
                    [
                        [
                            iModule,
                            0,
                            0,
                            ia_gx,
                            ia_gy,
                            ia_gz,
                            0,
                            0,
                            0,
                            0,
                            iphi,
                            0,
                            0,
                            textra_us,
                            0,
                            waveform,
                        ],
                        drot,
                        [
                            trig,
                            trigout,
                        ],
                    ]
                )
            elif toppeVer == 6:
                d[d_index, :] = np.concatenate(
                    [
                        [
                            iModule,
                            0,
                            0,
                            ia_gx,
                            ia_gy,
                            ia_gz,
                            0,
                            0,
                            0,
                            0,
                            iphi,
                            0,
                            0,
                            textra_us,
                            0,
                            waveform,
                        ],
                        drot,
                        [
                            trig,
                            trigout,
                            core,
                        ],
                    ]
                )

            d_index += 1

        # replace (not sure if that's needed)
        if module["hasRF"]:
            if RFspoil:
                self.rf_spoil_phase = rf_spoil_phase
            self.irfphase = irfphase
            self.idaqphase = idaqphase
        self.modules = modules
        self.d = d
        self.d_index = d_index

    def updateSpoilPhase(self):
        self.rf_spoil_seed_cnt = self.rf_spoil_seed_cnt + 1
        self.rf_spoil_phase = (
            self.rf_spoil_phase
            + np.deg2rad(self.rf_spoil_seed()) * self.rf_spoil_seed_cnt
        )
        return self.rf_spoil_phase


# %% Helper functions
def _angleaxis2rotmat(angle, axis):
    # Do the work: =================================================================
    s = np.sin(angle)
    c = np.cos(angle)

    # Normalized vector:
    u = np.asarray(axis, dtype=np.float32)
    u = u / np.sqrt(u.T @ u)

    # 3D rotation matrix:
    x = u[0]
    y = u[1]
    z = u[2]
    mc = 1 - c

    # build rows
    R0 = np.stack(
        (c + x * x * mc, x * y * mc - z * s, x * z * mc + y * s), axis=-1
    )  # (nalpha, 3)
    R1 = np.stack(
        (x * y * mc + z * s, c + y * y * mc, y * z * mc - x * s), axis=-1
    )  # (nalpha, 3)
    R2 = np.stack(
        (x * z * mc - y * s, y * z * mc + x * s, c + z * z * mc), axis=-1
    )  # (nalpha, 3)

    # stack rows
    R = np.stack((R0, R1, R2), axis=1)

    return R


def _phase2int(phase):
    # Implement this function to convert phase to an integer value
    phstmp = np.arctan2(np.sin(phase), np.cos(phase))  # wrap phase to (-pi, pi) range
    return int(2 * np.round(phstmp / np.pi * max_pg_iamp() / 2))  # short int


def _getscantime(sysGE, loopArr, mods):
    # Call plotseq with 'doTimeOnly' argument to get the sequence duration quickly
    T = plotseq(sysGE, timeRange=[0, 99999999999], loop=loopArr, timeOnly=True)
    dur = T[1]

    return dur


# %% Constants
def max_pg_iamp():
    return 2**15 - 2  # Maximum single int gradient value


def rf_spoil_seed():
    return 117


def trig_intern():
    return 0


def trig_ecg():
    return 1


def dformat():
    # Return the format string for the header line
    return "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d \n"


def headerline():
    return "Core a_rf a_th a_gx a_gy a_gz dabslice dabecho dabview dabmode rot rfphase recphase textra freq waveform rotmat...\n"


def dabval(dabmode):
    # Return the integer value based on the 'dabmode' input
    if dabmode == "off":
        return 0
    elif dabmode == "on":
        return 1
    elif dabmode == "add":
        return 2
    elif dabmode == "reset":
        return 3
    else:
        raise ValueError("Invalid dabmode option specified")
