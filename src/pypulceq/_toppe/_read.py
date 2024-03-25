"""TOPPE reading utils"""

import numpy as np

def _readloop(loopfile="scanloop.txt"):
    # Read header
    with open(loopfile, "r") as fid:
        _ = fid.readline()  # Skip the first line
        hdr = {}
        (
            hdr["nt"],
            hdr["maxslice"],
            hdr["maxecho"],
            hdr["maxview"],
            hdr["scandur"],
            hdr["version"],
        ) = map(int, fid.readline().split())

    # Read data starting from row 4
    d = np.loadtxt(loopfile, delimiter="\t", skiprows=3)

    return d.astype(int), hdr


def _readmodulelistfile(modulesListFile="modules.txt"):
    
    # Read modules.txt and get waveforms
    with open(modulesListFile, "r") as fid:
        _ = fid.readline()  # Skip line
        ncores = int(fid.readline())  # Number of cores
        _ = fid.readline()  # Skip line

        modArr = []  # Initialize list of module structs

        for ic in range(ncores):
            # Prepare module
            module = {}

            # Read name value pair
            name_value = fid.readline().split("\t")
            name_value = np.asarray(name_value)

            # Module name
            module["fname"] = name_value[0]  # .mod file name

            # Read integer values
            v = name_value[1:].astype(int)

            module["dur"] = int(v[0])
            if module["dur"] < 0:
                raise ValueError("Module duration must be >= 0")

            module["hasRF"] = bool(v[1])
            if module["hasRF"] not in [0, 1]:
                raise ValueError("hasRF must be 0 or 1")

            module["hasDAQ"] = bool(v[2])

            if v.size > 3:
                module["trigpos"] = v[3]
            else:
                module["trigpos"] = -1

            if module["hasDAQ"] not in [0, 1]:
                raise ValueError("hasDAQ must be 0 or 1")

            if module["hasRF"] and module["hasDAQ"]:
                raise ValueError(
                    "Module must be either an RF or DAQ module (or neither)"
                )

            # Read .mod file
            (
                module["rf"],
                module["gx"],
                module["gy"],
                module["gz"],
                desc,
                module["paramsint16"],
                module["paramsfloat"],
                hdr,
            ) = _readmod(module["fname"], False)
            module["res"] = hdr["res"]
            module["npre"] = hdr["npre"]
            module["rfres"] = hdr["rfres"]
            module["wavdur"] = module["gx"].shape[0] * 4  # waveform duration [us]

            modArr.append(module)

    return modArr


def _readmod(fname, showinfo=False):
    # Initialize variables
    nReservedInts = 2  # [nChop(1) rfres], rfres = # samples in RF/ADC window

    # Read ASCII description
    with open(fname, "rb") as fid:
        asciisize = np.fromfile(fid, dtype=np.int16, count=1).byteswap()[0]
        desc = fid.read(asciisize).decode("utf-8")

        # Read rest of the header
        hdr = {}
        hdr["ncoils"] = np.fromfile(fid, dtype=np.int16, count=1).byteswap()[0]
        hdr["res"] = np.fromfile(fid, dtype=np.int16, count=1).byteswap()[0]
        hdr["npulses"] = np.fromfile(fid, dtype=np.int16, count=1).byteswap()[0]
        hdr["b1max"] = float(fid.readline().decode("utf-8").split(":")[1])
        hdr["gmax"] = float(fid.readline().decode("utf-8").split(":")[1])

        # Get nparamsint16 and nparamsfloat
        nparamsint16 = np.fromfile(fid, dtype=np.int16, count=1).byteswap()[0]
        paramsint16 = np.fromfile(fid, dtype=np.int16, count=nparamsint16).byteswap()
        nparamsfloat = np.fromfile(fid, dtype=np.int16, count=1).byteswap()[0]
        paramsfloat = []
        for n in range(nparamsfloat):
            paramsfloat.append(float(fid.readline().decode("utf-8")))
        paramsfloat = np.asarray(paramsfloat)

        # Get nChop (added in v4)
        hdr["npre"] = paramsint16[0]  # number of discarded RF/ADC samples at start
        hdr["rfres"] = paramsint16[1]  # total number of RF/ADC samples

        if showinfo:
            print("\n" + desc)
            print(f'number of coils/channels: {hdr["ncoils"]}')
            print(f'number points in waveform: {hdr["res"]}')
            print(f'number of waveforms: {hdr["npulses"]}')
            print(f"data offset (bytes): {fid.tell()}\n")

        # Read waveforms
        rho = np.zeros((hdr["res"], hdr["npulses"], hdr["ncoils"]))
        theta = np.zeros((hdr["res"], hdr["npulses"], hdr["ncoils"]))
        gx = np.zeros((hdr["res"], hdr["npulses"]))
        gy = np.zeros((hdr["res"], hdr["npulses"]))
        gz = np.zeros((hdr["res"], hdr["npulses"]))

        for ip in range(hdr["npulses"]):
            for ic in range(hdr["ncoils"]):
                rho[:, ip, ic] = np.fromfile(fid, dtype=np.int16, count=hdr["res"]).byteswap()
            for ic in range(hdr["ncoils"]):
                theta[:, ip, ic] = np.fromfile(fid, dtype=np.int16, count=hdr["res"]).byteswap()

            gx[:, ip] = np.fromfile(fid, dtype=np.int16, count=hdr["res"]).byteswap()
            gy[:, ip] = np.fromfile(fid, dtype=np.int16, count=hdr["res"]).byteswap()
            gz[:, ip] = np.fromfile(fid, dtype=np.int16, count=hdr["res"]).byteswap()

    # Convert back to physical units
    max_pg_iamp = 2**15 - 2  # max instruction amplitude (max value of signed short)
    rho = rho * hdr["b1max"] / max_pg_iamp  # Gauss
    theta = theta * np.pi / max_pg_iamp  # radians
    gx = gx * hdr["gmax"] / max_pg_iamp  # Gauss/cm
    gy = gy * hdr["gmax"] / max_pg_iamp
    gz = gz * hdr["gmax"] / max_pg_iamp

    paramsint16 = paramsint16[
        (nReservedInts + 1) :
    ]  # NB! Return only the user-defined ints passed to writemod.m

    rf = rho * np.exp(1j * theta)
    
    # squeeze for python
    rf = rf.squeeze()
    gx = gx.squeeze()
    gy = gy.squeeze()
    gz = gz.squeeze()

    return rf, gx, gy, gz, desc, paramsint16, paramsfloat, hdr


def _readcoresfile(fname):
    blockGroups = []

    with open(fname, "r") as fid:
        _ = fid.readline()  # skip line
        nGroups = int(fid.readline())
        _ = fid.readline()  # skip line

        for i in range(nGroups):
            tmp = "".join(fid.readline().split("\n")[:-1]).split("\t")
            blockIDs = np.asarray([el for el in tmp[1:]]).astype(int)
            blockGroups.append(blockIDs)

    return blockGroups
