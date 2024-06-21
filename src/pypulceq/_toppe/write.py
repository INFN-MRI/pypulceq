"""Main GE Pulseq writing routine."""

__all__ = ["write_sequence"]

import tarfile
import os

from .systemspecs import SystemSpecs
from .writecoresfile import writecoresfile
from .writemod import writemod
from .writemod import writemodfile
from .write2loop import Loop
from .writeentryfile import writeentryfile
from .preflightcheck import preflightcheck


def write_sequence(seqname, seqdict, ignore_trigger=False, sequence_path=None, verbose=False):
    
    # generate filepath
    if sequence_path is None:
        "/usr/g/research/pulseq/v6/seq2ge/"
        
    # extract readout file and b1 scaling file names
    readout_name = seqdict["readout_name"]
    b1scaling_name = seqdict["b1scaling_name"]
    
    # get hardware specifications
    sysdict = seqdict["sys"]
    sys = SystemSpecs(**sysdict)

    # loop over module and write all of them
    modules = seqdict["modules"]
    if verbose:
        print("Writing sequence modules...", end="\tab")
    for k, m in modules.items():
        writemod(sys, **m)
    if verbose:
        print("done!\n")

    # now write modulelist
    if verbose:
        print("Writing modules list...", end="\tab")
    writemodfile(modules, sys)
    if verbose:
        print("done!\n")

    # write corefiles
    cores = seqdict["cores"]
    if verbose:
        print("Writing group list...", end="\tab")
    writecoresfile(cores, modules, ignore_trigger)
    if verbose:
        print("done!\n")

    # iterate and write loop
    loop = seqdict["loop"]
    seq = Loop(sys, toppeVer=6, modules=[mod["ofname"] for mod in modules.values()])
    if verbose:
        print("Writing scan loop...", end="\tab")
    for event in loop:
        seq.write2loop(**event)
    seq.finish()
    if verbose:
        print("done!\n")

    # write entry file
    if verbose:
        print("Writing entry file...", end="\tab")
    writeentryfile(
        "toppeN.entry", filePath=f"{sequence_path}{seqname}/", b1ScalingFile=b1scaling_name,
        readoutFile=readout_name,
    )
    if verbose:
        print("done!\n")

    # create 'sequence stamp' file for TOPPE.
    # This file is listed in line 6 of toppeN.entry
    if verbose:
        print("Doing preflight check...", end="\tab")
    preflightcheck("toppeN.entry", "seqstamp.txt", sys)
    if verbose:
        print("done!\n")

    # put TOPPE files in a .tar file (for convenience) and cleaup
    modfiles = [mod["ofname"] for mod in modules.values()]
    if verbose:
        print("Archive and clean-up...", end="\tab")
    archive_and_cleanup(
        f"{seqname}.tar",
        ["toppeN.entry", "seqstamp.txt", "modules.txt", "scanloop.txt", "cores.txt"]
        + modfiles,
    )
    if verbose:
        print("done!\n")


# %%  local utils
def archive_and_cleanup(archive_filename, files_to_delete):
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
