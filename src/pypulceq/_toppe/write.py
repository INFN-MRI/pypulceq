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


def write_sequence(seqname, seqdict):
    # get hardware specifications
    sysdict = seqdict["sys"]
    sys = SystemSpecs(**sysdict)

    # loop over module and write all of them
    modules = seqdict["modules"]
    print("Writing sequence modules...\n")
    for k, m in modules.items():
        writemod(sys, **m)

    # now write modulelist
    print("Writing modules list...\n")
    writemodfile(modules, sys)

    # write corefiles
    cores = seqdict["cores"]
    print("Writing group list...\n")
    writecoresfile(cores, modules)

    # iterate and write loop
    loop = seqdict["loop"]
    seq = Loop(sys, toppeVer=6, modules=[mod["ofname"] for mod in modules.values()])
    print("Writing loop...\n")
    for event in loop:
        seq.write2loop(**event)
    seq.finish()

    # write entry file
    print("Entry file...\n")
    writeentryfile(
        "toppeN.entry", filePath=f"/usr/g/research/pulseq/seqfiles/{seqname}/"
    )

    # create 'sequence stamp' file for TOPPE.
    # This file is listed in line 6 of toppeN.entry
    print("Doing preflight check...\n")
    preflightcheck("toppeN.entry", "seqstamp.txt", sys)

    # put TOPPE files in a .tar file (for convenience) and cleaup
    modfiles = [mod["ofname"] for mod in modules.values()]
    print("Archive and clean-up...\n")
    archive_and_cleanup(
        f"{seqname}.tar",
        ["toppeN.entry", "seqstamp.txt", "modules.txt", "scanloop.txt", "cores.txt"]
        + modfiles,
    )

    print(f"Sequence file {seqname} ready for execution on GE scanners\n")


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
