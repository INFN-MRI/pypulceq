"""Porting of TOPPE writecoresfile.m"""

__all__ = ["writecoresfile"]


def writecoresfile(cores):
    """
    Write core information to 'cores.txt'.

    Parameters
    ----------
    cores : list of lists
        List of lists containing module IDs for each core.

    """
    with open("cores.txt", "wt") as fid:
        fid.write("Total number of cores\n")
        fid.write(f"{len(cores)}\n")
        fid.write("nmodules modIds... \n")

        for icore in range(len(cores)):
            nmod = len(cores[icore])
            fid.write(f"{nmod}\t")
            for imod in range(nmod):
                modid = cores[icore][imod]
                fid.write(f"{modid}")
                if imod < nmod - 1:
                    fid.write("\t")
                else:
                    fid.write("\n")
