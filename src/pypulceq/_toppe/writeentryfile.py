"""Porting of TOPPE writeentryfile."""

__all__ = ["writeentryfile"]

def writeentryfile(entry_file, filePath='/usr/g/research/pulseq/seqfiles/myscan/',
                   moduleListFile='modules.txt', loopFile='scanloop.txt',
                   b1ScalingFile='tipdown.mod', readoutFile='readout.mod',
                   seqStampFile='seqstamp.txt', coresFile='cores.txt'):
    """
    Convenience function for writing a toppe<CV1>.entry file.

    This function writes the content of a toppe<CV1>.entry file with specified parameters to the given `entry_file`.
    It ensures that the file has the correct format.

    Parameters
    ----------
    entry_file : str 
        The entry file to be written.
    filePath : str, optional 
        The file path. The default is ``'/usr/g/research/pulseq/myscan/'``.
    moduleListFile : str, optional 
        The module list file. The default is ``'modules.txt'``.
    loopFile : str, optional
        The loop file. The default is ``'scanloop.txt'``.
    b1ScalingFile : str, optional 
        The B1 scaling file. The default is ``'tipdown.mod'``.
    readoutFile : str, optional
        The readout file. The default is ``'readout.mod'``.
    seqStampFile : str, optional
        The sequence stamp file. The default is ``'seqstamp.txt'``.
    coresFile : str, optional
        The cores file. The default is ``'cores.txt'``.

    Example
    -------
    >>> write_entry_file("my_entry_file.txt", filePath="/my/custom/path", moduleListFile="custom_modules.txt")
    
    """
    with open(entry_file, 'wt') as file:
        file.write(f"{filePath}\n")
        file.write(f"{moduleListFile}\n")
        file.write(f"{loopFile}\n")
        file.write(f"{b1ScalingFile}\n")
        file.write(f"{readoutFile}\n")
        file.write(f"{seqStampFile}\n")
        file.write(f"{coresFile}\n")

