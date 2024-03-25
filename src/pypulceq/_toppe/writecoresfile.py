"""Porting of TOPPE writecoresfile.m"""

__all__ = ["writecoresfile"]

def writecoresfile(cores, modules):
    
    with open('cores.txt', 'wt') as fid:
        fid.write('Total number of cores\n')
        fid.write(f'{len(cores)}\n')
        fid.write('nmodules modIds... \n')

        for _, core in cores.items():
            nmod = len(core)
            fid.write(f'{nmod}\t')
            for imod in range(nmod):
                if core[imod] == "delay":
                    modid = 0
                else:
                    modid = list(modules.keys()).index(core[imod]) + 1
                fid.write(f'{modid}')
                if imod < nmod-1:
                    fid.write('\t')
                else:
                    fid.write('\n')