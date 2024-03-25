
import numpy as np
import toppe

def ceq2ge(ceq, sysGE, ofname, ignoreTrigger=False, seqGradRasterTime=10e-6):
    """
    Write a Ceq struct to a set of files that can be executed on GE scanners using the TOPPE interpreter (v6).

    Args:
        ceq (Namespace): A namespace representing the Ceq struct.
        sysGE (dict): System parameters for the GE scanner.
        ofname (str): Output file name.
        ignoreTrigger (bool, optional): Whether to ignore trigger events. Defaults to False.
        seqGradRasterTime (float, optional): Gradient raster time in .seq file. Defaults to 10e-6.
        
    """

    # Define .mod file names
    modFiles = [f'module{p}.mod' for p in range(1, ceq.nParentBlocks + 1)]

    # Initialize default arguments
    b1ScalingFileIsDefined = False
    peakB1InSequence = 0
    adcCount = 0
    toppeVersion = 6 if ceq.nGroups > 0 else 5

    # Write .mod files
    for p in range(1, ceq.nParentBlocks + 1):
        b = ceq.parentBlocks[p - 1]

        # Initialize defaults
        rf, grad = None, {'x': None, 'y': None, 'z': None}
        isDelayBlock = True

        hasRF = bool(b.rf)
        hasADC = bool(b.adc)

        # Interpolate RF waveforms and convert to Gauss
        if hasRF:
            if not b1ScalingFileIsDefined or max(abs(b.rf.signal)) > peakB1InSequence:
                b1ScalingFile = modFiles[p - 1]
                peakB1InSequence = max(abs(b.rf.signal))
                b1ScalingFileIsDefined = True

            if b.rf.delay + sysGE.rfDeadTime * 1e-6 < b.blockDuration:
                raise ValueError(f'Parent block {p}: RF delay must be >= sysGE.rfDeadTime')

            if b.rf.delay + b.rf.shape_dur + sysGE.rfRingdownTime * 1e-6 > b.blockDuration:
                raise ValueError(f'Parent block {p}: RF ringdown extends past end of block')

            tge = np.arange(seqGradRasterTime / 2, b.rf.shape_dur, seqGradRasterTime)
            rf = np.interp(tge, b.rf.t, b.rf.signal) / sysGE.gamma * 1e4  # Gauss
            npre = int(np.round(b.rf.delay / seqGradRasterTime))
            rf = np.concatenate((np.zeros(npre), rf))

            isDelayBlock = False

        # Interpolate gradient waveforms and convert to Gauss/cm
        for ax in ['x', 'y', 'z']:
            g = getattr(b, f'g{ax}', None)
            if g is not None:
                isDelayBlock = False
                grad[ax] = toppe.gradinterp(g, sysGE, seqGradRasterTime=seqGradRasterTime)

        # ADC
        if hasADC:
            if b.adc.delay + sysGE.adcDeadTime * 1e-6 < b.blockDuration:
                raise ValueError(f'Parent block {p}: ADC delay is < sysGE.adcDeadTime')

            if b.adc.delay + b.adc.numSamples * b.adc.dwell > b.blockDuration:
                raise ValueError(f'Parent block {p}: ADC window extends past end of block')

            npre = int(np.round(b.adc.delay / seqGradRasterTime))
            rfres = int(np.round(b.adc.numSamples * b.adc.dwell / seqGradRasterTime))
            readoutFile = modFiles[p - 1]

        # Set nChop, which is the number of samples to trim from beginning and end of RF/ADC window
        n = max(len(rf) if rf is not None else 0, *[len(grad[ax]) for ax in grad])
        nChop = [npre, n - npre - rfres] if 'rfres' in locals() else [0, 0]

        # Write .mod file
        if not isDelayBlock:
            toppe.writemod(sysGE, ofname=modFiles[p - 1], rf=rf, gx=grad['x'], gy=grad['y'], gz=grad['z'],
                           nChop=nChop)

    # Write modules.txt
    with open('modules.txt', 'w') as fid:
        fid.write('Total number of unique cores\n')
        fid.write(f'{ceq.nParentBlocks}\n')
        fid.write('fname dur(us) hasRF hasADC trigpos\n')

        for p in range(1, ceq.nParentBlocks + 1):
            b = ceq.parentBlocks[p - 1]

            if hasRF and hasADC:
                raise ValueError('Block cannot contain both RF and ADC events')

            # Determine trigger position
            if hasattr(b, 'trig') and not ignoreTrigger:
                trigpos = round(b.trig.delay * 1e6) if b.trig.delay + eps >= 100e-6 else 100
            else:
                trigpos = -1  # no trigger

            rf = toppe.readmod(modFiles[p - 1]) if hasRF else []
            dur = max(len(rf) * seqGradRasterTime * 1e6,
                      round(floor(b.blockDuration / seqGradRasterTime) * seqGradRasterTime * 1e6))

            fid.write(f'{modFiles[p - 1]}\t{int(dur)}\t{int(hasRF)}\t{int(hasADC)}\t{int(trigpos)}\n')

    # Write segment definition file (cores.txt) and determine TOPPE version
    if ceq.nGroups > 0:
        toppeVersion = 6
        blockGroups = [[blockID + 1 for blockID in group.blockIDs] for group in ceq.groups]
        toppe.writecoresfile(blockGroups)
    else:
        toppeVersion = 5

    # Write scanloop.txt
    toppe.write2loop('setup', sysGE, version=toppeVersion)

    for n in range(1, ceq.nMax + 1):
        i, p = ceq.loop[n - 1, :2]

        if p == 0:  # delay block
            toppe.write2loop('delay', sysGE, textra=round(ceq.loop[n - 1, 9] * 1e6) / 1e3, core=i)
            continue

        if ceq.parentBlocks[p - 1].amp.rf > 0:
            RFamplitude = ceq.loop[n - 1, 2] / ceq.parentBlocks[p - 1].amp.rf
        else:
            RFamplitude = 0

        RFphase = ceq.loop[n - 1, 3]
        RFoffset = round(ceq.loop[n - 1, 4])

        # Set slice/echo/view indeces (if block is an acquisition block)
        if hasADC:
            view = adcCount % sysGE['maxView'] + 1
            sl = adcCount // sysGE['maxView'] + 1
            if sl > sysGE['maxSlice']:
                raise ValueError(f'max number of slices exceeded ({sysGE["maxSlice"]})')
            echo = adcCount // (sysGE['maxView'] * sysGE['maxSlice'])
            if echo > sysGE['maxEcho']:
                raise ValueError(f'max number of echoes exceeded ({sysGE["maxEcho"]})')
            adcCount += 1

        amp = {ax: ceq.loop[n - 1, 4 + i] / ceq.parentBlocks[p - 1].amp[ax] if ceq.parentBlocks[p - 1].amp[ax] > 0 else 0
               for i, ax in enumerate(['gx', 'gy', 'gz'])}

        DAQphase = ceq.loop[n - 1, 9]

        trigout = 1 if hasattr(ceq.parentBlocks[p - 1], 'trig') else 0

        toppe.write2loop(modFiles[p - 1], sysGE, Gamplitude=np.array([amp['gx'], amp['gy'], amp['gz']]),
                          RFamplitude=RFamplitude, RFphase=RFphase, DAQphase=DAQphase, RFoffset=RFoffset,
                          slice=sl, echo=echo + 1, view=view, dabmode='on', textra=0, waveform=1,
                          trigout=trigout, core=i)

    toppe.write2loop('finish', sysGE)

    # Write .entry file
    toppe.writeentryfile('toppeN.entry', filePath='/usr/g/research/pulseq/v6/seq2ge/', b1ScalingFile=b1ScalingFile,
                         readoutFile=readoutFile)

    # Create 'sequence stamp' file for TOPPE
    toppe.preflightcheck('toppeN.entry', 'seqstamp.txt', sysGE)

    # Put TOPPE files in a .tar file (for convenience)
    if toppeVersion > 5:
        system(f'tar cf {ofname} toppeN.entry seqstamp.txt modules.txt scanloop.txt cores.txt')
    else:
        system(f'tar cf {ofname} toppeN.entry seqstamp.txt modules.txt scanloop.txt')

    for p in range(1, ceq.nParentBlocks + 1):
        system(f'tar rf {ofname} {modFiles[p - 1]}')

    # Clean up (unless in verbose mode)
    if not arg.verbose:
        if toppeVersion > 5:
            system('rm toppeN.entry seqstamp.txt modules.txt scanloop.txt cores.txt')
        else:
            system('rm toppeN.entry seqstamp.txt modules.txt scanloop.txt')

        for p in range(1, ceq.nParentBlocks + 1):
            system(f'rm {modFiles[p-1]})

    print(f'Sequence file {ofname} ready for execution on GE scanners')
                   

