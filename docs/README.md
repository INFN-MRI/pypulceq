# Implementation Notes

This document represents a brief overview on the **seq2ge** implementation. It includes a short review of the Pulseq format and the key elements of the corresponding GE-specific representation, as well as a high-level description of the algorithms used for the conversion and the repository organization.

## Background

The [Pulseq format](https://pulseq.github.io/specification.pdf) consists of the following hierarchical structure:

- **[BLOCKS]**: this is the main loop. The ``<id>``-th row is written as  ``<duration><rf><gx><gy><gz><adc><ext>`` and describes a specific event. The ``<rf><gx><gy><gz><adc><ext>`` columns represent the row indexes of **[RF]**, **[GRADIENTS]**/**[TRAPS]** (for *x*, *y*, *z*, respectively), **[ADC]** and **[EXTENSION]** events executed at each sequence step.

- **[RF]**, **[GRADIENTS]**/**[TRAPS]**, **[ADC]**,**[EXTENSION]** : each entry describe a specific RF pulse, gradient, ADC and extension event. Specifically:

  - **[RF]**: this is the matrix describing RF events.  The ``<id>``-th row is written as  ``<amp><mag_id><phase_id><time_id><delay><freq><phase>``. The ``<mag_id><phase_id>`` columns (IDs identifying the peak-normalized magnitude / phase of RF pulse) uniquely identify the RF shape, while the other columns are event-specific parameters such as RF peak amplitude, frequency and phase offset.
  - **[GRADIENTS]**: this is the matrix describing arbitrary gradient events.  The ``<id>``-th row is written as  ``<amp><shape_id><time_id><delay>``.  The ``<shape_id>`` column (ID identifying the peak-normalized gradient waveform) uniquely identify the GRAD shape, while the other columns are event-specific parameters such as the peak gradient amplitude.
  - **[TRAPS]**: this is the matrix describing trapezoidal gradient events.  The ``<id>``-th row is written as  ``<amp><rise><flat><fall><delay>``.  The ``<rise><flat><fall><delay>`` columns (trapeze ramp-up, plateau and ramp-down durations and delay with respect block start) uniquely identify the TRAP event, while the other columns are event-specific parameters such as the peak gradient amplitude.
  - **[ADC]** this is the matrix describing ADC events.  The ``<id>``-th row is written as  ``<num><dwell><delay><freq><phase>``. The ``<num><dwell><delay>`` columns (number of ADC samples, raster time and delay with respect block start) uniquely identify the ADC event, while the other columns are event-specific parameters such as ADC frequency and phase offset.

  The other parameters uniquely identifying an event are the ``<duration>`` (column 1 in **[BLOCK]** matrix), the presence or absence of a trigger pulse and (experimental) the gradient rotation matrix (both contained in ``<ext>`` column, see [Pulseq C++ implementation](https://github.com/pulseq/pulseq/tree/master/src)). If a specific type of event is absent, the corresponding column value in the **[BLOCKS]** row is 0. A block row with no events (all zeroes but duration column) is a *pure delay* event.

- **[SHAPES]**: this contains the shapes referred by ``<mag_id><phase_id><shape_id><time_id>`` in **[RF]**, **[GRADIENTS]**/**[TRAPS]**; each ``<mag_id>`` and ``<shape_id>`` is normalized with respect to the peak amplitude.

The GE-specific implementation of the Pulseq format needs to identify a set of **Parent Blocks** (collection of unique Pulseq events) and the **scanloop** matrix (with *nevents* rows), containing the event-specific RF/gradient amplitudes, frequency and phase offset, etc. In addition, "groups" of blocks always executed together (**segments**)  can be specifed to optimize execution on the scanner. These segments can be reused throughout the sequence as long as its waveform shapes and duration remain constant, while the specific amplitude, frequency/phase of the individual blocks composing a segment can be dynamically adjusted during the experiment (following **scanloop**).

## Implementation Details

Pulseq to GE conversion consists of two main steps:

1. **seq2ceq**: this is an intermediate conversion which identifies the main sequence modules (**Parent Blocks**) and the sequence loop (**scanloop**, i.e., module ordering through the sequence and their scaling, phase / frequency modulation, etc...).
2. **ceq2ge**: this is the actual writing step, where the intermediate representation is written in a set of files which are used by GEHC interpreter (TOPPE) to execute the sequence.

### seq2ceq

This routine (``src/pypulceq/_seq2ceq.py``) identifies 1) the **Parent Blocks**, 2) **scanloop** and 3) **segments**. Specifically, we have the following steps:

- Looping over ``seq`` structure and parsing of the **scanloop** using high-level ``seq.get_block()`` routine:

  - ``<rf><amp>``: amplitude of RF event (if present).
  - ``<rf><phase>``: phase offset (e.g., for spoiling / phase cycling) of RF event (if present).
  - ``<rf><freq>``: frequency offset (e.g., for slice selection) of RF event (if present).

  - ``<adc><phase>``: phase offset (e.g., for demodulation) of ADC event (if present).

  - ``<gi><amp>``: amplitude of gradient event along the *i-th* axis(if present).
  - ``<duration>``: block duration.
  - presence / absence of trigger (``<hastrigger>``).

  In addition, we store the cumulative ``adc_count``, the presence or absence of a ``rotation`` event (and the corresponding ``rot_matrix``, if it is present), and the ``TRID`` label, if it is present.

- We now perform low level access to ``seq`` ``block_events.values()``, ``rf_library.values()``, ``grad_library.values()``and ``adc_library.values()``, we extract the unique identifiers of ``RF``, ``grad`` and ``adc`` events (see **Background** section) and we build a matrix with ``nevents`` rows and the following columns: `` [<duration>, <rf><mag_id>, <rf><phase_id>, (grad)_{x, y, z}, <adc><num>, <adc><dwell>, <adc><delay>, <hastrigger>]``. Here,`` (grad)_{x, y, z}`` are three (for x, y and z axes) tuples of 4 elements, whose content depends on the ``grad.type`` (inferred from the length of the row in ``grad_library.values()``):

  - **trapezoidal gradients**: ``(<rise>, <flat>, <fall>, <delay>)``.
  - **arbitrary gradients**: ``(<shape_id>, 0, 0, 0)``. Notice that ``(<flat>, <fall>, <delay>)`` are strictly positive integer, so an arbitrary gradient event cannot be confused with a trapezoidal gradient event.

- We identify the unique non-zero rows (zero rows marks pure delay events), the indexes of their first occurrences and we build an ``(nevents,)`` shaped array with the ID of each unique row using the efficient ``np.unique`` function

- We now parse from ``seq`` structure the unique (parent) blocks using the high-level ``seq.get_block()`` function.

- We set the amplitude of parent blocks to maximum by performing the following:

  - for each parent block, we extract (and decompress) the normalized shapes using ``<mag_id><phase_id><shape_id>``.
  - for each parent block, we extract the corresponding **scanloop** rows and the ``[<rf><amp>, <gx><amp>, <gy><amp>, <gz><amp>]`` columns; we then find the maximum of each column of these sub-matrices and set the amplitudes to this value (``<rf><amp>``; ``<grad><amp>`` for **RF** and **TRAP**; ``<grad><waveforms> *= value`` for **GRADIENTS**).

- We finally identify the segments. This can be done using ``TRID`` label; if not provided, we attempt to identify them automatically (see **Segment Identification** section).

#### Automatic Segments identification

If ``TRID`` is not provided, we attempt to identify repeating groups of blocks automatically. This is based on the following assumptions:

- Most sequences will consist of a preparation block (e.g., TG calibration, dummy scans for steady state, coil sensitivity) and a longer main loop sequence.
- Preparation block will be played at the beginning of the sequence.

For this reason, we proceed as follows:

1. Extract the  ``block_id`` array and use  ``hasrot`` array to change the sign of rotated modules.
2. Starting from the beginning of the sequence, we attempt to identify periodic sequences following [this reference](https://stackoverflow.com/questions/29481088/how-can-i-tell-if-a-string-repeats-itself-in-python). If no periodic sequences are detected, repeat the search starting from the second element (i.e., assume we had a preparation block at the beginning). Iterate until a periodic sequence is found or the end of the sequence is reached. Function will then return ``(dummy_block, main_loop)`` tuple.
3. If the sequence does not have preparation blocks, ``dummy_block`` will be empty. Otherwise, perform another level of recursion on ``dummy_block`` to split TG calibration and dummy blocks for steady-state preparation.
4. Flatten the ``dummy_block`` (e.g., ``[[TGcal], [SteadyStatePrep]] -> [*TGcal, *SteadyStatePrep]``) and then append at the beginning of main loop, returning a single ``list`` whose elements are ``lists`` of the ``block_ids`` composing each segment, e.g., ``segments = [[0, 1], [2, 3, 4]]`` corresponds to two segments, one containing **parent blocks** 0,1 and the other containing 2, 3, 4.
5. While amplitude, phase and frequency of the blocks composing a segment can be independently modulated during the sequence, the segment can be rotated only as a whole. For this reason, we split rotated segments in separated segments, e.g., ``segments = [[0, 1], [2, -3, 4]]->[[0, 1], [2], [-3], [4]]``. Finally, we take the magnitude of each segment definition to get rid of the sign. Here, we make the (somewhat?) reasonable assumption that waveforms with different rotation matrices (e.g., different spiral readouts) are separated by non rotated blocks (i.e., RF pulses or spoiler gradients). This would work for multi-echo GRE acquisitions, as long as each echo share the same readout, but not if multiple echoes undergo different rotations (e.g., [https://archive.ismrm.org/2020/0616.html]()). In this case, user would have to either design the base consecutive readouts explicitly, or manually specify the segments using ``TRID`` label.
6. Finally, we loop over segments definition and search all the occurrences of the sequence in ``block_id`` array to build the ``core`` column of scanloop.

Notice that all these steps could be easily ported in the MATLAB implementation.

### ceq2ge

This second step pretty much follows the [original MATLAB implementation](https://github.com/HarmonizedMRI/PulCeq/blob/main/matlab/ceq2ge.m), and briefly consists of:

- Looping over **parent blocks**, create explicit waveforms from TRAP gradients and interpolate all the waveforms to target raster time.
- Identify the peak B1 across each block and the readout block (i..e, the one containing ADC event).
- Write **parent_blocks** as **moduleN.mod**, the list of modules in **modules.txt**,  the segment definitions **cores.txt**, the loop dynamics in **scanloop.txt**, the **seqstamp.txt** (containing results of ``preflightcheck``) and the TOPPE entry **toppeN.entry**

Here, we made two minor changes:

1. SAR calculation is performed in ``preflightcheck`` by calculating all the 10s energy depositions values in parallel using Python [multiprocess](https://pypi.org/project/multiprocess/) module and the maximum is calculated afterwards. This would not be very efficient in MATLAB as its ``parpool`` spawning is much slower than Python's.
2. We let the user specify a files location on the scanner for **toppeN.entry**, the default being the same as MATLAB implementation. This way, sequence files can be stored in custom location and it is sufficient to create a soft-link to **toppeN.entry** in ``/usr/g/research/pulseq/v6/seq2ge/``.

## Benchmark

Runtimes for conversion of a Cartesian 3D GRE (256 x 256 x 150 voxels, 10 TG calibration scans, 256 steady-state prep pulses) are the following:

```
MATLAB: 699 [s]
Python: 136 [s]
```

The two conversions were performed on the same machine. The Python implementation uses the parallelized ``preflightcheck`` version: a previous iteration with serialized ``preflightcheck`` took approximately `200 [s]`.

Please notice that runtimes may vary greatly depending on the hardware configuration used for testing. The same example executed on another machine (without MATLAB, hence the missing comparison) took approximately `100 [s]`  with serial ``preflightcheck`` and approximately `40 [s]` with the parallelized version.

## Repository organization

The repository is organized as follows:

- ``src/pypulceq/`` contains the source code. Specifically:

  - ``src/pypulceq/_seq2ceq.py`` contains the main ``seq2ceq`` (i.e., step 1 of conversion).
  - ``src/pypulceq/_autosegment.py`` contains automatic segment identification subroutines.

  - ``src/pypulceq/_ceq2ge.py`` contains the main ``ceq2ge`` (i.e., step 2 of conversion).

  - ``src/pypulceq/_interp.py`` contains block conversion subroutines (``trap2ge`` and interpolations).
  - ``src/pypulceq/_toppe`` contains a porting of [TOPPE toolbox](https://github.com/toppeMRI/toppe/tree/main/%2Btoppe), i.e., ``preflightcheck`` and the actual file writing routines.

  - ``src/pypulceq/_seq2ge.py`` contains the main ``seq2ge`` wrapper (i.e., conversion of ``pypulseq.Opts`` to ``_toppe.SystemSpecs`` and the 2 steps of conversion).

  - ``src/pypulceq/_cli.py`` contains the main entrypoint for Command Line Interface.
  - ``src/pypulceq/demo`` contains representative Cartesian and Non-Cartesian PyPulseq design routines.

- ``tests`` contains unit tests for the different subroutines.

- ``examples`` contains an introductory notebook, executable on Google Colab.

- ``docs``: these notes.





