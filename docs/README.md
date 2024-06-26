# Implementation Notes

Pulseq to GE conversion consists of two main steps:

1. **seq2ceq**: this is an intermediate conversion which identifies the main sequence modules (*Parent Blocks*) and the sequence loop (i.e., module ordering through the sequence and their scaling, phase / frequency modulation, etc...).
2. **ceq2ge**: this is the actual writing step, where the intermediate representation is written in a set of files which are used by GEHC interpreter (TOPPE) to execute the sequence.

## seq2ceq

The [Pulseq format](https://pulseq.github.io/specification.pdf) consists of the following hierarchical structure:

- **[BLOCKS]**: this is the main loop. The ``<id>``-th row is written as  ``<duration><rf><gx><gy><gz><adc><ext>`` and describes a specific event. The ``<rf><gx><gy><gz><adc><ext>`` columns represent the row indexes of **[RF]**, **[GRADIENTS]**/**[TRAPS]** (for *x*, *y*, *z*, respectivelt), **[ADC]** and **[EXTENSION]** events executed at each sequence step.

- **[RF]**, **[GRADIENTS]**/**[TRAPS]**, **[ADC]**,**[EXTENSION]** : each entry describe a specific RF pulse, gradient, ADC and extension event. Specifically:
  
  - **[RF]**: this is the matrix describing RF events.  The ``<id>``-th row is written as  ``<amp><mag_id><phase_id><time_id><delay><freq><phase>``. The ``<mag_id><phase_id>`` columns (IDs identifying the peak-normalized magnitude / phase of RF pulse) uniquely identify the RF shape, while the other columns are event-specific parameters such as RF peak amplitude, frequency and phase offset.
  - **[GRADIENTS]**: this is the matrix describing arbitrary gradient events.  The ``<id>``-th row is written as  ``<amp><shape_id><time_id><delay>``.  The ``<shape_id>`` column (ID identifying the peak-normalized gradient waveform) uniquely identify the GRAD shape, while the other columns are event-specific parameters such as the peak gradient amplitude.
  - **[TRAPS]**: this is the matrix describing trapezoidal gradient events.  The ``<id>``-th row is written as  ``<amp><rise><flat><fall><delay>``.  The ``<rise><flat><fall><delay>`` columns (trapeze ramp-up, plateau and ramp-down durations and delay with respect block start) uniquely identify the TRAP shape, while the other columns are event-specific parameters such as the peak gradient amplitude.
  - **[ADC]** this is the matrix describing ADC events.  The ``<id>``-th row is written as  ``<num><dwell><delay><freq><phase>``. The ``<num><dwell><delay>`` columns (number of ADC samples, raster time and delay with respect block start) uniquely identify the ADC event, while the other columns are event-specific parameters such as ADC frequency and phase offset.
  
  The other parameters uniquely identifying an event are the ``<duration>`` (column 1 in **[BLOCK]** matrix), the presence or absence of a trigger pulse and (experimental) the gradient rotation matrix (both contained in ``<ext>`` column, see [Pulseq C++ implementation](https://github.com/pulseq/pulseq/tree/master/src))

​		If a specific type of event is absent, the corresponding column value in the **[BLOCKS]** row is 0. A block 		row with no events (all zeroes but duration column) is a *pure delay* event.

- **[SHAPES]**: this contains the shapes referred my *mag_id*, *time_id*, *shape_id* in **[RF]**, **[GRADIENTS]**/**[TRAPS]**.

This routine (``src/pypulceq/_seq2ceq.py``) 

