# Implementation Notes

Pulseq to GE conversion consists of two main steps:

1. **seq2ceq**: this is an intermediate conversion which identifies the main sequence modules (*Parent Blocks*) and the sequence loop (i.e., module ordering through the sequence and their scaling, phase / frequency modulation, etc...).
2. **ceq2ge**: this is the actual writing step, where the intermediate representation is written in a set of files which are used by GEHC interpreter (TOPPE) to execute the sequence.

## seq2ceq

The [Pulseq format](https://pulseq.github.io/specification.pdf) consists of the following hierarchical structure:

- **[BLOCKS]**: this is the main loop, each row describing a specific event and containing the indexes (as in row-index) of **[RF]**, **[GRADIENTS]**/**[TRAPS]** (for *Gx*, *Gy*, *Gz*), **[ADC]** and **[EXTENSION]** executed at each sequence step.
- **[RF]**, **[GRADIENTS]**/**[TRAPS]**, **[ADC]**,**[EXTENSION]** : each entry describe a specific RF pulse, gradient, ADC and extension event. Specifically:
  - **[RF]** contains *mag_id* / *phase_id* (unique IDs identifying the peak-normalized magnitude / phase of RF pulse), and a set of event-specific parameters such as *amp*, *freq*, *phase* (RF peak amplitude, frequency and phase offset).
  - **[GRADIENTS]** contains *shape_id* (unique ID identifying the peak-normalized gradient waveform), and a set of event-specific parameters such as *amp* (peak gradient amplitude).
  - **[TRAPS]** contains *rise*, *flat*, *fall* (unique tuple identifying a trapezoid by duration of ramp up, flat and ramp down portions), and a set of event-specific parameters such as *amp* (flat portion amplitude).
  - **[ADC]** contains the ADC duration and delay with respect the start of the event .

​		If a specific type of event is absent, the corresponding column value in the **[BLOCKS]** row is 0.

- **[SHAPES]**: this contains the shapes referred my *mag_id*, *time_id*, *shape_id* in **[RF]**, **[GRADIENTS]**/**[TRAPS]**.

This routine (``src/pypulceq/_seq2ceq.py``) 

