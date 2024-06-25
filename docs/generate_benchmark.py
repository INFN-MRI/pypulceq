"""Generate sequences for WTools sim and timing."""

import time

from scipy.io import savemat

import pypulceq

# cartesian
seq1 = pypulceq.demo.design_gre(fov=(256, 180), mtx=(256, 150), write_seq=True)

# convert
t0 = time.time()
pypulceq.seq2ge("cart_pytoppe", seq1, nviews=256, nslices=150, ignore_segments=True, verbose=True)
t1 = time.time()
print(f"Total elapsed time: {round(t1-t0, 2)} [s]")
savemat("pypulceq_timing.mat", {"t": t1-t0})

# non cartesian
seq2 = pypulceq.demo.design_sos(fov=(256, 180), mtx=(256, 150))
pypulceq.seq2ge("noncart_pytoppe", seq2, nviews=256, nslices=150, verbose=True)

