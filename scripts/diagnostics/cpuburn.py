"""Can this host actually exceed ~3.6 cores, or is that a hard ceiling?

Our RL jobs plateau near 3.6 cores no matter how many run. That is either
(a) a hard CPU cap on the box, in which case adding jobs only redistributes,
or (b) our workload being latency-bound, in which case genuine headroom exists.
A pure compute burner distinguishes the two.
"""
import multiprocessing as mp, os, time

def burn(seconds):
    import numpy as np
    a = np.random.rand(256, 256)
    b = np.random.rand(256, 256)
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        a = (a @ b) * 1.0000001
        if a[0, 0] > 1e8 or a[0, 0] < 1e-8:
            a = np.random.rand(256, 256)
        n += 1
    return n

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    for nproc in (2, 4, 8):
        t0 = time.time()
        with mp.Pool(nproc) as p:
            res = p.map(burn, [12] * nproc)
        el = time.time() - t0
        print("  %d burner procs: %d matmuls in %.1fs -> %.0f matmul/s aggregate"
              % (nproc, sum(res), el, sum(res) / el))
