"""Build and plot concrete DtN matrices."""

import time
import numpy as np
from scipy import linalg
from poisson_rectangle import PoissonRectangle
import matplotlib.pyplot as plt


# Parameters
Nx = 128
Ny = 128
Lx = 1
Ly = 1
dtype = np.float64
layout = 'c'
homogeneous = True
use_schur_inv = True
plot_dtn = False

# Solver
solver = PoissonRectangle(Nx, Ny, Lx, Ly, dtype)
solver.set_interior_forcing(0)

# Build operators
print()
print(f'Nx = {Nx}, Ny = {Ny}')
print('Building solution & DtN matrices')
start_time = time.time()
sol, dtn = solver.build_operators_vectorized(homogeneous=homogeneous, use_schur_inv=use_schur_inv)
end_time = time.time()
print('First time : %.2f sec' %(end_time - start_time))
start_time = time.time()
sol, dtn = solver.build_operators_vectorized(homogeneous=homogeneous, use_schur_inv=use_schur_inv)
end_time = time.time()
print('Second time: %.2f sec' %(end_time - start_time))

# Check nullity
svdvals = linalg.svdvals(dtn)
nullity = np.sum(svdvals < 1e-10)
print('DtN nullity:', nullity, '/', 2*(Nx+Ny))
print()

# Plot DtN
if plot_dtn:
    plt.figure(figsize=(6,6))
    plt.imshow(dtn, interpolation='nearest')
    plt.title('DtN map, layout %s' %layout)
    plt.tight_layout()
    plt.savefig('dtn_matrix.pdf')

