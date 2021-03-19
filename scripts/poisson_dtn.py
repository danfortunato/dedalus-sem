"""Build and plot concrete DtN matrices."""

import time
import numpy as np
from scipy import linalg
from poisson_rectangle import PoissonRectangle
import matplotlib.pyplot as plt


# Parameters
Nx = 64
Ny = 64
Lx = 1
Ly = 1
dtype = np.float64
layout = 'c'

# Solver
solver = PoissonRectangle(Nx, Ny, Lx, Ly, dtype)
solver.set_interior_forcing(0)
uL = np.zeros((1, Ny))
uR = np.zeros((1, Ny))
uT = np.zeros((Nx, 1))
uB = np.zeros((Nx, 1))

# Build DtN column by column
# Clockwise starting from top
print()
print(f'Nx = {Nx}, Ny = {Ny}')
print('Building solution & DtN matrices')
start_time = time.time()
sol, dtn = solver.build_operators(layout, verbose=True)
end_time = time.time()
print('Time: %.2f sec' %(end_time - start_time))

# Check nullity
svdvals = linalg.svdvals(dtn)
nullity = np.sum(svdvals < 1e-10)
print('DtN nullity:', nullity, '/', 2*(Nx+Ny))
print()

# Plot DtN
plt.figure(figsize=(6,6))
plt.imshow(dtn, interpolation='nearest')
plt.title('DtN map, layout %s' %layout)
plt.tight_layout()
plt.savefig('dtn_matrix.pdf')

