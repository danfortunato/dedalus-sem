"""Build and plot concrete DtN matrices."""

import time
import numpy as np
from scipy import linalg
from poisson_rectangle import PoissonRectangle
import matplotlib.pyplot as plt


# Parameters
Nx = 32
Ny = 32
Lx = 1
Ly = 1
dtype = np.float64

# Solver
solver = PoissonRectangle(Nx, Ny, Lx, Ly, dtype)
f = 0
uL = np.zeros((1, Ny))
uR = np.zeros((1, Ny))
uT = np.zeros((Nx, 1))
uB = np.zeros((Nx, 1))

# Build grid DtN column by column
# Clockwise starting from top
start_time = time.time()
print()
print('Building grid DtN matrix')
cols = []
print('  top')
for nx in range(Nx):
    uT[nx, 0] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'g')
    uT[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel()[::-1], duB.ravel()[::-1], duL.ravel())))
print('  right')
for ny in reversed(range(Ny)):
    uR[0, ny] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'g')
    uR[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel()[::-1], duB.ravel()[::-1], duL.ravel())))
print('  bottom')
for nx in reversed(range(Nx)):
    uB[nx, 0] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'g')
    uB[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel()[::-1], duB.ravel()[::-1], duL.ravel())))
print('  left')
for ny in range(Ny):
    uL[0, ny] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'g')
    uL[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel()[::-1], duB.ravel()[::-1], duL.ravel())))
dtn_g = np.array(cols).T
end_time = time.time()
print('Time: %.2f sec' %(end_time - start_time))

# Check nullity
svdvals = linalg.svdvals(dtn_g)
nullity = np.sum(svdvals < 1e-10)
print('Grid DtN nullity:', nullity, '/', 2*(Nx+Ny))
print()

# Build grid DtN column by column
# Clockwise starting from top
start_time = time.time()
print('Building coeff DtN matrix')
cols = []
print('  top')
for nx in range(Nx):
    uT[nx, 0] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'c')
    uT[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
print('  right')
for ny in range(Ny):
    uR[0, ny] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'c')
    uR[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
print('  bottom')
for nx in range(Nx):
    uB[nx, 0] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'c')
    uB[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
print('  left')
for ny in range(Ny):
    uL[0, ny] = 1
    duL, duR, duT, duB = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, 'c')
    uL[:] = 0
    cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
dtn_c = np.array(cols).T
end_time = time.time()
print('Time: %.2f sec' %(end_time - start_time))

# Check nullity
svdvals = linalg.svdvals(dtn_c)
nullity = np.sum(svdvals < 1e-10)
print('Coeff DtN nullity:', nullity, '/', 2*(Nx+Ny))
print()

# Plot DtN
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(dtn_g, interpolation='nearest')
plt.title('Grid DtN map')
plt.subplot(122)
plt.imshow(dtn_c, interpolation='nearest')
plt.title('Coeff DtN map')
plt.tight_layout()
plt.savefig('dtn_matrices.pdf')

