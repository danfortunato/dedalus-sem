"""PoissonRectangle class definition and tests."""

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.config import config

config['matrix construction']['STORE_EXPANDED_MATRICES'] = "False"
config['linear algebra']['MATRIX_FACTORIZER'] = "SuperLUNaturalFactorized"


class PoissonRectangle:
    """
    Solver for Poisson's equation on a rectangle.

    Parameters
    ----------
    Nx, Ny : int
        Solver resolutions (currently must be equal).
    Lx, Ly : float
        Rectangle side lengths.
    dtype : dtype
        Solution dtype.
    """

    def __init__(self, Nx, Ny, Lx, Ly, dtype):
        # Bases
        self.c = c = coords.CartesianCoordinates('x', 'y')
        self.d = d = distributor.Distributor((c,))
        self.xb = xb = basis.ChebyshevT(c.coords[0], Nx, bounds=(0, Lx))
        self.yb = yb = basis.ChebyshevT(c.coords[1], Ny, bounds=(0, Ly))
        self.x = x = xb.local_grid(1)
        self.y = y = yb.local_grid(1)
        # Forcing
        self.f = f = field.Field(name='f', dist=d, bases=(xb, yb), dtype=dtype)
        # Boundary conditions
        self.uL = uL = field.Field(name='f', dist=d, bases=(yb,), dtype=dtype)
        self.uR = uR = field.Field(name='f', dist=d, bases=(yb,), dtype=dtype)
        self.uT = uT = field.Field(name='f', dist=d, bases=(xb,), dtype=dtype)
        self.uB = uB = field.Field(name='f', dist=d, bases=(xb,), dtype=dtype)
        # Fields
        self.u = u = field.Field(name='u', dist=d, bases=(xb, yb), dtype=dtype)
        xb2 = xb._new_a_b(1.5, 1.5)
        yb2 = yb._new_a_b(1.5, 1.5)
        self.tx1 = tx1 = field.Field(name='tx1', dist=d, bases=(xb2,), dtype=dtype)
        self.tx2 = tx2 = field.Field(name='tx2', dist=d, bases=(xb2,), dtype=dtype)
        self.ty1 = ty1 = field.Field(name='ty1', dist=d, bases=(yb2,), dtype=dtype)
        self.ty2 = ty2 = field.Field(name='ty2', dist=d, bases=(yb2,), dtype=dtype)
        # Problem
        Lap = lambda A: operators.Laplacian(A, c)
        self.problem = problem = problems.LBVP([u, tx1, tx2, ty1, ty2])
        problem.add_equation((Lap(u), f))
        problem.add_equation((u(x=0), uL))
        problem.add_equation((u(x=Lx), uR))
        problem.add_equation((u(y=0), uB))
        problem.add_equation((u(y=Ly), uT))
        # Solver
        self.solver = solver = solvers.LinearBoundaryValueSolver(problem)
        # Tau entries
        L = solver.subproblems[0].L_min.tolil()
        # Taus
        for nx in range(Nx):
            L[Ny-1+nx*Ny, Nx*Ny+0*Nx+nx] = 1  # tx1 * Py1
            L[Ny-2+nx*Ny, Nx*Ny+1*Nx+nx] = 1  # tx2 * Py2
        for ny in range(Ny-2):
            L[(Nx-1)*Ny+ny, Nx*Ny+2*Nx+0*Ny+ny] = 1  # ty1 * Px1
            L[(Nx-2)*Ny+ny, Nx*Ny+2*Nx+1*Ny+ny] = 1  # ty2 * Px2
        # BC taus not resolution safe
        if Nx != Ny:
            raise ValueError("Current implementation requires Nx == Ny.")
        else:
            L[-8, Nx*Ny+2*Nx+0*Ny+Ny-2] = 1
            L[-7, Nx*Ny+2*Nx+0*Ny+Ny-1] = 1
            L[-4, Nx*Ny+2*Nx+1*Ny+Ny-2] = 1
            L[-3, Nx*Ny+2*Nx+1*Ny+Ny-1] = 1
        solver.subproblems[0].L_min = L

    def dirichlet_to_interior(self, f, uL, uR, uT, uB, layout):
        """
        Produce interior solution given forcing and Dirichlet data.

        Parameters
        ----------
        f : number or ndarray
            Interior forcing data. Must be broadcastable to shape (Nx, Ny).
        uL, uR : numbers or ndarrays
            Left and right Dirichlet data. Must be broadcastable to shape (1, Ny).
        uT, uB : numbers or ndarrays
            Top and bottom Dirichlet data. Must be broadcastable to shape (Nx, 1).
        layout : ('g', 'c')
            Layout for input and output data: grid ('g') or coefficient ('c') values.

        Returns
        -------
        u : ndarray
            Interior solution data. Shape (Nx, Ny).
        """
        self.f[layout] = f
        self.uL[layout] = uL
        self.uR[layout] = uR
        self.uT[layout] = uT
        self.uB[layout] = uB
        self.solver.solve()
        return self.u[layout]

    def interior_to_neumann(self, u, layout):
        """
        Product Neumann data given interior solution.

        Parameters
        ----------
        u : number or ndarray
            Interior solution data. Must be broadcastable to shape (Nx, Ny).
        layout : ('g', 'c')
            Layout for input and output data: grid ('g') or coefficient ('c') values.

        Returns
        -------
        duL, duR : numbers or ndarrays
            Left and right Neumann data. Shape (1, Ny).
        duT, duB : numbers or ndarrays
            Top and bottom Neumann data. Shape (Nx, 1).
        """
        self.u[layout] = u
        ux = operators.Differentiate(self.u, self.c.coords[0]).evaluate()
        uy = operators.Differentiate(self.u, self.c.coords[1]).evaluate()
        duL = (-ux(x='left')).evaluate()
        duR = ux(x='right').evaluate()
        duT = uy(y='right').evaluate()
        duB = (-uy(y='left')).evaluate()
        return duL[layout], duR[layout], duT[layout], duB[layout]

    def dirichlet_to_neumann(self, f, uL, uR, uT, uB, layout):
        """
        Produce Neumann data given forcing and Dirichlet data.

        Parameters
        ----------
        f : number or ndarray
            Interior forcing data. Must be broadcastable to shape (Nx, Ny).
        uL, uR : numbers or ndarrays
            Left and right Dirichlet data. Must be broadcastable to shape (1, Ny).
        uT, uB : numbers or ndarrays
            Top and bottom Dirichlet data. Must be broadcastable to shape (Nx, 1).
        layout : ('g', 'c')
            Layout for input and output data: grid ('g') or coefficient ('c') values.

        Returns
        -------
        duL, duR : numbers or ndarrays
            Left and right Neumann data. Shape (1, Ny).
        duT, duB : numbers or ndarrays
            Top and bottom Neumann data. Shape (Nx, 1).
        """
        u = self.dirichlet_to_interior(f, uL, uR, uT, uB, layout)
        return self.interior_to_neumann(u, layout)


if __name__ == "__main__":

    print('Test problem: u = sin(2πx) sin(2πy) on [0,1]^2')
    # Parameters
    Nx = 32
    Ny = 32
    Lx = 1
    Ly = 1
    dtype = np.float64
    # Solver
    solver = PoissonRectangle(Nx, Ny, Lx, Ly, dtype)
    x = solver.x
    y = solver.y
    # Check matrix
    L = solver.solver.subproblems[0].L_min
    print("Solver condition number:", np.linalg.cond(L.A))
    # Forcing
    Kx = 2 * np.pi / Lx
    Ky = 2 * np.pi / Ly
    f = - (Kx**2 + Ky**2) * np.sin(Kx*x) * np.sin(Ky*y)
    # Boundary data
    uL = uR = uT = uB = 0
    # Test solution
    u = solver.dirichlet_to_interior(f, uL, uR, uT, uB, layout='g')
    u_true = np.sin(Kx*x) * np.sin(Ky*y)
    u_error = np.max(np.abs(u - u_true))
    print('Interior max error:', u_error)
    du = solver.dirichlet_to_neumann(f, uL, uR, uT, uB, layout='g')
    duL_true = - Kx * np.sin(Ky*y)
    duR_true = + Kx * np.sin(Ky*y)
    duT_true = + Ky * np.sin(Kx*x)
    duB_true = - Ky * np.sin(Kx*x)
    du_true = [duL_true, duR_true, duT_true, duB_true]
    du_error = [np.max(np.abs(dui - dui_true)) for dui, dui_true in zip(du, du_true)]
    print('Neumann max error:', np.max(du_error))

