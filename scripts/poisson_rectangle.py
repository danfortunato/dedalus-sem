"""PoissonRectangle class definition and tests."""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers
from dedalus.tools.cache import CachedMethod
from dedalus.libraries.matsolvers import matsolvers


# Helper functions
top_identity = lambda m, n: sp.eye(m, n)
bottom_identity = lambda m, n: sp.eye(m, n, n-m)


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
    **kw
        Other keywords passed to Dedalus solver.
    """

    def __init__(self, Nx, Ny, Lx, Ly, dtype, **kw):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dtype = dtype
        self.N = Nx * Ny
        self.R = 2*Nx + 2*Ny
        self.M = self.N + self.R
        # Bases
        self.c = c = coords.CartesianCoordinates('x', 'y')
        self.d = d = distributor.Distributor((c,))
        self.xb = xb = basis.ChebyshevT(c.coords[0], Nx, bounds=(0, Lx))
        self.yb = yb = basis.ChebyshevT(c.coords[1], Ny, bounds=(0, Ly))
        self.x = x = xb.local_grid(1)
        self.y = y = yb.local_grid(1)
        xb2 = xb._new_a_b(1.5, 1.5)
        yb2 = yb._new_a_b(1.5, 1.5)
        # Forcing
        self.f = f = field.Field(name='f', dist=d, bases=(xb2, yb2), dtype=dtype)
        # Boundary conditions
        self.uL = uL = field.Field(name='f', dist=d, bases=(yb,), dtype=dtype)
        self.uR = uR = field.Field(name='f', dist=d, bases=(yb,), dtype=dtype)
        self.uT = uT = field.Field(name='f', dist=d, bases=(xb,), dtype=dtype)
        self.uB = uB = field.Field(name='f', dist=d, bases=(xb,), dtype=dtype)
        # Fields
        self.u = u = field.Field(name='u', dist=d, bases=(xb, yb), dtype=dtype)
        self.tx1 = tx1 = field.Field(name='tx1', dist=d, bases=(xb2,), dtype=dtype)
        self.tx2 = tx2 = field.Field(name='tx2', dist=d, bases=(xb2,), dtype=dtype)
        self.ty1 = ty1 = field.Field(name='ty1', dist=d, bases=(yb2,), dtype=dtype)
        self.ty2 = ty2 = field.Field(name='ty2', dist=d, bases=(yb2,), dtype=dtype)
        # Problem
        Lap = lambda A: operators.Laplacian(A, c)
        self.problem = problem = problems.LBVP([u, tx1, tx2, ty1, ty2])
        problem.add_equation((Lap(u), f))
        problem.add_equation((u(y=Ly), uT))
        problem.add_equation((u(x=Lx), uR))
        problem.add_equation((u(y=0), uB))
        problem.add_equation((u(x=0), uL))
        # Solver
        self.solver = solver = solvers.LinearBoundaryValueSolver(problem,
            bc_top=False, tau_left=False, store_expanded_matrices=False, **kw)
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
            # Remember L is left preconditoined but not right preconditioned
            L[-7, Nx*Ny+2*Nx+0*Ny+Ny-2] = 1 # Right -2
            L[-5, Nx*Ny+2*Nx+0*Ny+Ny-1] = 1 # Right -1
            L[-3, Nx*Ny+2*Nx+1*Ny+Ny-2] = 1 # Left -2
            L[-1, Nx*Ny+2*Nx+1*Ny+Ny-1] = 1 # Left -1
        solver.subproblems[0].L_min = L.tocsr()
        # Neumann operators
        ux = operators.Differentiate(u, c.coords[0])
        uy = operators.Differentiate(u, c.coords[1])
        self.duL = - ux(x='left')
        self.duR = ux(x='right')
        self.duT = uy(y='right')
        self.duB = - uy(y='left')
        # Neumann matrix
        duT_mat = self.duT.expression_matrices(solver.subproblems[0], vars=[u])[u]
        duR_mat = self.duR.expression_matrices(solver.subproblems[0], vars=[u])[u]
        duB_mat = self.duB.expression_matrices(solver.subproblems[0], vars=[u])[u]
        duL_mat = self.duL.expression_matrices(solver.subproblems[0], vars=[u])[u]
        self.interior_to_neumann_matrix = sp.vstack([duT_mat, duR_mat, duB_mat, duL_mat], format='csr')

    def set_interior_forcing(self, f):
        """
        Set interior forcing data on the grid.

        Parameters
        ----------
        f : number or ndarray
            Interior forcing data on the grid. Must be broadcastable to shape (Nx, Ny).
        """
        self.f['g'] = f

    def dirichlet_to_interior_naive(self, uL, uR, uT, uB, layout):
        """
        Produce interior solution given Dirichlet data.

        Parameters
        ----------
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
        self.uL[layout] = uL
        self.uR[layout] = uR
        self.uT[layout] = uT
        self.uB[layout] = uB
        self.solver.solve()
        return self.u[layout]

    def interior_to_neumann_naive(self, u, layout):
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
        duL = self.duL.evaluate()
        duR = self.duR.evaluate()
        duT = self.duT.evaluate()
        duB = self.duB.evaluate()
        return duL[layout], duR[layout], duT[layout], duB[layout]

    def dirichlet_to_neumann_naive(self, uL, uR, uT, uB, layout):
        """
        Produce Neumann data given Dirichlet data.

        Parameters
        ----------
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
        u = self.dirichlet_to_interior_naive(uL, uR, uT, uB, layout)
        return self.interior_to_neumann_naive(u, layout)

    def build_operators_naive(self, layout, verbose=False):
        """
        Build solution and DtN operators.

        Parameters
        ----------
        layout : ('g', 'c')
            Layout for input and output data: grid ('g') or coefficient ('c') values.

        Returns
        -------
        sol : ndarray
            Dirichlet-to-solution matrix. Shape (Nx*Ny, 2*Nx+2*Ny).
        dtn : ndarray
            Dirichlet-to-Neumann matrix. Shape (2*Nx+2*Ny, 2*Nx+2*Ny).
        """
        Nx, Ny = self.Nx, self.Ny
        uL = np.zeros((1, Ny), dtype=self.dtype)
        uR = np.zeros((1, Ny), dtype=self.dtype)
        uT = np.zeros((Nx, 1), dtype=self.dtype)
        uB = np.zeros((Nx, 1), dtype=self.dtype)
        sol_cols = []
        dtn_cols = []
        if verbose:
            print('  traversing top')
        for nx in range(self.Nx):
            uT[nx, 0] = 1
            u = self.dirichlet_to_interior_naive(uL, uR, uT, uB, layout)
            sol_cols.append(u.ravel().copy())
            duL, duR, duT, duB = self.interior_to_neumann_naive(u, layout)
            dtn_cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
            uT[:] = 0
        if verbose:
            print('  traversing right')
        for ny in range(Ny):
            uR[0, ny] = 1
            u = self.dirichlet_to_interior_naive(uL, uR, uT, uB, layout)
            sol_cols.append(u.ravel().copy())
            duL, duR, duT, duB = self.interior_to_neumann_naive(u, layout)
            dtn_cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
            uR[:] = 0
        if verbose:
            print('  traversing bottom')
        for nx in range(Nx):
            uB[nx, 0] = 1
            u = self.dirichlet_to_interior_naive(uL, uR, uT, uB, layout)
            sol_cols.append(u.ravel().copy())
            duL, duR, duT, duB = self.interior_to_neumann_naive(u, layout)
            dtn_cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
            uB[:] = 0
        if verbose:
            print('  traversing left')
        for ny in range(Ny):
            uL[0, ny] = 1
            u = self.dirichlet_to_interior_naive(uL, uR, uT, uB, layout)
            sol_cols.append(u.ravel().copy())
            duL, duR, duT, duB = self.interior_to_neumann_naive(u, layout)
            dtn_cols.append(np.concatenate((duT.ravel(), duR.ravel(), duB.ravel(), duL.ravel())))
            uL[:] = 0
        sol = np.array(sol_cols).T
        dtn = np.array(dtn_cols).T
        return sol, dtn

    @CachedMethod
    def _setup_schur(self, matsolver=None):
        Nx, Ny = self.Nx, self.Ny
        N, M = self.N, self.M
        # Build partition matrices
        select_eqn_rows = top_identity(N, M)
        select_var_cols = top_identity(N, M)
        drop_high_modes = sp.kron(top_identity(Nx-2, Nx), top_identity(Ny-2, Ny))
        drop_low_modes = sp.kron(bottom_identity(Nx-2, Nx), bottom_identity(Ny-2, Ny))
        SL0 = drop_high_modes @ select_eqn_rows
        SL1 = sp.identity(M) - SL0.T @ SL0
        SL1 = SL1[SL1.getnnz(1) > 0]
        SR0 = drop_low_modes @ select_var_cols
        SR1 = sp.identity(M) - SR0.T @ SR0
        SR1 = SR1[SR1.getnnz(1) > 0]
        SL0 = SL0.tocsr()
        SL1 = SL1.tocsr()
        SR0 = SR0.tocsc()
        SR1 = SR1.tocsc()
        # Compute partitions
        L = self.solver.subproblems[0].L_min.tocsr()
        PR = self.solver.subproblems[0].pre_right.tocsr()
        L = PR @ L
        L.data[np.abs(L.data) < 1e-6] = 0
        L.eliminate_zeros()
        L00 = SL0 @ L @ SR0.T
        L01 = SL0 @ L @ SR1.T
        L10 = SL1 @ L @ SR0.T
        L11 = SL1 @ L @ SR1.T
        # Schur setup
        if matsolver is None:
            matsolver = self.solver.matsolver
        if isinstance(matsolver, str):
            matsolver = matsolvers[matsolver.lower()]
        L00_LU = matsolver(L00)
        L00_inv = L00_LU.solve
        L00_inv_L01 = L00_inv(L01.A)
        L11_comp = L11.A - L10 @ L00_inv_L01
        L11_comp_LU = linalg.lu_factor(L11_comp)
        L11_comp_inv = lambda A, LU=L11_comp_LU: linalg.lu_solve(LU, A, check_finite=False)
        return SL0, SL1, SR0, SR1, L00, L01, L10, L11, L00_inv, L00_inv_L01, L11_comp_inv

    def dirichlet_to_interior_vectorized(self, RHS, homogeneous=False, use_schur_inv=True, **kw):
        """
        Produce interior solution given interior forcing and Dirichlet data.

        Parameters
        ----------
        RHS : ndarray
            Vectors of interior forcing and Dirichlet coefficients stacked as (F, uT, uR, uB, uL). Shape (Nx*Ny+2*Nx+2*Ny, P).
        homogeneous : bool, optional
            Whether to assume F=0 for performance (default: False).
        use_schur_inv : True, optional
            Whether to use inverse of or solve Schur factor when reconstructing solution (default: True).
        **kw
            Other keywords passed to self._setup_schur.

        Returns
        -------
        U : ndarray
            Interior solution vectors. Shape (Nx*Ny, P).
        """
        SL0, SL1, SR0, SR1, L00, L01, L10, L11, L00_inv, L00_inv_L01, L11_comp_inv = self._setup_schur(**kw)
        # Form Schur complement RHS
        if homogeneous:
            F_comp = SL1 @ RHS
        else:
            F0 = SL0 @ RHS
            F1 = SL1 @ RHS
            L00_inv_F0 = L00_inv(F0)
            F_comp = F1 - L10 @ L00_inv_F0
        # Solve Schur complement
        if sp.isspmatrix(F_comp):
            F_comp = F_comp.A
        y = L11_comp_inv(F_comp)
        # Solve back for x
        if use_schur_inv:
            z = L00_inv_L01 @ y
        else:
            z = L00_inv(L01 @ y)
        if homogeneous:
            x = - z
        else:
            x = L00_inv_F0 - z
        # Return interior solution
        X = SR0.T @ x + SR1.T @ y
        return X[:self.N]

    def interior_to_neumann_vectorized(self, U, **kw):
        """
        Produce Neumann data given interior solution.

        Parameters
        ----------
        U : ndarray
            Interior solution vectors. Shape (Nx*Ny, P).

        Returns
        -------
        dU : ndarray
            Vectors of Neumann coefficients stacked as (duT, duR, duB, duL). Shape (2*Nx+2*Ny, P).
        """
        return self.interior_to_neumann_matrix @ U

    def build_operators_vectorized(self, homogeneous=False, **kw):
        """
        Build solution and DtN operators.

        Parameters
        ----------
        homogeneous : bool, optional
            Whether to assume F=0 for performance (default: False).
        **kw
            Other keywords passed to self.dirichlet_to_interior_vectorized.

        Returns
        -------
        sol : ndarray
            Dirichlet-to-solution matrix. Shape (Nx*Ny, 2*Nx+2*Ny).
        dtn : ndarray
            Dirichlet-to-Neumann matrix. Shape (2*Nx+2*Ny, 2*Nx+2*Ny).
        """
        N, M, R = self.N, self.M, self.R
        if homogeneous:
            RHS = bottom_identity(M, R)
        else:
            raise NotImplementedError()
        # Solve interior
        sol = self.dirichlet_to_interior_vectorized(RHS, homogeneous=homogeneous, **kw)
        # Solve Neumann
        dtn = self.interior_to_neumann_vectorized(sol)
        return sol, dtn


if __name__ == "__main__":

    print()
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
    print("  Solver condition number:", np.linalg.cond(L.A))
    # Forcing
    Kx = 2 * np.pi / Lx
    Ky = 2 * np.pi / Ly
    f = - (Kx**2 + Ky**2) * np.sin(Kx*x) * np.sin(Ky*y)
    # Boundary data
    uL = uR = uT = uB = 0
    # Test solution
    solver.set_interior_forcing(f)
    u = solver.dirichlet_to_interior_naive(uL, uR, uT, uB, layout='g')
    u_true = np.sin(Kx*x) * np.sin(Ky*y)
    u_error = np.max(np.abs(u - u_true))
    print('  Interior max error:', u_error)
    du = solver.dirichlet_to_neumann_naive(uL, uR, uT, uB, layout='g')
    duL_true = - Kx * np.sin(Ky*y)
    duR_true = + Kx * np.sin(Ky*y)
    duT_true = + Ky * np.sin(Kx*x)
    duB_true = - Ky * np.sin(Kx*x)
    du_true = [duL_true, duR_true, duT_true, duB_true]
    du_error = [np.max(np.abs(dui - dui_true)) for dui, dui_true in zip(du, du_true)]
    print('  Neumann max error:', np.max(du_error))
    print()

