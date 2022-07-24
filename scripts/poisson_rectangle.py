"""PoissonRectangle class definition and tests."""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg
import dedalus.public as d3
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
        self.R = 2*Nx + 2*Ny + 4
        self.M = self.N + self.R
        # Bases
        self.coords = coords = d3.CartesianCoordinates('x', 'y')
        self.dist = dist = d3.Distributor(coords, dtype=dtype)
        self.xb = xb = d3.ChebyshevT(coords['x'], Nx, bounds=(0, Lx))
        self.yb = yb = d3.ChebyshevT(coords['y'], Ny, bounds=(0, Ly))
        self.x = x = xb.local_grid(1)
        self.y = y = yb.local_grid(1)
        xb2 = xb.derivative_basis(2)
        yb2 = yb.derivative_basis(2)
        # Forcing
        self.f = f = dist.Field(name='f', bases=(xb, yb))
        # Boundary conditions
        self.uL = uL = dist.Field(name='uL', bases=yb)
        self.uR = uR = dist.Field(name='uR', bases=yb)
        self.uB = uB = dist.Field(name='uB', bases=xb)
        self.uT = uT = dist.Field(name='uT', bases=xb)
        # Fields
        self.u = u = dist.Field(name='u', bases=(xb, yb))
        self.tx1 = tx1 = dist.Field(name='tx1', bases=xb2)
        self.tx2 = tx2 = dist.Field(name='tx2', bases=xb2)
        self.ty1 = ty1 = dist.Field(name='ty1', bases=yb2)
        self.ty2 = ty2 = dist.Field(name='ty2', bases=yb2)
        self.t1 = t1 = dist.Field(name='t1')
        self.t2 = t2 = dist.Field(name='t1')
        self.t3 = t3 = dist.Field(name='t1')
        self.t4 = t4 = dist.Field(name='t1')
        # Substitutions
        Lap = d3.Laplacian
        Lift = d3.Lift
        tau_u = (Lift(tx1, yb2, -1) + Lift(tx2, yb2, -2) +
                 Lift(ty1, xb2, -1) + Lift(ty2, xb2, -2))
        tau_T = Lift(t1, xb, -1) + Lift(t2, xb, -2)
        tau_B = Lift(t3, xb, -1) + Lift(t4, xb, -2)
        tau_L = 0
        tau_R = 0
        # Problem
        self.problem = problem = d3.LBVP([u, tx1, tx2, ty1, ty2, t1, t2, t3, t4])
        problem.add_equation((Lap(u) + tau_u, f))
        problem.add_equation((u(x=0) + tau_L, uL))
        problem.add_equation((u(x=Lx) + tau_R, uR))
        problem.add_equation((u(y=0) + tau_B, uB))
        problem.add_equation((u(y=Ly) + tau_T, uT))
        problem.add_equation((tx1(x=0), 0))
        problem.add_equation((tx2(x=0), 0))
        problem.add_equation((tx1(x=Lx), 0))
        problem.add_equation((tx2(x=Lx), 0))
        # Solver
        self.solver = solver = problem.build_solver(store_expanded_matrices=False, **kw)
        # Neumann operators
        ux = d3.Differentiate(u, coords['x'])
        uy = d3.Differentiate(u, coords['y'])
        self.duL = - ux(x=0)
        self.duR = ux(x=Lx)
        self.duB = - uy(y=0)
        self.duT = uy(y=Ly)
        # Neumann matrix
        duL_mat = self.duL.expression_matrices(solver.subproblems[0], vars=[u])[u]
        duR_mat = self.duR.expression_matrices(solver.subproblems[0], vars=[u])[u]
        duB_mat = self.duB.expression_matrices(solver.subproblems[0], vars=[u])[u]
        duT_mat = self.duT.expression_matrices(solver.subproblems[0], vars=[u])[u]
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
        duL, duR : ndarrays
            Left and right Neumann data. Shape (1, Ny).
        duT, duB : ndarrays
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
        duL, duR : ndarrays
            Left and right Neumann data. Shape (1, Ny).
        duT, duB : ndarrays
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

    def pack_boundary_data(self, L, R, T, B):
        """
        Pack boundary data arrays into single vector.

        Parameters
        ----------
        L, R : numbers or ndarrays
            Left and right boundary data. Must be broadcastable to shape (1, Ny).
        T, B : numbers or ndarrays
            Top and bottom boundary data. Must be broadcastable to shape (Nx, 1).

        Returns
        -------
        out : ndarray
            Combined vector of boundary data ordered as (T, R, B, L).
        """
        out = np.zeros(2*self.Nx + 2*self.Ny)
        out[0*Nx+0*Ny:1*Nx+0*Ny] = T.ravel()
        out[1*Nx+0*Ny:1*Nx+1*Ny] = R.ravel()
        out[1*Nx+1*Ny:2*Nx+1*Ny] = B.ravel()
        out[2*Nx+1*Ny:2*Nx+2*Ny] = L.ravel()
        return out


if __name__ == "__main__":

    print()
    print("Test problem: u = sin(2πx) sin(2πy) on [0,1/2]*[0,1]")
    # Parameters
    Nx = 24
    Ny = 32
    Lx = 0.5
    Ly = 1
    dtype = np.float64
    # Solver
    solver = PoissonRectangle(Nx, Ny, Lx, Ly, dtype)
    L = solver.solver.subproblems[0].L_min
    print(f"\n  Solver condition number: {np.linalg.cond(L.A):.3e}")
    # Forcing
    Kx = 2 * np.pi
    Ky = 2 * np.pi
    x = solver.x
    y = solver.y
    f = - (Kx**2 + Ky**2) * np.sin(Kx*x) * np.sin(Ky*y)
    solver.set_interior_forcing(f)
    # Boundary data
    uL = uR = uT = uB = 0
    # Test interior solution
    u_true = np.sin(Kx*x) * np.sin(Ky*y)
    u = solver.dirichlet_to_interior_naive(uL, uR, uT, uB, layout='g')
    print(f"  Interior max error (naive): {np.max(np.abs(u - u_true)):.3e}")
    # Test Neumann solution
    duL_true = - Kx * np.cos(Kx*0) * np.sin(Ky*y)
    duR_true = Kx * np.cos(Kx*Lx) * np.sin(Ky*y)
    duT_true = Ky * np.sin(Kx*x) * np.cos(Ky*Ly)
    duB_true = - Ky * np.sin(Kx*x) * np.cos(Ky*0)
    du_true = [duL_true, duR_true, duT_true, duB_true]
    du_true = solver.pack_boundary_data(*du_true)
    du = solver.dirichlet_to_neumann_naive(uL, uR, uT, uB, layout='g')
    du = solver.pack_boundary_data(*du)
    print(f"  Neumann max error: {np.max(np.abs(du - du_true)):.3e}")
    print()
