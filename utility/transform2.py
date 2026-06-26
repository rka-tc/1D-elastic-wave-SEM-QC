
from typing import Dict, List

# -------- CLASSES --------
"""
Spectral Element Hamiltonian Builder (multi-element, GLL-based)
Author: R.K.

This builds the global Hamiltonian for 1D elastic wave simulation using
the Spectral Element Method (SEM) with Legendre–Gauss–Lobatto (GLL) nodes.

"""

import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt


class HamiltonianBuilderSEM:
    """
    Build the Hamiltonian for 1D elastic SEM.
    Supports multi-element domain with mean-valued rho, mu per element.
    """

    def __init__(self, nx=16, order=4, L=1, mu=None, rho=None):
        self.Np = order
        self.Nel = int((nx - 1) / (order - 1))
        self.L = L
        self.nx = nx
        self.rho = rho
        self.mu = mu

        # Set materials
        #self.rho = np.ones(Nel) if rho is None else np.array(rho)
        #self.mu = np.ones(Nel) if mu is None else np.array(mu)


        # precompute reference GLL
        self.xi, self.wi = self._gll_nodes_weights(self.Np)

        self.D_ref = self._compute_derivative_matrix(self.xi)
        
        # compute matrices
        print("Assembling global matrices...")
        self.x, self.m, self.u, self.k = self._assemble_global()
        self.nx = len(self.x)
        self.q, self.h, self.t, self.inv_t, self.sqrt_M = self._mass_scale_and_build_H(self.m, self.k, self.u)

        # build transform mass matrix
        self.sqrt_m = np.block([[self.sqrt_M, np.zeros(self.m.shape)],
                         [np.zeros(self.m.shape), self.sqrt_M]])
        print("Mass matrix sqrt :", self.sqrt_m.shape)
        
        self.inv_sqrt_m = np.linalg.inv(self.sqrt_m)
        print("Mass matrix inv sqrt :", self.inv_sqrt_m.shape)

    # ------------------------ GLL Utilities -----------------------------
    def _gll_nodes_weights(self, N):
        """
        Compute Gauss–Lobatto–Legendre (GLL) nodes and weights.
        Stable, symmetric, and ensures endpoints are exactly ±1.
        """
        if N < 2:
            raise ValueError("GLL requires N >= 2")

        from numpy.polynomial.legendre import legval, legder, legroots

        # Legendre polynomial of degree N-1
        Pn = np.polynomial.legendre.Legendre.basis(N - 1)

        # Interior nodes = roots of (1 - x^2) * P'_{N-1}(x) = 0
        # So we find roots of P'_{N-1}(x)
        dPn = Pn.deriv()
        x_internal = legroots(dPn.coef)

        # Full set of nodes (include endpoints)
        x = np.concatenate(([-1.0], x_internal, [1.0]))
        x = np.sort(np.real_if_close(x))

        # Compute weights: w_i = 2 / [N(N-1) * (P_{N-1}(x_i))^2]
        w = 2.0 / (N * (N - 1) * (legval(x, Pn.coef) ** 2))

        # Guarantee positivity and double precision
        w = np.abs(np.real_if_close(w)).astype(np.float64)

        #-------------plot-----------------------------#
        return x.astype(np.float64), w


    def _compute_derivative_matrix(self, xi):
        """
        Stable derivative matrix at GLL points using Legendre polynomials.
        """
        N = len(xi)
        D = np.zeros((N, N))
        P = np.polynomial.legendre.Legendre.basis(N - 1)

        Pn = P(xi)
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (Pn[i] / Pn[j]) / (xi[i] - xi[j])
            # Diagonal term (stable formulation)
            if xi[i] == -1:
                D[i, i] = -N * (N - 1) / 4
            elif xi[i] == 1:
                D[i, i] = N * (N - 1) / 4
            else:
                D[i, i] = -xi[i] / (2 * (1 - xi[i] ** 2))
        return D



    # ------------------------ Element Matrices --------------------------
    def _element_matrices(self, rho_e, mu_e, dx_e):
        """Local element matrices (mass, stiffness)."""
        N = self.Np
        J = dx_e / 2.0
        #print(J, rho_e, self.wi)
        M_e = rho_e * np.diag(self.wi) * J
        M_einvsqrt = np.diag(1/np.sqrt(M_e.diagonal()))
        E_ehalf = np.sqrt(mu_e * np.eye(N))
        Uref = E_ehalf @ self.D_ref @ M_einvsqrt
        K_e = - self.D_ref.T @ ( mu_e * (np.diag(self.wi) * (2.0 / dx_e)) @ self.D_ref) 
        return M_e, K_e, Uref

    # ------------------------ Global Assembly ---------------------------
    def _assemble_global(self):
        """Assemble global M and K matrices."""
        Nel, Np, L = self.Nel, self.Np, self.L
        dx = L / Nel
        N_global = int(Nel * (Np - 1) + 1)
        M = np.zeros((N_global, N_global))
        Ehalf = np.zeros((N_global, N_global))
        Minvsqrt = np.zeros((N_global, N_global))
        K = np.zeros((N_global, N_global))
        U = np.zeros((N_global, N_global)) 
        dx = L / Nel
        # self.xi has length Np, with self.xi[0] = -1, self.xi[-1] = +1
        x_parts = []
        for e in range(Nel):
            x_left = e * dx
            x_e = x_left + 0.5 * dx * (self.xi + 1.0)   # length Np
            if e == 0:
                x_parts.append(x_e)          # keep all Np nodes
            else:
                x_parts.append(x_e[1:])      # drop duplicated interface node
        x = np.concatenate(x_parts)          # length Nel*(Np-1)+1
        assert len(x) == N_global
        np.save('x_sem.npy', x)

       

        for e in range(Nel):
            idx = np.arange(e * (Np - 1), e * (Np - 1) + Np)
            M_e, K_e, Uref = self._element_matrices(self.rho[e], self.mu[e], dx)
           
            M[np.ix_(idx, idx)] += M_e
            K[np.ix_(idx, idx)] += K_e
            #U[np.ix_(idx, idx)] = Uref
        # K = -1 * K
            

        # # Apply Dirichlet boundary conditions
        x = x[1:-1]
        # M = M[1:-1, 1:-1]
        # U = U[1:-1, 1:-1]
        # K = K[1:-1, 1:-1]


        # 2. Apply Boundary Conditions (truncate M and K)
        M_sub = M[1:-1, 1:-1]
        K_sub = K[1:-1, 1:-1]

        # 3. Mass-Normalize the Stiffness Matrix
        # This accounts for the density (rho) changes
        inv_sqrt_M = np.diag(1.0 / np.sqrt(np.diag(M_sub)))
        K_tilde = inv_sqrt_M @ K_sub @ inv_sqrt_M
        # D = _compute_derivative_matrix(self.xi)

        # 4. Compute the GLOBAL U
        # This accounts for the shear modulus (mu) changes across the whole system
        # vals, vecs = np.linalg.eigh(-K_tilde)
        # vals[vals < 0] = 0  # Numerical stability
        # U = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        U = K_sub @ inv_sqrt_M
        M = M_sub
        K = K_sub
        np.save('M_sem.npy', M)
        np.save('K_sem.npy', K)
        print("K simmetric check (should be close to 0):", np.max(np.abs(K - K.T)))
        print("Mass matrix M :", M.shape)
        print("Stiffness matrix K :", K.shape)
        print("matrix U :", U.shape)
        print("is M diagonal?", np.allclose(M, np.diag(np.diag(M))))

        return x, M, U, K

    # ------------------------ Boundary & Scaling ------------------------
    def _apply_boundary_conditions(self, M, K, U):
        return M[1:-1, 1:-1], K[1:-1, 1:-1], U[1:-1, 1:-1]

    def _mass_scale_and_build_H(self, M, K, U):
        """Mass scaling, Q matrix, and Hermitian H."""
        max_diag_M = np.max(np.diag(M))
        # Scale the matrix M
        #Mscaled = M / max_diag_M
        Mhalf = np.sqrt(M)
        Minvhalf = np.linalg.inv(Mhalf)
        Ktilde = -U.T @ U
        #Ktilde = Minvhalf @ K @ Minvhalf

        N = U.shape[0]
        I = np.eye(N)
        #print("Ktilde matrix :", Ktilde)

        #Transformation matrix T and its inverse
        T = np.block([[U, np.zeros((N, N))],
                      [np.zeros((N, N)), I]])
        T_inv = np.linalg.inv(T.T @ T) @ T.T

        #Hermitian H and Q matrix
        Q = np.block([[np.zeros((N, N)), I],
                      [Ktilde, np.zeros((N, N))]])

        H =  T @ Q @ T_inv
        H = 1j * H

        print("H matrix shape:", H.shape)
        print("H matrix log2 shape:", np.log2(H.shape))
        print(np.allclose(U.T @ U, -Ktilde))
        #H /= np.max(np.abs(np.linalg.eigvals(H)))
        H = 1j * np.block([[np.zeros((N, N)), U],
                          [-U.T, np.zeros((N, N))]])
        np.save('H_sem.npy', H)
        print("H matrix max eigenvalue:", np.max(np.abs(np.linalg.eigvals(H))))
        herm_err = np.max(np.abs(H - H.conj().T))
        assert herm_err < 1e-10, \
            f"H is NOT Hermitian (max error={herm_err:.3e}) — check U assembly."
        print(f"✓ Hermitian: max|H - H†| = {herm_err:.3e}")
        


        return Q, H, T, T_inv, Mhalf

    # ------------------------ Build Full System -------------------------
    def build(self):
        """Run full SEM assembly and Hamiltonian construction."""
        self.x, M_full, K_full, U = self._assemble_global()
        self.Q, self.H, self.U, self.T, self.T_inv, self.Mhalf = self._mass_scale_and_build_H(M_full, K_full, U)
        return self

    # ------------------------ Getters / Setters -------------------------
    def get_dict(self) -> dict:
        """
        Returns a dictionary containing the transformation matrices.
        
        Returns:
            dict: The transformation matrices.
        """
        return {'h': self.h,
                't': self.t,
                'inv_t': self.inv_t,
                'q': self.q,
                'u': self.u,
                'sqrt_m': self.sqrt_m,
                'inv_sqrt_m': self.inv_sqrt_m}

    def set_materials(self, rho, mu):
        """Update materials and rebuild system."""
        self.rho = np.array(rho)
        self.mu = np.array(mu)
        self.build()
        return self




# -------- FUNCTIONS --------
def scale(array: np.ndarray, rows: int = 0, cols: int = 0) -> np.ndarray:
    """
    Scales a matrix by adding rows and columns of zeros.
    
    Args:
        array (np.ndarray): The matrix to be scaled.
        rows (int, optional): The number of rows to be added. Defaults to 0.
        cols (int, optional): The number of columns to be added. Defaults to 0.
    
    Returns:
        np.ndarray: The scaled matrix.
    """
    l = np.zeros((array.shape[0]+rows, array.shape[1]+cols))
    l[:array.shape[0], :array.shape[1]] = array
    return l

def boundary(array: np.ndarray, bcs: dict) -> np.ndarray:
    """
    Applies boundary conditions to a matrix.
    
    Args:
        array (np.ndarray): The matrix to be modified.
        bcs (dict): The boundary conditions.
        
    Returns:
        np.ndarray: The modified matrix.
    """
    for side in ['left', 'right']:
        index = 0 if side == 'left' else -1
        if bcs[side] == 'DBC':
            array[index, index] = 1
        elif bcs[side] == 'NBC':
            array[index, index] = 0
        else:
            pass
    return array
