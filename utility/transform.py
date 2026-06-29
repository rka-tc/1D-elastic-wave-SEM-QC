
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
from scipy.linalg import eigh


class HamiltonianBuilderSEM:
    """
    Build the Hamiltonian for 1D elastic SEM.
    Supports multi-element domain with mean-valued rho, mu per element.
    """

    def __init__(self, nx=16, order=4, L=1, mu=None, rho=None):
        self.Np = order
        self.Nel = int((nx - 1) / (order-1))
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
        self.plot_sem_domain()
        N = self.m.shape[0]
        self.sqrt_m = np.block([[self.sqrt_M,          np.zeros((N, N))],
                                [np.zeros((N, N)),  self.sqrt_M]])   
        # print("Mass matrix sqrt :", self.sqrt_m.shape)

        self.inv_sqrt_m = np.diag(1.0 / self.sqrt_m.diagonal())
        # print("Mass matrix inv sqrt :", self.inv_sqrt_m.shape)

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
        N = len(xi)
        D = np.zeros((N, N))
        P = np.polynomial.legendre.Legendre.basis(N - 1)
        Pn = P(xi)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (Pn[i] / Pn[j]) / (xi[i] - xi[j])
            D[i, i] = -np.sum(D[i, :])
        
        return D



    # ------------------------ Element Matrices --------------------------
    def _element_matrices(self, rho_e, mu_e, dx_e):
        """Local element matrices (mass, stiffness)."""
        N = self.Np
        J = dx_e / 2.0
        #print(J, rho_e, self.wi)
        M_e = rho_e * np.diag(self.wi) * J
        M_einvsqrt = np.diag(1/np.sqrt(M_e.diagonal()))
        E_ehalf = np.sqrt(mu_e * (np.diag(self.wi) * (2.0 / dx_e)))
        Uref = E_ehalf @ self.D_ref
        # Uref = (mu_e * (np.diag(self.wi) * (2.0 / dx_e))) @ self.D_ref @ M_einvsqrt
        K_e = -self.D_ref.T @ ( mu_e * (np.diag(self.wi) * (2.0 / dx_e)) @ self.D_ref) 
        K_etilde = M_einvsqrt @ (-K_e) @  M_einvsqrt
        # print("U.T @ U - K check (should be close to 0):", np.max(np.abs(Uref.T @ Uref - K_etilde)))
        return M_e, K_e, Uref


    # ------------------------ Global Assembly ---------------------------
    def _assemble_global(self):
        # """Assemble global M and K matrices."""
        Nel, Np, L = self.Nel, self.Np, self.L
        dx       = L / Nel
        N_global = int(Nel * (Np - 1) + 1)

        M = np.zeros((N_global, N_global))
        K = np.zeros((N_global, N_global))
        B_raw = np.zeros((Nel * Np, N_global))
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

            # ADD THIS BLOCK — scatter Uref columns using same idx connectivity
            for local_j, global_j in enumerate(idx):
                B_raw[e * Np:(e + 1) * Np, global_j] += Uref[:, local_j]

        # after the loop — Dirichlet BCs
        M_s   = M[1:-1, 1:-1]
        K_s   = K[1:-1, 1:-1]
        B_int = B_raw[:, 1:-1]          # drop boundary columns
        N     = M_s.shape[0]

        # GLOBAL mass scaling — this is the critical fix vs local scaling
        inv_sqrt_M = 1.0 / np.sqrt(np.diag(M_s))
        B = B_int * inv_sqrt_M[np.newaxis, :]   # broadcast: divide each column j by sqrt(M_s[j,j])

        # thin QR → U_global
        from scipy.linalg import qr, cholesky
        _, R    = qr(B, mode='economic')
        signs   = np.sign(np.diag(R))
        U_global = np.diag(signs) @ R           # enforce positive diagonal convention

        # sanity checks
        K_tilde = -np.diag(inv_sqrt_M) @ K_s @ np.diag(inv_sqrt_M)
        K_hat   = K_tilde
        print("shapes of matrix:","1) M:",M_s.shape,"2) K",K_s.shape,"3) U_global:", U_global.shape)
        L = cholesky(K_hat, lower=False)
        print(np.max(np.abs(L - U_global)))
        print("B.T @ B - K_hat:",np.max(np.abs(B.T @ B - K_hat))) 
        # print(B.T @ B, K_hat)  # < 1e-10, "B assembly wrong"
        print("U_global.T @ U_global - K_hat:",np.max(np.abs(U_global.T @ U_global - K_hat))) #< 1e-10, "Cholesky violated"
        # print("is M diagonal?", np.allclose(M, np.diag(np.diag(M))))

        return x, M_s, U_global, K_hat

    # ------------------------ Boundary & Scaling ------------------------
    def _apply_boundary_conditions(self, M, K, U):
        return M[1:-1, 1:-1], K[1:-1, 1:-1], U[1:-1, 1:-1]

    def _mass_scale_and_build_H(self, M, Ktilde, U):
        """Mass scaling, Q matrix, and Hermitian H."""

        Mhalf    = np.diag(np.sqrt(np.diag(M)))
        Minvhalf = np.diag(1.0 / np.sqrt(np.diag(M)))
        # Ktilde = -U.T @ U
        # Ktilde = -Minvhalf @ K @ Minvhalf

        N = U.shape[0]
        I = np.eye(N)

        #Transformation matrix T and its inverse
        T = np.block([[U, np.zeros((N, N))],
                      [np.zeros((N, N)), I]])
        T_inv = np.linalg.inv(T.T @ T) @ T.T

        #Hermitian H and Q matrix
        Q = np.block([[np.zeros((N, N)), I],
                       [Ktilde, np.zeros((N, N))]])

        # H =  T @ Q @ T_inv
        # H = 1j * H

        # print("H matrix shape:", H.shape)
        # print("H matrix log2 shape:", np.log2(H.shape))
        # print(np.allclose(U.T @ U, -Ktilde))
        #H /= np.max(np.abs(np.linalg.eigvals(H)))
        H = 1j * np.block([[np.zeros((N, N)), U],
                           [-U.T, np.zeros((N, N))]])
        print("H shape:", H.shape, "T shape:", T.shape)
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

    
    def plot_sem_domain(self, figsize=(14, 5)):
        """
        Plot the SEM domain showing all elements, GLL nodes, and quadrature weights.
        
        Displays:
        - Element boundaries as vertical dashed lines
        - GLL node positions as scatter points
        - Quadrature weights as bar heights (scaled)
        - Node indices and element labels
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        Nel, Np, L = self.Nel, self.Np, self.L
        dx = L / Nel

        fig, axes = plt.subplots(2, 1, figsize=figsize,
                                gridspec_kw={'height_ratios': [2, 1]},
                                sharex=True)
        ax_nodes, ax_weights = axes

        # ── colour palette (one colour per element, cycling) ──────────────
        cmap   = plt.get_cmap('tab10')
        colors = [cmap(e % 10) for e in range(Nel)]

        # ── precompute global GLL node coordinates ─────────────────────────
        # xi in [-1,1], mapped to physical [x_left, x_right] per element
        all_x_phys   = []   # physical coordinate of each GLL node
        all_weights  = []   # quadrature weight (in physical space) of each node
        all_elem_idx = []   # which element owns each node
        all_node_idx = []   # local node index within element

        for e in range(Nel):
            x_left = e * dx
            x_phys = x_left + 0.5 * dx * (self.xi + 1.0)   # length Np
            J = dx / 2.0                                      # Jacobian
            w_phys = self.wi * J                              # physical weights
            all_x_phys.append(x_phys)
            all_weights.append(w_phys)
            all_elem_idx.append([e] * Np)
            all_node_idx.append(list(range(Np)))

        # ── top panel: node positions ──────────────────────────────────────
        ax_nodes.set_title("SEM domain — GLL nodes and element structure",
                            fontsize=12, fontweight='bold', pad=10)

        # Element background bands
        for e in range(Nel):
            x0, x1 = e * dx, (e + 1) * dx
            ax_nodes.axvspan(x0, x1, alpha=0.08, color=colors[e], zorder=0)

        # Element boundaries
        for e in range(Nel + 1):
            ax_nodes.axvline(e * dx, color='0.3', linewidth=1.2,
                            linestyle='--', zorder=1)

        # GLL nodes (shared interface nodes plotted once, split-coloured)
        plotted_x = set()
        for e in range(Nel):
            x_phys = all_x_phys[e]
            for ni, (xp, wp) in enumerate(zip(x_phys, all_weights[e])):
                key = round(xp, 12)
                is_interface = (ni == 0 and e > 0)   # shared with previous element

                if is_interface:
                    # Half-left: previous element colour; half-right: current
                    ax_nodes.scatter([xp], [0], s=120,
                                    color=colors[e - 1], zorder=5,
                                    marker='D', edgecolors='k', linewidths=0.6)
                    ax_nodes.scatter([xp], [0], s=60,
                                    color=colors[e], zorder=6,
                                    marker='D', edgecolors='none')
                elif key not in plotted_x:
                    ax_nodes.scatter([xp], [0], s=90,
                                    color=colors[e], zorder=4,
                                    marker='o', edgecolors='k', linewidths=0.6)
                plotted_x.add(key)

                # Node index label (global)
                global_ni = e * (Np - 1) + ni
                ax_nodes.text(xp, 0.07, str(global_ni),
                            ha='center', va='bottom', fontsize=7,
                            color='0.3')

        # Element labels (centred in band)
        for e in range(Nel):
            xc = (e + 0.5) * dx
            ax_nodes.text(xc, -0.12, f"Ω{e}",
                        ha='center', va='top', fontsize=9,
                        color=colors[e], fontweight='bold')

        ax_nodes.set_ylim(-0.25, 0.35)
        ax_nodes.set_yticks([])
        ax_nodes.spines[['top', 'right', 'left']].set_visible(False)
        ax_nodes.set_ylabel("Nodes", fontsize=9, color='0.5')

        # ── bottom panel: quadrature weights ──────────────────────────────
        ax_weights.set_title("GLL quadrature weights  w_i · J  (physical space)",
                            fontsize=10, color='0.4', pad=6)

        bar_half = dx / (2.2 * Np)   # half-width of each bar
        for e in range(Nel):
            x_phys  = all_x_phys[e]
            w_phys  = all_weights[e]
            for ni, (xp, wp) in enumerate(zip(x_phys, w_phys)):
                ax_weights.bar(xp, wp, width=bar_half * 1.6,
                            color=colors[e], alpha=0.75,
                            edgecolor='k', linewidth=0.4, zorder=3)
                ax_weights.text(xp, wp + 0.001, f"{wp:.3f}",
                                ha='center', va='bottom', fontsize=6,
                                color='0.35', rotation=70)

        # Element boundaries (mirrored)
        for e in range(Nel + 1):
            ax_weights.axvline(e * dx, color='0.3', linewidth=1.2,
                            linestyle='--', zorder=1)

        ax_weights.set_xlim(-0.01 * L, 1.01 * L)
        ax_weights.set_xlabel("x (physical)", fontsize=10)
        ax_weights.set_ylabel("w · J", fontsize=9)
        ax_weights.spines[['top', 'right']].set_visible(False)

        # ── legend ────────────────────────────────────────────────────────
        patches = [mpatches.Patch(color=colors[e], alpha=0.7, label=f"Ω{e}")
                for e in range(min(Nel, 10))]
        ax_nodes.legend(handles=patches, loc='upper right',
                        fontsize=8, ncol=min(Nel, 5),
                        framealpha=0.5, title="Elements")

        plt.tight_layout()
        plt.savefig("sem_domain.png", dpi=150, bbox_inches='tight')
        plt.show()
        return fig, axes




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




