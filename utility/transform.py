
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
        
        # print("\nGLL nodes xi:", self.xi)
        # print("GLL weights wi:", self.wi)

        self.D_ref = self._compute_derivative_matrix(self.xi)
        # print("D_ref:")
        # print(np.array2string(self.D_ref, precision=6, suppress_small=True))

        # print("\nD_ref column sums (should all be ~0):")
        # print(self.D_ref.sum(axis=0))

        # print("\nD_ref row sums:")
        # print(self.D_ref.sum(axis=1))
        
        # compute matrices
        print("Assembling global matrices...")
        self.x, self.m, self.u, self.k = self._assemble_global()
        self.nx = len(self.x)
        self.q, self.h, self.t, self.inv_t, self.sqrt_M = self._mass_scale_and_build_H(self.m, self.k, self.u)
        self.plot_sem_domain()
        # self.Pi_ac, self.eigenvalues, self.eigenvectors = self.build_acoustic_projector(-1 * self.k, self.m, self.Nel)
        # self.H_proj, self.U_proj, self.Pi_block = self.build_projected_block_H(self.u, self.Pi_ac, self.h)

        # # build transform mass matrix
        # self.sqrt_m = np.block([[self.sqrt_M, np.zeros(self.m.shape)],
        #                  [np.zeros(self.m.shape), self.sqrt_M]])

        # CORRECT — only scale displacement half
        N = self.m.shape[0]
        self.sqrt_m = np.block([[self.sqrt_M,          np.zeros((N, N))],
                                [np.zeros((N, N)), np.eye(N)]])          # identity on velocity
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
                D[i, i] = +N * (N - 1) / 4
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
        # print("Element mass matrix M_e :", M_e)
        # print("Element stiffness matrix K_e :", K_e)
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
        # 2. Apply Boundary Conditions (truncate M and K)
        M_sub = M[1:-1, 1:-1]
        K_sub = K[1:-1, 1:-1]

        # 3. Mass-Normalize the Stiffness Matrix
        # This accounts for the density (rho) changes
        inv_sqrt_M = np.diag(1.0 / np.sqrt(np.diag(M_sub)))
        K_tilde = inv_sqrt_M @ (-K_sub) @ inv_sqrt_M

        vals, vecs = np.linalg.eigh(K_tilde)        # K_tilde = inv_sqrt_M @ (-K_sub) @ inv_sqrt_M
        vals = np.maximum(vals, 0.0)                  # numerical safety
        U = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        # U = K_tilde @ inv_sqrt_M
        M = M_sub
        K = K_sub
        np.save('M_sem.npy', M)
        np.save('K_sem.npy', K)
        np.save('Ktilde.npy', K_tilde)
        # print("global_idx for element 0:", [0*(Np-1) + i for i in range(Np)])  # [0,1,2,3]
        # print("global_idx for element 1:", [1*(Np-1) + i for i in range(Np)])  # [3,4,5,6]
        # print("global_idx for element 2:", [2*(Np-1) + i for i in range(Np)])
        # print("K simmetric check (should be close to 0):", np.max(np.abs(K - K.T)))
        # print("Mass matrix M :", M.shape)
        # print("Stiffness matrix K :", K.shape)
        # print("matrix U :", U.shape)
        # print("U.T @ U - K check (should be close to 0):", np.max(np.abs(U.T @ U - K_tilde)))
        # print("is M diagonal?", np.allclose(M, np.diag(np.diag(M))))

        return x, M, U, K

    # ------------------------ Boundary & Scaling ------------------------
    def _apply_boundary_conditions(self, M, K, U):
        return M[1:-1, 1:-1], K[1:-1, 1:-1], U[1:-1, 1:-1]

    def _mass_scale_and_build_H(self, M, K, U):
        """Mass scaling, Q matrix, and Hermitian H."""

        Mhalf    = np.diag(np.sqrt(np.diag(M)))
        Minvhalf = np.diag(1.0 / np.sqrt(np.diag(M)))
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

        # print("H matrix shape:", H.shape)
        # print("H matrix log2 shape:", np.log2(H.shape))
        # print(np.allclose(U.T @ U, -Ktilde))
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
        # self.Pi_ac, self.eigenvalues, self.eigenvectors = self.build_acoustic_projector(K_full, M_full, self.Nel)
        # self.H_proj, self.U_proj, self.Pi_block = self.build_projected_block_H(U, self.Pi_ac, self.H)
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

        #------------------projector------------------

    def build_acoustic_projector(self, K, M, Nel_ac):
        """
        Solves K v = omega^2 M v.
        Extracts Nel_ac acoustic eigenvectors.
        Returns M-orthogonal projector Pi_ac in the N-dimensional space.
        """
        eigenvalues, eigenvectors = eigh(K, M)

        # Spectral gap check
        om_ac  = np.sqrt(np.abs(eigenvalues[Nel_ac - 1]))
        om_opt = np.sqrt(np.abs(eigenvalues[Nel_ac]))
        print(f"\nSpectral gap:")
        print(f"  omega_acoustic_max = {om_ac:.4f}")
        print(f"  omega_optical_min  = {om_opt:.4f}")
        print(f"  Gap ratio          = {om_opt/om_ac:.2f}x")

        # Acoustic eigenvectors in physical space
        V_ac  = eigenvectors[:, :Nel_ac]              # shape (N, Nel_ac)

        # M-orthogonal projector: Pi = V_ac V_ac^T M
        Pi_ac = V_ac @ V_ac.T @ M                     # shape (N, N)

        # Idempotency check
        err = np.linalg.norm(Pi_ac @ Pi_ac - Pi_ac)
        print(f"  Projector idempotency error: {err:.2e}")

        return Pi_ac, eigenvalues, eigenvectors

    def build_projected_block_H(self, U, Pi_ac, H_full):
        """
        Lifts Pi_ac to 2N x 2N block structure and projects H.

        Pi_block = [[Pi_ac,   0   ],
                    [  0,   Pi_ac ]]

        H_proj = Pi_block @ H @ Pi_block
            = 1j * [[0,         Pi_ac U Pi_ac ],
                    [-Pi_ac U.T Pi_ac, 0      ]]
        """
        N = U.shape[0]

        # Project U in the physical subspace
        U_proj = Pi_ac @ U @ Pi_ac                    # shape (N, N)

        # Lift projector to block space
        Pi_block = np.block([
            [Pi_ac,              np.zeros((N, N))],
            [np.zeros((N, N)),   Pi_ac           ]
        ])                                             # shape (2N, 2N)

        # Build projected block H
        H_proj = 1j * np.block([
            [np.zeros((N, N)),   U_proj          ],
            [-U_proj.T,          np.zeros((N, N))]
        ])

        # Verify Hermitian
        herm_err = np.linalg.norm(H_proj - H_proj.conj().T)
        print(f"\nH_proj Hermitian error: {herm_err:.2e}")

        # Verify via direct projection: H_proj = Pi_block H Pi_block
        H_proj_2 = Pi_block @ H_full @ Pi_block
        consist  = np.linalg.norm(H_proj - H_proj_2)
        print(f"Consistency ‖direct - block‖: {consist:.2e}")

        return H_proj, U_proj, Pi_block
    
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




