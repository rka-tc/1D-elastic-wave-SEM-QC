"""
Spectral Element Hamiltonian Builder (multi-element, GLL-based)
Author: R.K.
 
Builds the global Hermitian Hamiltonian for 1D elastic wave simulation
using the Spectral Element Method (SEM) with GLL nodes.
 
Follows exactly the formulation in:
  Schade et al. (2023) "A quantum computing concept for 1-D elastic
  wave simulation", equations (22)-(25), extended to SEM.
 
PAPER EQUATIONS (FD case, adapted here for SEM):
─────────────────────────────────────────────────
  U      = E^{1/2} @ D @ M^{-1/2}               (eq. 22)
           → SEM:  U_e = sqrt(mu_e) * D_ref @ M_e^{-1/2}  per element
 
  Ktilde = -U^T @ U = M^{-1/2} K M^{-1/2}       (eq. 23)
 
  T      = [[U, 0],[0, I]]                        (eq. 24)
 
  H      = i * [[0,   U  ],                       (eq. 25)
                [-U^T, 0  ]]
 
H is Hermitian because:
  H† = (-i) * [[0, -U],[U^T, 0]]^T = i * [[0, U],[-U^T, 0]] = H  ✓
 
NOTE ON SEM vs FD:
  In FD, U^T @ U = -Ktilde holds exactly (U is upper triangular Cholesky).
  In SEM with multi-element assembly, U^T @ U ≠ -Ktilde globally due to
  element boundary contributions. However H = i*[[0,U],[-U^T,0]] is still
  Hermitian and correct. Do NOT use the T@Q@T_inv route for SEM — use the
  direct block form only.
 
STATE VECTORS:
  Forward:   |Φ(0)⟩ = T @ [M^{1/2} u0, M^{1/2} v0]^T   (eq. 20)
  Evolution: |Φ(t)⟩ = exp(-iHt) |Φ(0)⟩                  (eq. 9)
  Inverse:   [u_tilde, v_tilde] = T^{-1} |Φ(t)⟩          (eq. 21)
             u(t) = M^{-1/2} u_tilde(t)
"""
 
import numpy as np
 
 
class HamiltonianBuilderSEM:
    """
    Build the Hermitian Hamiltonian for 1D elastic SEM.
    Follows Schade et al. (2023) eq. 22-25, adapted for SEM.
    """
 
    def __init__(self, nx=16, order=4, L=1.0, mu=None, rho=None):
        self.Np  = order
        self.L   = L
        self.nx  = nx
        self.rho = rho
        print("Rho :", rho)
        self.mu = mu
 
        # nx = number of global nodes
        assert (nx - 1) % (order - 1) == 0, (
            f"nx={nx}, order={order} incompatible: "
            f"(nx-1) must be divisible by (order-1)"
        )
        self.Nel = (nx - 1) // (order - 1)
 
        # Precompute GLL reference nodes, weights, derivative matrix
        self.xi, self.wi = self._gll_nodes_weights(self.Np)
        self.D_ref       = self._compute_derivative_matrix(self.xi)
 
        # Full build
        self._build()
 
    # ------------------------------------------------------------------ #
    #  GLL utilities                                                       #
    # ------------------------------------------------------------------ #
 
    def _gll_nodes_weights(self, N):
        """GLL nodes and weights. Endpoints exactly ±1, weights positive."""
        if N < 2:
            raise ValueError("GLL requires N >= 2")
        from numpy.polynomial.legendre import legval, legroots
        Pn   = np.polynomial.legendre.Legendre.basis(N - 1)
        dPn  = Pn.deriv()
        x_in = legroots(dPn.coef)
        x    = np.sort(np.real_if_close(
                   np.concatenate(([-1.0], x_in, [1.0]))
               )).astype(np.float64)
        w    = 2.0 / (N * (N-1) * legval(x, Pn.coef)**2)
        return x, np.abs(np.real_if_close(w)).astype(np.float64)
 
    def _compute_derivative_matrix(self, xi):
        """Stable derivative matrix at GLL points."""
        N  = len(xi)
        D  = np.zeros((N, N))
        Pn = np.polynomial.legendre.Legendre.basis(N-1)(xi)
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (Pn[i] / Pn[j]) / (xi[i] - xi[j])
            if   xi[i] == -1.0:  D[i, i] =  N*(N-1)/4.0
            elif xi[i] ==  1.0:  D[i, i] = -N*(N-1)/4.0
            else:                 D[i, i] = -xi[i] / (2.0*(1.0 - xi[i]**2))
        return D
 
    # ------------------------------------------------------------------ #
    #  Element matrices                                                    #
    # ------------------------------------------------------------------ #
 
    def _element_matrices(self, rho_e, mu_e, dx_e):
        """
        Local element matrices.
 
        M_e = rho_e * J * diag(wi)              diagonal GLL mass matrix
        K_e = D^T @ (mu_e * diag(wi) * 2/dx) @ D   symmetric stiffness
        U_e = sqrt(mu_e) * D_ref @ M_e^{-1/2}   SEM version of eq. 22
        """
        J      = dx_e / 2.0
        m_diag = rho_e * J * self.wi
        M_e    = np.diag(m_diag)
        W      = np.diag(self.wi) * (2.0 / dx_e)
        K_e    = self.D_ref.T @ (mu_e * W) @ self.D_ref
        U_e    = np.sqrt(mu_e) * self.D_ref @ np.diag(1.0 / np.sqrt(m_diag))
        return M_e, K_e, U_e
 
    # ------------------------------------------------------------------ #
    #  Global assembly                                                     #
    # ------------------------------------------------------------------ #
 
    def _assemble_global(self):
        """
        Assemble global M, K, U.
 
        M, K: standard additive FEM assembly at shared nodes.
        U:    assigned per element (not additive at shared nodes).
        K:    negated ONCE after the loop.
        """
        Nel, Np = self.Nel, self.Np
        dx      = self.L / Nel
        N       = Nel * (Np - 1) + 1
 
        M = np.zeros((N, N))
        K = np.zeros((N, N))
        U = np.zeros((N, N))
        x = np.linspace(0, self.L, N)
 
        for e in range(Nel):
            idx = np.arange(e*(Np-1), e*(Np-1)+Np)
            M_e, K_e, U_e = self._element_matrices(self.rho[idx], self.mu[idx], dx)
            M[np.ix_(idx, idx)] += M_e
            K[np.ix_(idx, idx)] += K_e
            U[np.ix_(idx, idx)]  = U_e   # assign, not accumulate
 
        K = -K   # negate once after full assembly
 
        # Sanity checks
        assert np.allclose(M, np.diag(np.diag(M)), atol=1e-12), \
            "Global M is not diagonal — GLL assembly error."
        assert np.allclose(K, K.T, atol=1e-10), \
            f"K not symmetric, max|K-K^T|={np.max(np.abs(K-K.T)):.3e}"
 
        return x, M, K, U
 
    # ------------------------------------------------------------------ #
    #  Hamiltonian  (paper eq. 25)                                        #
    # ------------------------------------------------------------------ #
 
    def _build_hamiltonian(self, M, K, U):
        """
        H = i * [[0,   U  ],     (Schade et al. eq. 25, SEM version)
                 [-U^T, 0  ]]
 
        Hermitian proof:
          H† = (-i)*[[0,-U],[U^T,0]]^T = i*[[0,U],[-U^T,0]] = H  ✓
 
        Also build:
          sqrt_M   = M^{1/2}   (diagonal, exact)
          T        = [[U, 0],[0, I]]           (eq. 24)
          T_inv    = inv(T)
          sqrt_m   = block_diag(sqrt_M, sqrt_M)  for state transforms
          inv_sqrt_m = inv(sqrt_m)
          Q        = [[0, I],[Ktilde, 0]]      (impedance matrix, eq. 17)
        """
        N = M.shape[0]
        I = np.eye(N)
 
        # M^{1/2} and M^{-1/2} — exact because M is diagonal
        m_diag   = np.diag(M)
        sqrt_M   = np.diag(np.sqrt(m_diag))
        Minvhalf = np.diag(1.0 / np.sqrt(m_diag))
        Ktilde   = Minvhalf @ K @ Minvhalf

        # ── Transformation matrices (eq. 24) ──────────────────
        T     = np.block([[U,              np.zeros((N,N))],
                          [np.zeros((N,N)), I             ]])
        T_inv = np.linalg.inv(T)
 
        # ── Impedance matrix Q (eq. 17) ───────────────────────
        Q = np.block([[np.zeros((N,N)), I     ],
                      [Ktilde,          np.zeros((N,N))]])
 
        # ── Block sqrt_M for state-space transforms ────────────
        sqrt_m     = np.block([[sqrt_M,              np.zeros(M.shape)],
                                [np.zeros(M.shape),  sqrt_M           ]])
        inv_sqrt_m = np.linalg.inv(sqrt_m)
         
        # ── Hamiltonian (eq. 25) ──────────────────────────────
        H = 1j * np.block([[np.zeros((N,N)),  U    ],
                            [-U.T,             np.zeros((N,N))]])
        # H = 1j * T @ Q @ T_inv   # alternative form using T, Q (should be identical)
 
        # Verify Hermitian
        herm_err = np.max(np.abs(H - H.conj().T))
        assert herm_err < 1e-10, \
            f"H is NOT Hermitian (max error={herm_err:.3e}) — check U assembly."
        print(f"✓ Hermitian: max|H - H†| = {herm_err:.3e}")
 
 
        return H, Q, T, T_inv, sqrt_M, sqrt_m, inv_sqrt_m
 
    # ------------------------------------------------------------------ #
    #  Internal full build                                                 #
    # ------------------------------------------------------------------ #
 
    def _build(self):
        self.x, self.M, self.K, self.U = self._assemble_global()
        (self.h,
         self.q,
         self.t,
         self.inv_t,
         self.sqrt_M,
         self.sqrt_m,
         self.inv_sqrt_m) = self._build_hamiltonian(self.M, self.K, self.U)
 
    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
 
    def build(self):
        """Rebuild after material changes."""
        self._build()
        return self
 
    def set_materials(self, rho, mu):
        """Update per-element materials and rebuild."""
        self.rho = np.asarray(rho, dtype=float)
        self.mu  = np.asarray(mu,  dtype=float)
        return self.build()
 
    def apply_dirichlet_bc(self, side='left'):
        """Remove DOFs for Dirichlet BC and rebuild."""
        idx = list(range(self.M.shape[0]))
        if side in ('left',  'both'): idx = idx[1:]
        if side in ('right', 'both'): idx = idx[:-1]
        M_bc = self.M[np.ix_(idx, idx)]
        K_bc = self.K[np.ix_(idx, idx)]
        U_bc = self.U[np.ix_(idx, idx)]
        (self.h,
         self.q,
         self.t,
         self.inv_t,
         self.sqrt_M,
         self.sqrt_m,
         self.inv_sqrt_m) = self._build_hamiltonian(M_bc, K_bc, U_bc)
        return self
 
    def get_dict(self):
        """Return key matrices — same keys as original code."""
        return {
            'h':          self.h,
            't':          self.t,
            'inv_t':      self.inv_t,
            'q':          self.q,
            'u':          self.U,
            'sqrt_m':     self.sqrt_m,
            'inv_sqrt_m': self.inv_sqrt_m,
        }
 
    def verify_hermitian(self):
        """Standalone Hermitian check."""
        err = np.max(np.abs(self.h - self.h.conj().T))
        print(f"[{'PASS' if err<1e-10 else 'FAIL'}] max|H - H†| = {err:.3e}")
        return err < 1e-10
 
    def eigenvalue_summary(self):
        """Eigenvalues of H are ±ω (real wave frequencies)."""
        eigs = np.linalg.eigvalsh(self.h)
        pos  = eigs[eigs > 1e-10]
        print(f"H eigenvalues: range=[{eigs.min():.4f}, {eigs.max():.4f}]")
        print(f"Angular freqs ω: {np.round(pos, 4)}")
        return eigs
 
 
# ------------------------------------------------------------------ #
#  Helper functions (unchanged from original)                          #
# ------------------------------------------------------------------ #
 
def scale(array, rows=0, cols=0):
    out = np.zeros((array.shape[0]+rows, array.shape[1]+cols))
    out[:array.shape[0], :array.shape[1]] = array
    return out
 
def boundary(array, bcs):
    for side in ['left', 'right']:
        index = 0 if side == 'left' else -1
        if bcs[side] == 'DBC':   array[index, index] = 1
        elif bcs[side] == 'NBC': array[index, index] = 0
    return array