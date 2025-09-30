import abc
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import equinox as eqx
import sympy as sp
from jaxtyping import PRNGKeyArray

from damped_linoss.models.common import GLU, simple_uniform_init


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.
    Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


class _AbstractLinOSSLayer(eqx.Module):
    @abc.abstractmethod
    def _recurrence(self):
        raise NotImplementedError
    

class IMLayer(_AbstractLinOSSLayer):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        A_max: float,
        dt_std: float,
        key: PRNGKeyArray,
        **kwargs, 
    ):
        A_key, B_key, C_key, D_key, dt_key, key = jr.split(key, 6)
        self.dt = normal(stddev=dt_std)(dt_key, (state_dim,))
        self.A_diag = jr.uniform(A_key, shape=(state_dim,)) * A_max
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _recurrence(self, A_diag, dt, Bu_elements):
        """Compute the LxP output of LinOSS-IM given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        S = 1.0 + dt**2.0 * A_diag
        M_11 = 1.0 - dt**2.0 * A_diag / S
        M_12 = -1.0 * dt * A_diag / S
        M_21 = dt / S
        M_22 = 1 / S

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * A_diag.shape[0]))

        F1 = M_11 * Bu_elements * dt
        F2 = M_21 * Bu_elements * dt
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, A_diag.shape[0] :]

        return ys
    
    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        dt = nn.sigmoid(self.dt)
        A_diag = nn.relu(self.A_diag)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs


class IMEXLayer(_AbstractLinOSSLayer):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        A_max: float, 
        dt_std: float,
        key: PRNGKeyArray,
        **kwargs,
    ):
        A_key, B_key, C_key, D_key, dt_key, key = jr.split(key, 6)
        self.dt = normal(stddev=dt_std)(dt_key, (state_dim,))
        self.A_diag = jr.uniform(A_key, shape=(state_dim,)) * A_max
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))
    
    def _recurrence(self, A_diag, dt, Bu_elements):
        """Compute the LxP output of LinOSS-IMEX given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        A_ = jnp.ones_like(A_diag)
        B_ = -1.0 * dt * A_diag
        C_ = dt
        D_ = 1.0 - (dt**2.0) * A_diag

        M = jnp.concatenate([A_, B_, C_, D_])
        M_elements = M * jnp.ones((sql, 4 * A_diag.shape[0]))

        F1 = Bu_elements * dt
        F2 = Bu_elements * (dt**2.0)
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, A_diag.shape[0] :]

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        dt = nn.sigmoid(self.dt)
        A_diag = nn.relu(self.A_diag)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class DampedIMEX1Layer(_AbstractLinOSSLayer):
    """
    Based on the characteristic recurrence
    z_k+1 = z_k + dt * (-Ax_k - Gz_k+1 + Bu_k+1)
    x_k+1 = x_k + dt * (z_k+1)
    """
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float, 
        G_max: float, 
        A_min: float, 
        A_max: float, 
        dt_std: float, 
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.state_dim = state_dim
        init_key, B_key, C_key, D_key, key = jr.split(key, 5)
        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(A_min, A_max, G_min, G_max, dt_std, init_key)
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(r_min, r_max, theta_min, theta_max, dt_std, init_key)
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        """Boolean check if (A,G,dt) in valid region"""
        dt = nn.sigmoid(dt)
        return (G_diag >= 0) & (((G_diag - dt*A_diag)**2 - 4*A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        # Solve symbolically
        a, g, dt, lam1, lam2 = sp.symbols('a g dt lam1 lam2')

        # Characteristic recurrence for 1 decoupled 2x2 system
        M_i = sp.Matrix([[1/(1+dt*g), -a*dt/(1 + dt*g)], [dt/(1 + dt*g), 1 - a*dt**2/(1 + dt*g)]])
        # Eigenvalue pair expressions
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        # Sample timesteps
        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        # Sample eigenvalues in ring 
        mag = jnp.sqrt(jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        # Convert to (A, G) representation
        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        # Invertibility, stability, and validity checks
        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)
        invertible = jnp.all(jnp.isclose(lam1_out_vals, lam1_vals) | jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)) \
                   & jnp.all(jnp.isclose(lam2_out_vals, lam2_vals) | jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals))
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))
        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        # Cast to real (imag part is nonzero, ~machine precision)
        a_vals = a_vals.real
        g_vals = g_vals.real

        return a_vals, g_vals, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        """Uniform sampling over valid (A,G,dt) region"""
        bsz = 512
        done = False 
        A_vals = []
        G_vals = []
        dt_vals = []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim and len(G_vals) >= self.state_dim and len(dt_vals) >= self.state_dim:
                done = True

        A_diag = jnp.array(A_vals[:self.state_dim])
        G_diag = jnp.array(G_vals[:self.state_dim])
        dt = jnp.array(dt_vals[:self.state_dim])

        return A_diag, G_diag, dt
    
    def _soft_project_AGdt(self, A_diag, G_diag, dt):
        """soft projection to the _is_valid_AGdt region"""
        dt = nn.sigmoid(dt)

        G_diag = nn.relu(G_diag)
        
        A_low = (2 + dt * G_diag - 2 * jnp.sqrt(1 + dt * G_diag)) / jnp.maximum(dt**2, 1e-6)
        A_high = (2 + dt * G_diag + 2 * jnp.sqrt(1 + dt * G_diag)) / jnp.maximum(dt**2, 1e-6)
        A_diag = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
        
        return A_diag, G_diag, dt

    def _recurrence(self, A_diag, G_diag, dt, Bu_elements):
        """Compute the LxP output of Damped-LinOSS given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            G_diag          (float32):    diagonal damping matrix   (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        I = jnp.ones_like(A_diag)
        S = I + dt * G_diag
        M_11 = 1.0 / S
        M_12 = -dt / S * A_diag
        M_21 = dt / S
        M_22 = I - dt**2 / S * A_diag

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * self.state_dim))

        F1 = dt * (1.0 / S) * Bu_elements
        F2 = dt**2 * (1.0 / S) * Bu_elements
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]  # Position component

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        A_diag, G_diag, dt = self._soft_project_AGdt(self.A_diag, self.G_diag, self.dt)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, G_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class DampedIMEX2Layer(_AbstractLinOSSLayer):
    """
    Based on the characteristic recurrence
    z_k+1 = z_k + dt * (-Ax_k - Gz_k + Bu_k+1)
    x_k+1 = x_k + dt * (z_k+1)
    """
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float, 
        G_max: float, 
        A_min: float, 
        A_max: float, 
        dt_std: float, 
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.state_dim = state_dim
        init_key, B_key, C_key, D_key, key = jr.split(key, 5)
        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(A_min, A_max, G_min, G_max, dt_std, init_key)
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(r_min, r_max, theta_min, theta_max, dt_std, init_key)
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        """Boolean check if (A,G,dt) in valid region"""
        dt = nn.sigmoid(dt)
        return (G_diag >= 0) & (((G_diag + dt*A_diag)**2 - 4*A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        # Solve symbolically
        a, g, dt, lam1, lam2 = sp.symbols('a g dt lam1 lam2')

        # Characteristic recurrence for 1 decoupled 2x2 system
        M_i = sp.Matrix([[1-dt*g, -a*dt], [dt*(1-dt*g), 1 - dt**2*a]])
        # Eigenvalue pair expressions
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        # Sample timesteps
        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        # Sample eigenvalues in ring 
        mag = jnp.sqrt(jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        # Convert to (A, G) representation
        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        # Invertibility, stability, and validity checks
        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)
        invertible = jnp.all(jnp.isclose(lam1_out_vals, lam1_vals) | jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)) \
                   & jnp.all(jnp.isclose(lam2_out_vals, lam2_vals) | jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals))
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))
        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        # Cast to real (imag part is nonzero, ~machine precision)
        a_vals = a_vals.real
        g_vals = g_vals.real

        return a_vals, g_vals, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        """Uniform sampling over valid (A,G,dt) region"""
        bsz = 512
        done = False 
        A_vals = []
        G_vals = []
        dt_vals = []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim and len(G_vals) >= self.state_dim and len(dt_vals) >= self.state_dim:
                done = True

        A_diag = jnp.array(A_vals[:self.state_dim])
        G_diag = jnp.array(G_vals[:self.state_dim])
        dt = jnp.array(dt_vals[:self.state_dim])

        return A_diag, G_diag, dt
    
    def _soft_project_AGdt(self, A_diag, G_diag, dt):
        """soft projection to the _is_valid_AGdt region"""
        dt = nn.sigmoid(dt)

        G_diag = nn.relu(G_diag)

        A_low = (2 - dt * G_diag - 2 * jnp.sqrt(1 - dt * G_diag)) / jnp.maximum(dt**2, 1e-6)
        A_high = (2 - dt * G_diag + 2 * jnp.sqrt(1 - dt * G_diag)) / jnp.maximum(dt**2, 1e-6)
        A_diag = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
        
        return A_diag, G_diag, dt

    def _recurrence(self, A_diag, G_diag, dt, Bu_elements):
        """Compute the LxP output of Damped-LinOSS given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            G_diag          (float32):    diagonal damping matrix   (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        I = jnp.ones_like(A_diag)
        M_11 = I - dt * G_diag
        M_12 = -dt * A_diag
        M_21 = dt * (I - dt * G_diag)
        M_22 = I - dt**2 * A_diag

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * self.state_dim))

        F1 = dt * Bu_elements
        F2 = dt**2 * Bu_elements
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]  # Position component

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        A_diag, G_diag, dt = self._soft_project_AGdt(self.A_diag, self.G_diag, self.dt)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, G_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class DampedIMLayer(_AbstractLinOSSLayer):
    """
    Based on the characteristic recurrence
    z_k+1 = z_k + dt * (-Ax_k+1 - Gz_k+1 + Bu_k+1)
    x_k+1 = x_k + dt * (z_k+1)
    """
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float, 
        G_max: float, 
        A_min: float, 
        A_max: float, 
        dt_std: float, 
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.state_dim = state_dim
        init_key, B_key, C_key, D_key, key = jr.split(key, 5)
        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(A_min, A_max, G_min, G_max, dt_std, init_key)
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(r_min, r_max, theta_min, theta_max, dt_std, init_key)
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        """Boolean check if (A,G,dt) in valid region"""
        dt = nn.sigmoid(dt)
        return (G_diag + dt*A_diag >= 0) & ((G_diag**2 - 4*A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        # Solve symbolically
        a, g, dt, lam1, lam2 = sp.symbols('a g dt lam1 lam2')

        # Characteristic recurrence for 1 decoupled 2x2 system
        M_i = sp.Matrix([[1/(1 + dt*g + dt**2*a), -a*dt/(1 + dt*g + dt**2*a)], [dt/(1 + dt*g + dt**2*a), (1 + dt*g)/(1 + dt*g + dt**2*a)]])
        # Eigenvalue pair expressions
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        # Sample timesteps
        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        # Sample eigenvalues in ring 
        mag = jnp.sqrt(jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        # Convert to (A, G) representation
        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        # Invertibility, stability, and validity checks
        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)
        invertible = jnp.all(jnp.isclose(lam1_out_vals, lam1_vals) | jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)) \
                   & jnp.all(jnp.isclose(lam2_out_vals, lam2_vals) | jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals))
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))
        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        # Cast to real (imag part is nonzero, ~machine precision)
        a_vals = a_vals.real
        g_vals = g_vals.real

        return a_vals, g_vals, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        """Uniform sampling over valid (A,G,dt) region"""
        bsz = 512
        done = False 
        A_vals = []
        G_vals = []
        dt_vals = []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim and len(G_vals) >= self.state_dim and len(dt_vals) >= self.state_dim:
                done = True

        A_diag = jnp.array(A_vals[:self.state_dim])
        G_diag = jnp.array(G_vals[:self.state_dim])
        dt = jnp.array(dt_vals[:self.state_dim])

        return A_diag, G_diag, dt
    
    def _soft_project_AGdt(self, A_diag, G_diag, dt):
        """soft projection to the _is_valid_AGdt region"""
        dt = nn.sigmoid(dt)

        G_low = -dt * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)

        A_low = 1/4*G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)

        return A_diag, G_diag, dt

    def _recurrence(self, A_diag, G_diag, dt, Bu_elements):
        """Compute the LxP output of Damped-LinOSS given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            G_diag          (float32):    diagonal damping matrix   (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        I = jnp.ones_like(A_diag)
        S = I + dt * G_diag + dt**2 * A_diag
        M_11 = 1 / S
        M_12 = -dt * A_diag / S
        M_21 = dt / S
        M_22 = (I + dt * G_diag) / S

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * self.state_dim))

        F1 = dt * (1.0 / S) * Bu_elements
        F2 = dt**2 * (1.0 / S) * Bu_elements
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]  # Position component

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        A_diag, G_diag, dt = self._soft_project_AGdt(self.A_diag, self.G_diag, self.dt)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, G_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class DampedEXLayer(_AbstractLinOSSLayer):
    """
    Based on the characteristic recurrence
    z_k+1 = z_k + dt * (-Ax_k - Gz_k + Bu_k+1)
    x_k+1 = x_k + dt * (z_k)
    """
    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    dt: jax.Array
    state_dim: int

    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        G_min: float, 
        G_max: float, 
        A_min: float, 
        A_max: float, 
        dt_std: float, 
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.state_dim = state_dim
        init_key, B_key, C_key, D_key, key = jr.split(key, 5)
        if initialization == "uniform":
            self.A_diag, self.G_diag, self.dt = self._uniform_init_AGdt(A_min, A_max, G_min, G_max, dt_std, init_key)
        elif initialization == "ring":
            self.A_diag, self.G_diag, self.dt = self._ring_init_AGdt(r_min, r_max, theta_min, theta_max, dt_std, init_key)
        self.B = simple_uniform_init(B_key, shape=(state_dim, hidden_dim, 2), std=1.0 / jnp.sqrt(hidden_dim))
        self.C = simple_uniform_init(C_key, shape=(hidden_dim, state_dim, 2), std=1.0 / jnp.sqrt(state_dim))
        self.D = normal(stddev=1.0)(D_key, (hidden_dim,))

    def _is_valid_AGdt(self, A_diag, G_diag, dt):
        """Boolean check if (A,G,dt) in valid region"""
        dt = nn.sigmoid(dt)
        return (G_diag - dt*A_diag >= 0) & ((G_diag**2 - 4*A_diag) < 0)

    def _ring_init_AGdt(self, r_min, r_max, theta_min, theta_max, dt_std, key):
        # Solve symbolically
        a, g, dt, lam1, lam2 = sp.symbols('a g dt lam1 lam2')

        # Characteristic recurrence for 1 decoupled 2x2 system
        M_i = sp.Matrix([[1-dt*g, -dt*a], [dt, 1]])
        # Eigenvalue pair expressions
        eigs = list(M_i.eigenvals().keys())
        eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
        sol = sp.solve(eqs, (a, g))[0]
        f = sp.lambdify((lam1, lam2, dt), sol, "numpy")

        # Sample timesteps
        mag_key, arg_key, dt_key = jr.split(key, 3)
        dt_vals = normal(stddev=dt_std)(dt_key, (self.state_dim,))
        dt_sigmoid = nn.sigmoid(dt_vals)

        # Sample eigenvalues in ring 
        mag = jnp.sqrt(jr.uniform(mag_key, shape=(self.state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
        arg = jr.uniform(arg_key, shape=(self.state_dim,)) * (theta_max - theta_min) + theta_min
        lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
        lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

        # Convert to (A, G) representation
        a_vals, g_vals = f(lam1_vals, lam2_vals, dt_sigmoid)

        # Invertibility, stability, and validity checks
        h1 = sp.lambdify((a, g, dt), eigs[0], "numpy")
        h2 = sp.lambdify((a, g, dt), eigs[1], "numpy")
        lam1_out_vals = h1(a_vals, g_vals, dt_sigmoid)
        lam2_out_vals = h2(a_vals, g_vals, dt_sigmoid)
        invertible = jnp.all(jnp.isclose(lam1_out_vals, lam1_vals) | jnp.isclose(jnp.conjugate(lam1_out_vals), lam1_vals)) \
                   & jnp.all(jnp.isclose(lam2_out_vals, lam2_vals) | jnp.isclose(jnp.conjugate(lam2_out_vals), lam2_vals))
        stable = jnp.all(jnp.abs(lam1_out_vals) < 1.0) & jnp.all(jnp.abs(lam2_out_vals) < 1.0)
        valid = jnp.all(self._is_valid_AGdt(a_vals, g_vals, dt_vals))
        print(f"Invertibility check: {invertible}")
        print(f"Stability check: {stable}")
        print(f"Validity check: {valid}")

        # Cast to real (imag part is nonzero, ~machine precision)
        a_vals = a_vals.real
        g_vals = g_vals.real

        return a_vals, g_vals, dt_vals

    def _uniform_init_AGdt(self, A_min, A_max, G_min, G_max, dt_std, key):
        """Uniform sampling over valid (A,G,dt) region"""
        bsz = 512
        done = False 
        A_vals = []
        G_vals = []
        dt_vals = []

        while not done:
            A_key, G_key, dt_key, key = jr.split(key, 4)
            A_diag = jr.uniform(A_key, shape=(bsz,)) * (A_max - A_min) + A_min
            G_diag = jr.uniform(G_key, shape=(bsz,)) * (G_max - G_min) + G_min
            dt = normal(stddev=dt_std)(dt_key, (bsz,))

            mask = self._is_valid_AGdt(A_diag, G_diag, dt)
            A_vals.extend(list(A_diag[mask]))
            G_vals.extend(list(G_diag[mask]))
            dt_vals.extend(list(dt[mask]))

            if len(A_vals) >= self.state_dim and len(G_vals) >= self.state_dim and len(dt_vals) >= self.state_dim:
                done = True

        A_diag = jnp.array(A_vals[:self.state_dim])
        G_diag = jnp.array(G_vals[:self.state_dim])
        dt = jnp.array(dt_vals[:self.state_dim])

        return A_diag, G_diag, dt
    
    def _soft_project_AGdt(self, A_diag, G_diag, dt):
        """soft projection to the _is_valid_AGdt region"""
        dt = nn.sigmoid(dt)

        G_low = dt * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)

        A_low = 1/4*G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)

        return A_diag, G_diag, dt

    def _recurrence(self, A_diag, G_diag, dt, Bu_elements):
        """Compute the LxP output of Damped-LinOSS given an LxH input.
        Args:
            A_diag          (float32):    diagonal state matrix     (P,)
            G_diag          (float32):    diagonal damping matrix   (P,)
            dt              (float32):    discretization time-step  (P,)
            Bu_elements     (complex64):  B @ u                     (L, P)
        Returns:
            ys              (float32):    SSM states                (L, P)
        """
        sql = Bu_elements.shape[0]

        I = jnp.ones_like(A_diag)
        M_11 = I - dt * G_diag
        M_12 = -dt * A_diag
        M_21 = dt
        M_22 = I

        M = jnp.concatenate([M_11, M_12, M_21, M_22])
        M_elements = M * jnp.ones((sql, 4 * self.state_dim))

        F1 = dt * Bu_elements
        F2 = jnp.zeros_like(F1)
        F = jnp.hstack((F1, F2))

        _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
        ys = xs[:, self.state_dim:]  # Position component

        return ys

    def __call__(self, input_sequence):
        # Materialize parameters
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        # Project
        A_diag, G_diag, dt = self._soft_project_AGdt(self.A_diag, self.G_diag, self.dt)

        # Apply SSM
        Bu_elements = jax.vmap(lambda u: B_complex @ u)(input_sequence)
        ys = self._recurrence(A_diag, G_diag, dt, Bu_elements)
        xs = jax.vmap(lambda x, u: (C_complex @ x).real + self.D * u)(ys, input_sequence)

        return xs
    

class LinOSSBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    layer: _AbstractLinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        layer_name: str,
        state_dim: int,
        hidden_dim: int,
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        A_min: float, 
        A_max: float, 
        G_min: float, 
        G_max: float, 
        dt_std: float, 
        drop_rate: float,
        key: PRNGKeyArray,
        **kwargs,
    ):
        ssmkey, glukey = jr.split(key, 2)
        layer_map = {
            "IM": IMLayer,
            "IMEX": IMEXLayer,
            "DampedIMEX1": DampedIMEX1Layer,
            "DampedIMEX2": DampedIMEX2Layer,
            "DampedIM": DampedIMLayer,
            "DampedEX": DampedEXLayer,
        }
        if layer_name not in layer_map.keys():
            raise KeyError(f"Layer name {layer_name} not defined.")

        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", channelwise_affine=False, mode="batch"
        )
        self.layer = layer_map[layer_name](
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            initialization=initialization,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            A_min=A_min, 
            A_max=A_max, 
            G_min=G_min, 
            G_max=G_max, 
            dt_std=dt_std, 
            key=ssmkey,
        )
        self.glu = GLU(hidden_dim, hidden_dim, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.layer(x)
        x = jax.nn.gelu(x)
        x = self.drop(x, key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class LinOSS(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: list[LinOSSBlock]
    linear_decoder: eqx.nn.Linear
    classification: bool
    tanh_output: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(
        self,
        layer_name: str,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int,
        classification: bool,
        tanh_output: bool,
        output_step: int,
        initialization: str,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        A_min: float, 
        A_max: float, 
        G_min: float, 
        G_max: float, 
        dt_std: float, 
        drop_rate: float,
        key: PRNGKeyArray,
        **kwargs,
    ):
        linear_encoder_key, *block_keys, linear_decoder_key = jr.split(
            key, num_blocks + 2
        )
        self.linear_encoder = eqx.nn.Linear(input_dim, hidden_dim, key=linear_encoder_key)
        self.blocks = [
            LinOSSBlock(
                layer_name=layer_name,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                initialization=initialization,
                r_min=r_min,
                r_max=r_max,
                theta_min=theta_min,
                theta_max=theta_max,
                A_min=A_min, 
                A_max=A_max, 
                G_min=G_min, 
                G_max=G_max, 
                dt_std=dt_std, 
                drop_rate=drop_rate,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_decoder = eqx.nn.Linear(hidden_dim, output_dim, key=linear_decoder_key)

        self.classification = classification
        self.tanh_output = tanh_output
        self.output_step = output_step

    def __call__(self, x, state, key):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)

        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = self.linear_decoder(x)
            x = jax.nn.softmax(x, axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.linear_decoder)(x)
            if self.tanh_output:
                x = jax.nn.tanh(x)

        return x, state
    
