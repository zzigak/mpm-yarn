import taichi as ti, os, imageio, shutil
import math

ti.init(arch=ti.gpu)

# ===== Grid Parameters =====
dim = 3
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-5  # Even smaller timestep for stability

# ===== Material Parameters =====
p_rho = ti.field(dtype=float, shape=())
E = ti.field(dtype=float, shape=())
gamma = ti.field(dtype=float, shape=())
k = ti.field(dtype=float, shape=())
mu = ti.field(dtype=float, shape=())
lam = ti.field(dtype=float, shape=())
# Anisotropic friction coefficients
alpha_friction = ti.field(dtype=float, shape=()) # Perpendicular to fiber
beta_friction = ti.field(dtype=float, shape=())  # Parallel to fiber

# Initialize default values - Knitted Poncho parameters
p_rho[None] = 4    # ρ (rho) - density
E[None] = 500        # E - Young's modulus  
gamma[None] = 500    # γ (gamma) - shear stiffness
k[None] = 2000       # k - fiber stiffness
mu[None] = 100.0    # Shear modulus - increased for stiffer yarn
lam[None] = 100.0   # Bulk modulus - increased for stiffer yarn
alpha_friction[None] = 0.3 # Corresponds to paper's Drucker-Prager friction
beta_friction[None] = 0.1  # Friction for shear along the fiber

# ===== Yarn Configuration =====
ln_type2 = 100  # 50 particles, closer together
start_pos = ti.Vector([0.35, 0.3, 0.5])  # Half length, at height 0.3
end_pos = ti.Vector([0.65, 0.3, 0.5])    # Half length, at height 0.3
Length = (end_pos - start_pos).norm()
sl = Length / (ln_type2 - 1)

ln_type3 = ln_type2 - 1
n_type2 = ln_type2
n_type3 = ln_type3
n_segment = n_type3

# ===== Type 2 Particles (nodes) =====
x2 = ti.Vector.field(dim, dtype=float, shape=n_type2)  # position
v2 = ti.Vector.field(dim, dtype=float, shape=n_type2)  # velocity
C2 = ti.Matrix.field(dim, dim, dtype=float, shape=n_type2)  # affine velocity field
volume2 = (dx ** 2) * Length / (ln_type3 + ln_type2)

# ===== Type 3 Particles (quadrature points) =====
x3 = ti.Vector.field(dim, dtype=float, shape=n_type3)  # position
v3 = ti.Vector.field(dim, dtype=float, shape=n_type3)  # velocity
C3 = ti.Matrix.field(dim, dim, dtype=float, shape=n_type3)  # affine velocity field
F3 = ti.Matrix.field(dim, dim, dtype=float, shape=n_type3)  # deformation gradient
D3_inv = ti.Matrix.field(dim, dim, dtype=float, shape=n_type3)
D3 = ti.Matrix.field(dim, dim, dtype=float, shape=n_type3)  # Initial material directions
d3 = ti.Matrix.field(dim, dim, dtype=float, shape=n_type3)
volume3 = volume2

# ===== Grid Fields =====
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))
grid_f = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))

# ===== Visualization Helpers =====
attachment_pts = ti.Vector.field(3, dtype=float, shape=2)
floor_vertices = ti.Vector.field(3, dtype=float, shape=4)
floor_indices = ti.field(dtype=int, shape=6)
floor_colors = ti.Vector.field(3, dtype=float, shape=4)

@ti.func
def QR3(Mat):
    """QR decomposition for 3x3 matrix using Gram-Schmidt."""
    c0 = ti.Vector([Mat[0,0], Mat[1,0], Mat[2,0]])
    c1 = ti.Vector([Mat[0,1], Mat[1,1], Mat[2,1]])
    c2 = ti.Vector([Mat[0,2], Mat[1,2], Mat[2,2]])
    
    r11 = c0.norm(1e-9)
    q0 = c0 / r11 if r11 > 1e-9 else ti.Vector([1.0, 0.0, 0.0])
    
    r12 = c1.dot(q0)
    q1_hat = c1 - r12 * q0
    r22 = q1_hat.norm(1e-9)
    q1 = q1_hat / r22 if r22 > 1e-9 else ti.Vector([0.0, 1.0, 0.0])

    r13 = c2.dot(q0)
    r23 = c2.dot(q1)
    q2_hat = c2 - r13 * q0 - r23 * q1
    r33 = q2_hat.norm(1e-9)
    q2 = q2_hat / r33 if r33 > 1e-9 else ti.Vector([0.0, 0.0, 1.0])

    Q = ti.Matrix.cols([q0, q1, q2])
    R = ti.Matrix([[r11, r12, r13], [0, r22, r23], [0, 0, r33]])
    return Q, R

@ti.func
def svd2d(Mat):
    """SVD for 2x2 matrix."""
    U, sig, V = ti.svd(Mat)
    return U, sig, V

@ti.kernel
def p2g():
    """Transfer particle momentum to grid."""
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        affine = C2[p]
        mass = volume2 * p_rho[None]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v2[p] + affine @ dpos)

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        affine = C3[p]
        mass = volume3 * p_rho[None]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v3[p] + affine @ dpos)
@ti.func
def t2_from_t3(index):
    """
    Get type2 (vertex) particle indices from a type3 (quadrature) index for a simple yarn.
    This logic assumes a linear chain of segments.
    """
    # This logic from the 2D code is still valid for a 1D chain of particles in 3D space.
    # It accounts for the fact that there is one fewer segment than vertices.
    # For a simple line of yarn, the quadrature point 'p' sits between vertex 'p' and 'p+1'.
    return index, index + 1

@ti.kernel
def grid_force():
    """
    Computes 3D curve forces and distributes them to the grid based on the
    anisotropic elastoplastic model.
    """
    for p in x3:
        # Get the indices of the two endpoint vertices (type ii) for this segment (type iii)
        l, n = t2_from_t3(p)

        # --- Preliminaries: 3D Interpolation Weights and Gradients ---
        # These are standard MPM calculations, extended to 3D.
        # Weights for the left vertex 'l'
        base_l = (x2[l] * inv_dx - 0.5).cast(int)
        fx_l = x2[l] * inv_dx - base_l.cast(float)
        w_l = [0.5 * (1.5 - fx_l)**2, 0.75 - (fx_l - 1)**2, 0.5 * (fx_l - 0.5)**2]

        # Weights for the right vertex 'n'
        base_n = (x2[n] * inv_dx - 0.5).cast(int)
        fx_n = x2[n] * inv_dx - base_n.cast(float)
        w_n = [0.5 * (1.5 - fx_n)**2, 0.75 - (fx_n - 1)**2, 0.5 * (fx_n - 0.5)**2]

        # Weights and weight gradients for the quadrature particle 'p'
        base_q = (x3[p] * inv_dx - 0.5).cast(int)
        fx_q = x3[p] * inv_dx - base_q.cast(float)
        w_q = [0.5 * (1.5 - fx_q)**2, 0.75 - (fx_q - 1)**2, 0.5 * (fx_q - 0.5)**2]
        dw_dx_d_q = ti.Matrix.rows([fx_q - 1.5, 2.0 * (1.0 - fx_q), fx_q - 0.5]) * inv_dx

        # --- Step 1: QR Decomposition ---
        # Decomposes F into rotation Q and stretch/shear R.
        # The energy potential is defined in terms of R to be rotation-invariant.
        Q, R = QR3(F3[p])

        # --- Step 2: Construct the 'A' Matrix ---
        # The matrix A relates energy derivatives to the final stress tensor. 
        r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
        r22, r23, r33 = R[1, 1], R[1, 2], R[2, 2]
        
        # R_hat represents the 2D deformation of the yarn's cross-section. 
        # R is upper-triangular, so R[2,1] must be 0.0. 
        R_hat = ti.Matrix([[r22, r23], [0.0, r33]])
        
        # Energy derivatives from the potential ψ(R) = f(R1) + g(R2) + h(R3). 
        f_prime = k[None] * (r11 - 1.0)  # From f(R1) = k/2 * (r11-1)^2 
        g_prime = gamma[None]            # From g(R2) = γ/2 * (r12^2 + r13^2) 
        
        # Kirchhoff stress (tau_hat) for the cross-section, corresponding to the h(R3) term. 
        # This is the P_hat * R_hat^T term in the 'A' matrix formula. 
        U_hat, sig_hat, V_hat = svd2d(R_hat)
        e = ti.log(ti.Vector([sig_hat[0,0], sig_hat[1,1]]))
        # e = ti.log(ti.max(ti.Vector([sig_hat[0,0], sig_hat[1,1]]), 1e-6))
        tau_hat_diag = 2.0 * mu[None] * e + lam[None] * (e[0] + e[1]) * ti.Vector([1.0, 1.0])
        tau_hat = U_hat @ ti.Matrix([[tau_hat_diag[0], 0.0], [0.0, tau_hat_diag[1]]]) @ V_hat.transpose()
        
        # Assemble the 3x3 'A' matrix from its blocks, following the formula in the tech supplement. 
        r_vec = ti.Vector([r12, r13])
        A = ti.Matrix.zero(float, 3, 3)
        A[0, 0] = f_prime * r11 + g_prime * r_vec.dot(r_vec) # Top-left block
        # Compute A12 = g_prime * r_vec^T @ R_hat^T element-wise
        R_hat_T = R_hat.transpose()
        A12 = ti.Vector([g_prime * r_vec.dot(ti.Vector([R_hat_T[0,0], R_hat_T[1,0]])), 
                         g_prime * r_vec.dot(ti.Vector([R_hat_T[0,1], R_hat_T[1,1]]))])
        A[0, 1], A[0, 2] = A12[0], A12[1]
        A21 = g_prime * R_hat @ r_vec # Bottom-left block
        A[1, 0], A[2, 0] = A21[0], A21[1]
        A22 = tau_hat # Bottom-right block
        A[1, 1], A[1, 2] = A22[0, 0], A22[0, 1]
        A[2, 1], A[2, 2] = A22[1, 0], A22[1, 1]
        
        # --- Step 3: First Piola-Kirchhoff Stress ---
        # Compute P = Q * A * R^(-T). 
        dphi_dF = Q @ A @ R.inverse().transpose()

        # --- Step 4: Part 1 - In-Manifold Force (f^(ii)) ---
        # This force arises from stretching and shearing along the yarn.
        # It is computed for the vertex particles (type ii).
        # Implements Equation (13) from the tech supplement. 
        Dp_inv_c0 = ti.Vector([D3_inv[p][0,0], D3_inv[p][1,0], D3_inv[p][2,0]])
        force_on_vertex_n = -volume3 * (dphi_dF @ Dp_inv_c0)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight_l = w_l[i][0] * w_l[j][1] * w_l[k][2]
            weight_n = w_n[i][0] * w_n[j][1] * w_n[k][2]
            
            # Distribute forces to the grid. The right vertex 'n' gets the calculated force,
            # and the left vertex 'l' gets the equal and opposite force. 
            grid_f[base_l + offset] -= force_on_vertex_n * weight_l
            grid_f[base_n + offset] += force_on_vertex_n * weight_n

        # --- Step 5: Part 2 - Normal-Space Force (f^(iii)) ---
        # This force handles collision and repulsion in directions normal to the yarn.
        # It is computed at the quadrature particle's (type iii) location.
        # Implements the second term of Equation (15) from the tech supplement. [cite: 979]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dw_dx = ti.Vector([
                dw_dx_d_q[i,0] * w_q[j][1] * w_q[k][2],
                w_q[i][0] * dw_dx_d_q[j,1] * w_q[k][2],
                w_q[i][0] * w_q[j][1] * dw_dx_d_q[k,2]
            ])
            
            force_contrib = ti.Vector.zero(float, 3)
            # Sum over the two normal directions (ε = 2, 3). [cite: 979]
            for eps in ti.static(range(1, 3)):
                d_eps = ti.Vector([d3[p][0,eps], d3[p][1,eps], d3[p][2,eps]])
                dphi_dF_eps = ti.Vector([dphi_dF[0,eps], dphi_dF[1,eps], dphi_dF[2,eps]])
                
                # Inner sum: Σ_κ (∂w/∂x_κ * d_κε)
                term = dw_dx.dot(d_eps)
                force_contrib += dphi_dF_eps * term
            
            grid_f[base_q + offset] -= volume3 * force_contrib



bound = 3
@ti.kernel
def grid_collision():
    """Apply boundary conditions and external forces on grid."""
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            # 1. Add elastic force impulse to the momentum
            grid_v[i, j, k] += grid_f[i, j, k] * dt

            # 2. Add gravity impulse (mass * acceleration * dt) to the momentum
            grid_v[i, j, k].y -= grid_m[i, j, k] * dt * 9.81

            # 3. NOW, convert the final momentum to velocity
            grid_v[i, j, k] /= grid_m[i, j, k]

            # 4. Apply boundary conditions to the final VELOCITY
            # Note: Using a simple zeroing condition for now, not a bounce.
            if i < bound and grid_v[i, j, k].x < 0: grid_v[i, j, k].x = 0
            if i > n_grid - bound and grid_v[i, j, k].x > 0: grid_v[i, j, k].x = 0
            if k < bound and grid_v[i, j, k].z < 0: grid_v[i, j, k].z = 0
            if k > n_grid - bound and grid_v[i, j, k].z > 0: grid_v[i, j, k].z = 0
            if j < bound and grid_v[i, j, k].y < 0: grid_v[i, j, k].y = 0
            if j > n_grid - bound and grid_v[i, j, k].y > 0: grid_v[i, j, k].y = 0




@ti.kernel
def g2p():
    """Transfer grid velocities back to particles."""
    damping = 1.0 # Higher damping for stability
    for p in x2:
            
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx)
            g_v = grid_v[base + offset]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        
        v2[p] = new_v
        x2[p] += dt * v2[p]
        C_sym = 0.5 * (new_C + new_C.transpose())
        C_skw = 0.5 * (new_C - new_C.transpose())
        C2[p] = C_skw + C_sym * (1 - damping)

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_C = ti.Matrix.zero(float, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx)
            g_v = grid_v[base + offset]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        
        C_sym = 0.5 * (new_C + new_C.transpose())
        C_skw = 0.5 * (new_C - new_C.transpose())
        C3[p] = C_skw + C_sym * (1-damping)

@ti.kernel
def update_state():
    """Update type3 particle states from type2 neighbors."""
    for p in x3:
        l, n = p, p + 1
        v3[p] = 0.5 * (v2[l] + v2[n])
        x3[p] = 0.5 * (x2[l] + x2[n])

        dp1 = x2[n] - x2[l]
        dp2 = ti.Vector([d3[p][0,1], d3[p][1,1], d3[p][2,1]]) + dt * C3[p] @ ti.Vector([d3[p][0,1], d3[p][1,1], d3[p][2,1]])
        dp3 = ti.Vector([d3[p][0,2], d3[p][1,2], d3[p][2,2]]) + dt * C3[p] @ ti.Vector([d3[p][0,2], d3[p][1,2], d3[p][2,2]])
        
        d3[p] = ti.Matrix.cols([dp1, dp2, dp3])
        F3[p] = d3[p] @ D3_inv[p]

@ti.kernel
def return_mapping():
    """Anisotropic plasticity model for 3D curves."""
    for p in x3:
        Q, R_trial = QR3(F3[p])
        R = R_trial

        # Perpendicular friction (cross-section)
        R_hat = ti.Matrix([[R[1,1], R[1,2]], [0, R[2,2]]])
        U_hat, sig_hat, V_hat = svd2d(R_hat)
        clamped_s0 = ti.max(sig_hat[0,0], 1e-6)
        clamped_s1 = ti.max(sig_hat[1,1], 1e-6)
        e = ti.log(ti.Vector([clamped_s0, clamped_s1])) 
        # e_trial = ti.Vector([ti.log(sig_hat[0,0]), ti.log(sig_hat[1,1])])
        # e = e_trial

        if e[0] + e[1] < 0: # Compression
            # Drucker-Prager yield condition for 2D cross-section
            yield_val = (e[0] - e[1]) + alpha_friction[None] * (e[0] + e[1])
            if yield_val > 0:
                # Project back to yield surface
                e_pl_norm = ti.Vector([1.0 + alpha_friction[None], -1.0 + alpha_friction[None]])
                e -= yield_val / e_pl_norm.norm_sqr() * e_pl_norm
        
        sig_new = ti.exp(e)
        R_hat_new = U_hat @ ti.Matrix([[sig_new[0], 0.0], [0.0, sig_new[1]]]) @ V_hat.transpose()
        # R[1,1], R[1,2], R[2,1], R[2,2] = R_hat_new[0,0], R_hat_new[0,1], R_hat_new[1,0], R_hat_new[1,1]
        R[1,1], R[1,2] = R_hat_new[0,0], R_hat_new[0,1]
        # R[2,1] MUST remain 0.
        R[2,2] = R_hat_new[1,1] 
        
        # Parallel friction (shear along fiber)
        sigma22_plus_sigma33 = 2.0 * mu[None] * (e[0] + e[1]) + 2.0 * lam[None] * (e[0] + e[1])
        shear_mag = ti.sqrt(R[0,1]**2 + R[0,2]**2)
        yield_val_parallel = shear_mag * gamma[None] + beta_friction[None] * 0.5 * sigma22_plus_sigma33

        if yield_val_parallel > 0:
            scale = ti.max(0.0, 1.0 - yield_val_parallel / (shear_mag * gamma[None]))
            R[0,1] *= scale
            R[0,2] *= scale

        # Reconstruct F
        F_new = Q @ R
        F3[p] = F_new
        d3[p] = F_new @ D3[p]

@ti.kernel
def reset():
    """Clear grid fields for next timestep."""
    for i, j, k in grid_m:
        grid_v[i, j, k] = ti.Vector([0, 0, 0])
        grid_m[i, j, k] = 0
        grid_f[i, j, k] = ti.Vector([0, 0, 0])

@ti.kernel
def init_scene():
    """Initialize all particles horizontally straight at fixed height."""
    # Lay all particles horizontally straight at fixed height
    fixed_height = 0.3
    
    for i in range(n_type2):
        # Parameter t goes from 0 to 1 along the yarn
        t = i / (n_type2 - 1)
        
        # Linear interpolation for x (horizontal span)
        x_pos = start_pos[0] + t * (end_pos[0] - start_pos[0])
        
        # All particles start at fixed height
        y_pos = fixed_height
        
        pos = ti.Vector([
            x_pos,
            y_pos,
            start_pos[2]  # Keep z constant
        ])
        
        x2[i] = pos
        v2[i] = ti.Vector([0, 0, 0])
        C2[i] = ti.Matrix.zero(float, dim, dim)

    for i in range(n_segment):
        l, n = i, i + 1
        x3[i] = 0.5 * (x2[l] + x2[n])
        v3[i] = ti.Vector([0, 0, 0])
        F3[i] = ti.Matrix.identity(float, dim)
        C3[i] = ti.Matrix.zero(float, dim, dim)

        # Create orthonormal basis for initial material frame D
        D_tangent = (x2[n] - x2[l]).normalized()
        up = ti.Vector([0.0, 1.0, 0.0])
        if ti.abs(D_tangent.dot(up)) > 0.99:
            up = ti.Vector([1.0, 0.0, 0.0])
        
        D_normal1 = D_tangent.cross(up).normalized()
        D_normal2 = D_tangent.cross(D_normal1)
        
        D3[i] = ti.Matrix.cols([x2[n] - x2[l], D_normal1, D_normal2])
        D3_inv[i] = D3[i].inverse()
        d3[i] = D3[i]

    attachment_pts[0] = start_pos
    attachment_pts[1] = end_pos

    # Floor quad
    floor_vertices[0] = [0.0, 0.0, 0.0]
    floor_vertices[1] = [1.0, 0.0, 0.0]
    floor_vertices[2] = [1.0, 0.0, 1.0]
    floor_vertices[3] = [0.0, 0.0, 1.0]
    floor_indices[0], floor_indices[1], floor_indices[2] = 0, 1, 2
    floor_indices[3], floor_indices[4], floor_indices[5] = 0, 2, 3
    for i in range(4):
        floor_colors[i] = [0.3, 0.3, 0.3]


@ti.kernel
def fix_endpoints():
    """Fix only the left edge point to simulate attached end."""
    # Fix first particle (left end) only
    x2[0] = start_pos
    v2[0] = ti.Vector([0, 0, 0])



def main():
    if os.path.exists('frames'):
        shutil.rmtree('frames')
    os.makedirs('frames', exist_ok=True)

    init_scene()

    window = ti.ui.Window('3D Yarn Simulation', (2048, 2048), show_window=False)
    canvas, scene = window.get_canvas(), window.get_scene()
    canvas.set_background_color((1.0, 1.0, 1.0))
    cam = ti.ui.Camera()
    cam.position(0.5, 0.5, 2)
    cam.lookat(0.5, 0.5, 0)
    cam.up(0, 1, 0)
    
    frame_count = 0
    while window.running and frame_count < 400:  # Reduced from 500 to 200 frames
        for _ in range(100):  # More substeps for stability
            reset()
            p2g()
            grid_force()
            grid_collision()
            g2p()
            update_state()
            return_mapping()
            fix_endpoints()  

        scene.set_camera(cam)
        scene.ambient_light((1.0,)*3)
        # add multiple lights to brighten the scene
        scene.point_light(pos=(0.5, 1.5, 1.0), color=(1, 1, 1))
        scene.point_light(pos=(0.2, 0.8, -0.5), color=(1, 1, 1))
        scene.point_light(pos=(0.8, 0.8, 0.8), color=(1, 1, 1))
        
        scene.particles(x2, color=(0.2, 0.8, 0.2), radius=0.003)
        
        # Draw floor as mesh instead of lines
        scene.mesh(floor_vertices, floor_indices, per_vertex_color=floor_colors)
        
        canvas.scene(scene)
        window.save_image(f'frames/{frame_count:04d}.png')
        frame_count += 1

    # Create video
    imgs = [f'frames/{f:04d}.png' for f in range(frame_count)]
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = f'yarn_3d_horizontal_{timestamp}.mp4'
    imageio.mimsave(video_name, [imageio.imread(f) for f in imgs], fps=60)
    print(f'{video_name} saved')

if __name__ == "__main__":
    main()

