import taichi as ti
import random
import math

ti.init(arch = ti.gpu)

# ===== Obstacle Parameters =====
Circle1_Center = ti.Vector([0.25, 0.6])
Circle1_Radius = 0.08

Circle2_Center = ti.Vector([0.5, 0.3])  # Moved down
Circle2_Radius = 0.08

Circle3_Center = ti.Vector([0.75, 0.5])
Circle3_Radius = 0.08

# ===== Interactive Forces =====
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

# ===== Grid Parameters =====
dim = 2
n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 4e-5  # Increased for faster simulation

# ===== Material Parameters =====
p_rho = 1
E = 1000 # stretch
gamma = 500 # shear
k = 1000 # normal
s_stiffness = k # normal stiffness for 2D curve model

# ===== Yarn Configuration =====
N_Line = 12  # number of lines
dlx = 0.015  # line space distance
ln_type2 = 200  # type2 particle count per line
start_pos = ti.Vector([0.2, 0.8])
Length = 0.75  # line length
sl = Length/ (ln_type2-1)

ln_type3 = ln_type2 - 1
n_type2 = N_Line* ln_type2
n_type3 = N_Line* ln_type3
n_segment = n_type3

# ===== Type 2 Particles (nodes) =====
x2 = ti.Vector.field(2, dtype=float, shape=n_type2) # position 
v2 = ti.Vector.field(2, dtype=float, shape=n_type2) # velocity
C2 = ti.Matrix.field(2, 2, dtype=float, shape=n_type2) # affine velocity field
volume2 =  dx*Length / (ln_type3+ln_type2)

# ===== Type 3 Particles (quadrature points) =====
x3 = ti.Vector.field(2, dtype=float, shape=n_type3) # position
v3 = ti.Vector.field(2, dtype=float, shape=n_type3) # velocity
C3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # affine velocity field
F3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # deformation gradient
D3_inv = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
D3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # Initial material directions
d3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
volume3 = volume2

# ===== Grid Fields =====
grid_v = ti.Vector.field(2, dtype= float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
grid_f = ti.Vector.field(2, dtype= float, shape = (n_grid, n_grid))

# ===== Constants =====
ROT90 = ti.Matrix([[0,-1.0],[1.0,0]])

@ti.func
def QR2(Mat): 
    """QR decomposition for 2x2 matrix using Gram-Schmidt."""
    c0 = ti.Vector([Mat[0,0],Mat[1,0]])
    c1 = ti.Vector([Mat[0,1],Mat[1,1]])
    r11 = c0.norm(1e-6)
    q0 = c0/r11
    r12 = c1.dot(q0)
    q1 = c1 - r12 * q0
    r22 = q1.norm(1e-6)
    q1/=r22
    Q = ti.Matrix.cols([q0,q1])
    R = ti.Matrix([[r11,r12],[0,r22]])
    return Q,R


@ti.kernel
def p2g():
    """Transfer particle momentum to grid."""
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C2[p]
        mass = volume2* p_rho
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v2[p] +  affine@dpos)

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C3[p]
        mass = volume3* p_rho
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v3[p] +  affine@dpos)


@ti.kernel
def grid_force():
    """
    Computes stress from QR-decomposed F (Q,R) and builds a small "A" matrix of
    partial derivatives in R-space:
      - fiber stretch penalty on r11: df/dr11 = k*(r11 - 1)
      - shear penalty on r12:      df/dr12 = gamma * r12
      - normal / cross-section penalty on r22
    Then converts to first Piola-like measure via dphi_dF = Q @ A @ R^{-T}.
    Finally distributes forces to the grid, including spring terms between type2 particles.
    """
    # Zero-grid forcing is done in Reset; add forces from particles
    for p in x3:
        # Get attached type2 endpoints for this quadrature particle
        l, n = t2_from_t3(p)

        # local weights for x3 particle
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw_dx_d = ti.Matrix.rows([fx-1.5, 2*(1.0-fx), fx-0.5]) * inv_dx

        # local weights for left and right type2 particles (forces distributed there)
        base_l = (x2[l] * inv_dx - 0.5).cast(int)
        fx_l = x2[l] * inv_dx - base_l.cast(float)
        w_l = [0.5 * (1.5 - fx_l) ** 2, 0.75 - (fx_l - 1.0) ** 2, 0.5 * (fx_l - 0.5) ** 2]

        base_n = (x2[n] * inv_dx - 0.5).cast(int)
        fx_n = x2[n] * inv_dx - base_n.cast(float)
        w_n = [0.5 * (1.5 - fx_n) ** 2, 0.75 - (fx_n - 1.0) ** 2, 0.5 * (fx_n - 0.5) ** 2]

        # QR of the deformation d = F3[p] (F in your notation)
        Q, R = QR2(F3[p])

        # Read tri-diagonal upper-triangular entries (2x2)
        r11 = R[0, 0]
        r12 = R[0, 1]
        r22 = R[1, 1]

        
        # Energy derivatives based on the paper's 2D curve model
        # f' = k*(r11-1), g' = gamma, h' = -s*(1-r22)^2
        
        # A00: derivative wrt r11 from fiber stretch + shear energy
        # A00 = f' * r11 + g' * r12^2 = k*(r11-1)*r11 + gamma*r12^2
        A00 = k * (r11 - 1.0) * r11 + gamma * r12**2

        # A01: derivative wrt r12 coupling with r22
        # A01 = g' * r12 * r22 = gamma * r12 * r22
        A01 = gamma * r12 * r22

        # A11: derivative wrt r22 from normal energy h(R3)
        # h' = -s*(1-r22)^2 for cubic compression energy
        h_prime = 0.0
        if r22 < 1.0:
            h_prime = -s_stiffness * (1.0 - r22)**2
        A11 = h_prime * r22

        # Compose the symmetric A matrix according to tech.pdf
        A = ti.Matrix([[A00, A01],
                       [A01, A11]])

        # Compute dphi/dF = Q * A * (R^{-T})
        # safe inverse for R (upper triangular 2x2)
        detR = R[0,0]*R[1,1] - R[0,1]*0.0
        # Declare RinvT before conditional to ensure it's always defined
        RinvT = ti.Matrix.zero(float, 2, 2)
        # if detR is tiny, regularize inverse
        if ti.abs(detR) < 1e-12:
            RinvT = R.transpose()  # fallback (won't be correct but stable)
        else:
            Rinv = ti.Matrix([[R[1,1], -R[0,1]],[0.0, R[0,0]]]) / detR
            RinvT = Rinv.transpose()
        dphi_dF = Q @ A @ RinvT

        # f_2: force transmitted between the two type2 vertices (vector)
        # We compute f_2 = dphi_dF * Dp_inv_c0 
        Dp_inv_c0 = ti.Vector([D3_inv[p][0,0], D3_inv[p][1,0]])  # first column of D^{-1}
        f_2 = dphi_dF @ Dp_inv_c0

        # distribute forces to the grid for the left and right type2 particles
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])

            weight_l = w_l[i][0] * w_l[j][1]
            weight_n = w_n[i][0] * w_n[j][1]
            # left gets +volume * f_2, right gets -volume * f_2 
            grid_f[base_l + offset] += volume2 * weight_l * f_2
            grid_f[base_n + offset] += -volume2 * weight_n * f_2

            # dphi w / x piece for the quadrature particle (normal/shear coupling)
            dw_dx = ti.Vector([ dw_dx_d[i, 0] * w[j][1], w[i][0] * dw_dx_d[j, 1] ])
            dp_c1 = ti.Vector([d3[p][0,1], d3[p][1,1]])
            # project dphi_dF's normal component onto weights
            dphi_dF_c1 = ti.Vector([dphi_dF[0,1], dphi_dF[1,1]])
            grid_f[base + offset] += -volume3 * dphi_dF_c1 * dw_dx.dot(dp_c1)

    # spring force (bending-like) between type2 nodes (same as before)
    for p in range((ln_type2-2)* N_Line):
        nl = p // (ln_type2-2)

        v0 = p + nl* 2
        v1 = v0+ 2

        base_0 = (x2[v0] * inv_dx - 0.5).cast(int)
        fx_0 = x2[v0] * inv_dx - base_0.cast(float)
        w_0 = [0.5 * (1.5 - fx_0) ** 2, 0.75 - (fx_0 - 1.0) ** 2, 0.5 * (fx_0 - 0.5) ** 2]

        base_1 = (x2[v1] * inv_dx - 0.5).cast(int)
        fx_1 = x2[v1] * inv_dx - base_1.cast(float)
        w_1 = [0.5 * (1.5 - fx_1) ** 2, 0.75 - (fx_1 - 1.0) ** 2, 0.5 * (fx_1 - 0.5) ** 2]


        dir_x = x2[v1] - x2[v0]
        dist = dir_x.norm(1e-9)
        dir_x /= dist
        fn = dist - 2.0 * sl
        f = -1000 * fn * dir_x

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])

            weight_0 = w_0[i][0] * w_0[j][1]
            weight_1 = w_1[i][0] * w_1[j][1]

            grid_f[base_0 + offset] -= weight_0 * f
            grid_f[base_1 + offset] += weight_1 * f




bound = 3
@ti.kernel
def grid_collision():
    """Apply boundary conditions and external forces on grid."""
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i,j] +=  grid_f[i,j] * dt
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * 9.80
            
            # Attractor force
            dist = attractor_pos[None] - ti.Vector([i * dx, j * dx])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100

            # Three circles collision
            pos = ti.Vector([i * dx, j * dx])
            
            # Circle 1
            dist1 = pos - Circle1_Center
            if dist1.x**2 + dist1.y**2 < Circle1_Radius * Circle1_Radius:
                dist1 = dist1.normalized()
                grid_v[i, j] -= dist1 * min(0, grid_v[i, j].dot(dist1))
                grid_v[i, j] *= 0.9  # friction
            
            # Circle 2
            dist2 = pos - Circle2_Center
            if dist2.x**2 + dist2.y**2 < Circle2_Radius * Circle2_Radius:
                dist2 = dist2.normalized()
                grid_v[i, j] -= dist2 * min(0, grid_v[i, j].dot(dist2))
                grid_v[i, j] *= 0.9  # friction
            
            # Circle 3
            dist3 = pos - Circle3_Center
            if dist3.x**2 + dist3.y**2 < Circle3_Radius * Circle3_Radius:
                dist3 = dist3.normalized()
                grid_v[i, j] -= dist3 * min(0, grid_v[i, j].dot(dist3))
                grid_v[i, j] *= 0.9  # friction

            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0


@ti.kernel
def g2p():
    """Transfer grid velocities back to particles."""
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v2[p] = new_v
        x2[p] += dt * v2[p]
        C2[p] = new_C

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        C3[p] = new_C


@ti.kernel
def update_state():
    """Update type3 particle states from type2 neighbors."""
    for p in x3:
        l, n = t2_from_t3(p)
        v3[p] = 0.5 * (v2[l] + v2[n])
        x3[p] = 0.5 * (x2[l] + x2[n])

        dp1 = x2[n] - x2[l]
        dp2 = ti.Vector([d3[p][0,1],d3[p][1,1]])
        dp2 += dt * C3[p]@dp2
        d3[p] = ti.Matrix.cols([dp1,dp2])
        F3[p] = d3[p]@D3_inv[p]


cf = 0.05
# cf = 0.1 # You can tune this value
@ti.kernel
def return_mapping():
    """
    Corrected plasticity model for 2D curves based on the paper.
    This projects the deformation gradient back to the feasible (yield) surface
    to model Coulomb friction.
    """
    for p in x3:
        # Decompose F into elastic rotation Q and stretch R
        Q, R = QR2(F3[p])
        r11, r12, r22 = R[0, 0], R[0, 1], R[1, 1]

        # --- Apply Plastic Yield Condition ---

        # If normal is expanding (r22 > 1), there's no contact.
        if r22 > 1.0:
            r12 = 0.0 # Release shear
            r22 = 1.0 # Forget expansion, reset to rest length
        # Handle inverted case
        elif r22 < 0:
            r12 = 0.0 # Disable shear for inverted elements
        # If normal is compressed (0 <= r22 <= 1), apply friction.
        else:
            # Check if the yield condition is violated 
            # Condition: (gamma/s)|r12| - cf*(1-r22)^2 <= 0
            if (gamma / s_stiffness) * ti.abs(r12) > cf * (1.0 - r22)**2:
                # Project r12 back to the yield surface
                yield_strength = (s_stiffness / gamma) * cf * (1.0 - r22)**2
                r12 = ti.max(-yield_strength, ti.min(yield_strength, r12))

        # Reconstruct the projected R matrix
        R_new = ti.Matrix([[r11, r12], [0.0, r22]])

        # Reconstruct the deformation gradient from the projected R
        F_new = Q @ R_new
        F3[p] = F_new

        # Update the deformed material directions `d3` to be consistent
        d3[p] = F_new @ D3[p]



@ti.kernel
def reset():
    """Clear grid fields for next timestep."""
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
        grid_f[i, j] = [0.0,0.0]


@ti.func
def t2_from_t3(index):
    """Get type2 particle indices from type3 index."""
    index += index // ln_type3
    return index, index+1

@ti.kernel
def init_scene():
    """Initialize particle positions and material states."""
    for i in range(n_type2):
        sq = i // ln_type2
        x2[i] = ti.Vector([start_pos[0]+ (i- sq* ln_type2) * sl, start_pos[1] + sq* dlx])
        v2[i] = ti.Vector([0, 0])
        C2[i] =  ti.Matrix([[0,0],[0,0]])

    for i in range(n_segment):
        l, n = t2_from_t3(i)

        x3[i] = 0.5*(x2[l] + x2[n]) # Quadrature particle position
        v3[i] = ti.Vector([0, 0])
        F3[i] = ti.Matrix([[1.0, 0.0],[0.0, 1.0] ]) # Initial deformation gradient
        C3[i] =  ti.Matrix([[0,0],[0,0]])   

        # Initial material directions (D) for the quadrature particle
        # D[:,0] is the initial tangent vector (D1 in the paper for curves)
        D_tangent = x2[n] - x2[l]
        
        # D[:,1] is the initial normal vector (D2/D3 in 3D, D1 in 2D for orthogonal)
        # It needs to be a unit vector orthogonal to D_tangent
        D_normal = ROT90 @ D_tangent.normalized() 

        # Store the initial material configuration D
        D3[i] = ti.Matrix.cols([D_tangent, D_normal])
        
        # Store its inverse for F = d @ D_inv
        D3_inv[i] = D3[i].inverse() 

        # Initialize deformed directions d for consistency, d = F @ D
        # Since F is identity, d should be D
        d3[i] = D3[i] # Initialize d3 to D3




def main():
    init_scene()

    rainbow_colors = [
        0xE74C3C,  # Red
        0xE67E22,  # Orange  
        0xF1C40F,  # Yellow
        0x27AE60,  # Green
        0x3498DB,  # Blue
        0x9B59B6,  # Purple
    ]
    
    randColor = []
    for i in range(N_Line):
        pos = i / (N_Line - 1) * (len(rainbow_colors) - 1)
        idx = int(pos)
        frac = pos - idx
        
        if idx < len(rainbow_colors) - 1:
            c1 = rainbow_colors[idx]
            c2 = rainbow_colors[idx + 1]
            
            r1, g1, b1 = (c1 >> 16) & 0xFF, (c1 >> 8) & 0xFF, c1 & 0xFF
            r2, g2, b2 = (c2 >> 16) & 0xFF, (c2 >> 8) & 0xFF, c2 & 0xFF
            
            r = int(r1 + (r2 - r1) * frac)
            g = int(g1 + (g2 - g1) * frac)
            b = int(b1 + (b2 - b1) * frac)
            
            color = (r << 16) | (g << 8) | b
        else:
            color = rainbow_colors[-1]
        
        randColor.append(color)

    gui = ti.GUI("Anisotropic Yarn 2D - Three Circles", (512, 512))
    print("[Hint] Use left/right mouse buttons to attract/repel yarns.")
    
    while True:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
                
        # Get mouse position and button state
        mouse = gui.get_cursor_pos()
        attractor_pos[None] = [mouse[0], mouse[1]]
        attractor_strength[None] = 0
        if gui.is_pressed(ti.GUI.LMB):
            attractor_strength[None] = 1
        if gui.is_pressed(ti.GUI.RMB):
            attractor_strength[None] = -1
            
        for _ in range(50):  # Increased for smoother real-time speed
            reset()
            p2g()
            grid_force()
            grid_collision()
            g2p()
            update_state()
            return_mapping()

        gui.clear(0xFFFFFF)  # White background

        # Draw three circles
        gui.circle(Circle1_Center, radius=Circle1_Radius * 512, color=0x333333)
        gui.circle(Circle2_Center, radius=Circle2_Radius * 512, color=0x333333)
        gui.circle(Circle3_Center, radius=Circle3_Radius * 512, color=0x333333)
        
        # Draw attractor cursor
        if attractor_strength[None] != 0:
            color = 0x3498DB if attractor_strength[None] > 0 else 0xE74C3C
            gui.circle((mouse[0], mouse[1]), color=color, radius=10)

        x2_ny = x2.to_numpy()
        for li in range(N_Line):
            gui.circles(x2_ny[li* ln_type2 : (li+1)*ln_type2], radius=2, color= randColor[li])
        gui.show()

if __name__ == "__main__":
    main()

