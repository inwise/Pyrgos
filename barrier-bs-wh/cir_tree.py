from numpy import array, pi, sqrt, exp, zeros

T = 1.0
v0 = 0.01
kappa = 2.0 # mean rev
theta = 0.01 # long term
omega = 0.2 # vol
N = 3


def compute_f(r, omega):
    return 2*sqrt(r)/omega

def compute_v(R, omega):
    if R > 0:
        return (R**2) * (omega**2)/4.0
    else:
        return 0.0

def build_volatility_tree(T, v0, kappa, theta, omega, N):
    div_by_zero_counter = 0
    f = zeros((N+1, N+1))
    f[0, 0] = compute_f(v0, omega)
    dt = float(T/N)
    sqrt_dt = sqrt(dt)
    V = zeros((N+1, N+1))
    V[0, 0] = compute_v(f[0, 0], omega)
    f[1, 0] = f[0, 0]-sqrt_dt
    f[1, 1] = f[0, 0]+sqrt_dt
    V[1, 0] = compute_v(f[1, 0], omega)
    V[1, 1] = compute_v(f[1, 1], omega)

    for i in range(1, N):
        for j in range(i+1):
            f[i+1, j] = f[i, j] - sqrt_dt
            f[i+1, j+1] = f[i, j] + sqrt_dt
            V[i+1, j] = compute_v(f[i+1, j], omega)
            V[i+1, j+1] = compute_v(f[i+1, j+1], omega)

    f_down = zeros((N+1, N+1))
    f_up = zeros((N+1, N+1))
    pu_f = zeros((N+1, N+1))
    pd_f = zeros((N+1, N+1))
    for i in range(0, N):
        for j in range(i+1):
            # /*Compute mu_f*/
            v_curr = V[i][j]
            mu_r = kappa*(theta-v_curr)
            z = 0
            while V[i, j] + mu_r*dt < V[i+1, j-z] and j-z >= 0:
                z += 1
            f_down[i, j] = -z
            Rd = V[i+1, j-z]  # the next low vertice we can reach
            z = 0
            while V[i, j] + mu_r*dt > V[i+1, j+z] and j+z <= i:
                z += 1
            Ru = V[i+1, j+z]  # the next high vertice we can reach
            f_up[i, j] = z
            if Ru == Rd:
                div_by_zero_counter += 1
            pu_f[i, j] = (V[i, j]+mu_r*dt-Rd)/(Ru-Rd)

            if Ru-1.e-9 > V[i+1, i+1] or j+f_up[i][j] > i+1:
                pu_f[i][j] = 1
                f_up[i][j] = i+1-j
                f_down[i][j] = i-j

            if Rd+1.e-9 < V[i+1, 0] or j+f_down[i, j] < 0:
                pu_f[i, j] = 0
                f_up[i, j] = 1 - j
                f_down[i, j] = 0 - j
            pd_f[i, j] = 1 - pu_f[i][j]
    return [V, pu_f, pd_f, f_up, f_down]

def calculate_bond_price(T, v0, kappa, theta, omega, N):
    all_trees = build_volatility_tree(T, v0, kappa, theta, omega, N)
    v = all_trees[0]
    pu_f = all_trees[1]
    pd_f = all_trees[2]
    f_up = all_trees[3]
    f_down = all_trees[4]
    bond_price_matrix = zeros((N+1, N+1))
    for k in range(N+1):
        bond_price_matrix[N, k] = v[N, k]

    for n in range(N-1, -1, -1):
        for k in range(n+1):
            k_u = k + f_up[n, k]
            k_d = k + f_down[n, k]
            bond_price_matrix[n, k] = pu_f[n, k] * bond_price_matrix[n+1, k_u] + pd_f[n, k] * bond_price_matrix[n+1, k_d]
    return bond_price_matrix[0, 0]

print(calculate_bond_price(T, v0, kappa, theta, omega, N))
