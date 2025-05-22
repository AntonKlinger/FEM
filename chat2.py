import numpy as np
import matplotlib.pyplot as plt

# Gitterparameter
nx, ny = 41, 41        # Gitterpunkte
dx = 1 / (nx - 1)
dy = 1 / (ny - 1)
nt = 500               # Zeitschritte
nit = 50               # Druckkorrektur-Iterationen
dt = 0.001             # Zeitschrittweite

rho = 1.0              # Dichte
nu = 0.1               # Viskosität

# Felder initialisieren
u = np.zeros((ny, nx))   # x-Geschwindigkeit
v = np.zeros((ny, nx))   # y-Geschwindigkeit
p = np.zeros((ny, nx))   # Druck

# Druckquellen-Term (Poisson)
def build_up_b(rho, dt, u, v, dx, dy):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * (
        1 / dt * (
            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
            (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
        ) -
        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
             (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
        ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2
    )
    return b

# Druck-Poisson-Gleichung
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            (dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
             dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1]) -
             b[1:-1, 1:-1] * dx**2 * dy**2) /
            (2 * (dx**2 + dy**2))
        )
        # Randbedingungen
        p[:, -1] = p[:, -2]   # rechts
        p[:, 0]  = p[:, 1]    # links
        p[0, :]  = p[1, :]    # unten
        p[-1, :] = 0          # oben
    return p

# Hauptzeitintegration
def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
            vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
            dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
            nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                  dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
            vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
            dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
            nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                  dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        )

        # Randbedingungen
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1    # obere Wand bewegt sich
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
    return u, v, p

# Simulation durchführen
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

# Gitter erzeugen für Visualisierung
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Visualisierung
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, p, alpha=0.5, cmap='plasma')
plt.colorbar(label='Druck')
plt.quiver(X, Y, u, v)
plt.title('2D Navier-Stokes: Geschwindigkeitsfeld & Druck')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()
