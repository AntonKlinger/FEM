import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# FEM-Konstanten (geändert werden nur F0 später)
L0 = 1e3
E = 2.1e5
A0 = 1e3
A = A0 * np.array([2, 0.5, 1, 2, 1])

x = L0 * np.array([8, 4, 0, 0])
y = L0 * np.array([3, 0, 3, 0])
coords = np.vstack((x, y)).T

n_el = 5
n_kn = 4
fg = 2
ndof = n_kn * fg
KI = np.array([[1, 0], [2, 0], [2, 1], [1, 3], [3, 2]])  # 0-basiert

dirRB = np.array([0, 0, 0, 0, 1, 0, 1, 1], dtype=bool)
neuRB = ~dirRB

def assemble_K():
    """Gesamtsteifigkeitsmatrix berechnen."""
    K = np.zeros((ndof, ndof))
    for e in range(n_el):
        n1, n2 = KI[e]
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx**2 + dy**2)
        c = dx / L
        s = dy / L
        AE_L = A[e] * E / L
        k_local = AE_L * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        dofs = np.array([fg*n1, fg*n1+1, fg*n2, fg*n2+1])
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_local[i, j]
    return K

K = assemble_K()

def solve_fem(F0):
    """Löst das FEM-System für gegebenes F0."""
    f_ext = np.zeros(ndof)
    f_ext[1] = -F0

    u = np.zeros(ndof)
    u_known = np.zeros(np.sum(dirRB))
    u[dirRB] = u_known

    Kmod = K[neuRB][:, neuRB]
    fmod = f_ext[neuRB] - K[neuRB][:, dirRB] @ u_known

    u_unknown = np.linalg.solve(Kmod, fmod)
    u[neuRB] = u_unknown

    return u.reshape((-1, 2))

# ------------------- PLOT -------------------

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_title("Verformung des Fachwerks (mit Schieberegler für F0)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.axis('equal')
ax.grid(True)

# Initialwerte
F0_init = 1e4
scale = 1e2

u = solve_fem(F0_init)
deformed_coords = coords + scale * u

# Plotelemente initialisieren
orig_lines = []
def_lines = []

for e in range(n_el):
    n1, n2 = KI[e]
    # Original
    l1, = ax.plot([coords[n1, 0], coords[n2, 0]],
                  [coords[n1, 1], coords[n2, 1]], 'k--', lw=1)
    orig_lines.append(l1)

    # Verformt
    l2, = ax.plot([deformed_coords[n1, 0], deformed_coords[n2, 0]],
                  [deformed_coords[n1, 1], deformed_coords[n2, 1]], 'b-', lw=2)
    def_lines.append(l2)

# Slider hinzufügen
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(ax_slider, 'F0 [N]', valmin=1000, valmax=20000, valinit=F0_init, valstep=500)

def update(val):
    F0 = slider.val
    u = solve_fem(F0)
    deformed_coords = coords + scale * u
    for e in range(n_el):
        n1, n2 = KI[e]
        def_lines[e].set_xdata([deformed_coords[n1, 0], deformed_coords[n2, 0]])
        def_lines[e].set_ydata([deformed_coords[n1, 1], deformed_coords[n2, 1]])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
