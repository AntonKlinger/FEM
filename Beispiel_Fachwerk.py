import numpy as np
import matplotlib.pyplot as plt

# Gegebene Werte
F0 = 1e4
L0 = 1e3
E = 2.1e5
A0 = 1e3
A = A0 * np.array([2, 0.5, 1, 2, 1])

x = L0 * np.array([8, 4, 0, 0])
y = L0 * np.array([3, 0, 3, 0])
coords = np.vstack((x, y)).T

n_el = 5
n_kn = 4

# Koinzidenzmatrix: Element zu Knoten-Zuordnung (0-indiziert!)
KI = np.array([
    [1, 0],
    [2, 0],
    [2, 1],
    [1, 3],
    [3, 2]
])

# Streckenlasten (nicht verwendet, hier 0)
qu = np.zeros(n_el)

# Anzahl Freiheitsgrade pro Knoten (2D: x und y)
fg = 2
ndof = n_kn * fg

# Gesamtsteifigkeitsmatrix und Lastvektor initialisieren
K = np.zeros((ndof, ndof))
p = np.zeros(ndof)

# Elementsteifigkeit berechnen und zusammensetzen
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

    # Zuordnung zur globalen Steifigkeitsmatrix
    dofs = np.array([fg*n1, fg*n1+1, fg*n2, fg*n2+1])
    for i in range(4):
        for j in range(4):
            K[dofs[i], dofs[j]] += k_local[i, j]

# Externe Kräfte
f_ext = np.zeros(ndof)
f_ext[1] = -F0  # Knoten 1, y-Richtung

# Randbedingungen: true = Verschiebung bekannt (Dirichlet), false = unbekannt
dirRB = np.array([0, 0, 0, 0, 1, 0, 1, 1], dtype=bool)
neuRB = ~dirRB

# Bekannte Verschiebungen (alle 0)
u = np.zeros(ndof)
u_known = np.zeros(np.sum(dirRB))
u[dirRB] = u_known

# Reduziertes Gleichungssystem
Kmod = K[neuRB][:, neuRB]
fmod = f_ext[neuRB] - K[neuRB][:, dirRB] @ u_known

# Unbekannte Verschiebungen lösen
u_unknown = np.linalg.solve(Kmod, fmod)
u[neuRB] = u_unknown

# Reaktionskräfte
f_rem = K @ u

# Ausgabe
print("\nKnotenverschiebungen (u):")
print(np.round(u.reshape((-1, 2)), 4))

print("\nReaktionskräfte (f_rem):")
print(np.round(f_rem, 2))

# ----------------- Visualisierung -------------------
scale = 1e2  # Skalierung der Verformung für bessere Sichtbarkeit
deformed_coords = coords + scale * u.reshape((-1, 2))

plt.figure(figsize=(8, 6))
for e in range(n_el):
    n1, n2 = KI[e]

    # Ursprüngliche Linien
    x_orig = [coords[n1, 0], coords[n2, 0]]
    y_orig = [coords[n1, 1], coords[n2, 1]]
    plt.plot(x_orig, y_orig, 'k--', linewidth=1, label='Original' if e == 0 else "")

    # Verformte Linien
    x_def = [deformed_coords[n1, 0], deformed_coords[n2, 0]]
    y_def = [deformed_coords[n1, 1], deformed_coords[n2, 1]]
    plt.plot(x_def, y_def, 'b-', linewidth=2, label='Verformt' if e == 0 else "")

plt.title("Verformung des Fachwerks (maßstäblich überhöht)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
