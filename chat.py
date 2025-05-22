import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Material- & Querschnittsdaten
E = 210e9          # Elastizitätsmodul in N/m² (Stahl)
A = 1e-4           # Querschnittsfläche in m²

# 1. Knotendefinition (x, y, z)
nodes = np.array([
    [0, 0, 0],      # Knoten 0
    [1, 0, 0],      # Knoten 1
    [1, 1, 0],      # Knoten 2
    [0, 1, 0],      # Knoten 3
    [0.5, 0.5, 1]   # Knoten 4 (Spitze)
])

# 2. Elementverbindungen: [Startknoten, Endknoten]
elements = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],   # Basis
    [0, 2], [1, 3],                   # Diagonalen
    [0, 4], [1, 4], [2, 4], [3, 4]    # Seiten
])

n_dof = nodes.shape[0] * 3  # 3 Freiheitsgrade pro Knoten
K_global = np.zeros((n_dof, n_dof))  # globale Steifigkeitsmatrix

# Hilfsfunktion: Elementsteifigkeitsmatrix 3D
def element_stiffness(E, A, node_i, node_j):
    xi, yi, zi = node_i
    xj, yj, zj = node_j
    L = np.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
    cx = (xj - xi) / L
    cy = (yj - yi) / L
    cz = (zj - zi) / L
    T = np.array([cx, cy, cz])
    k_local = E * A / L * (np.outer(T, T))

    # Erweiterung zu 6x6 Matrix für das ganze Element (3 DOF je Knoten)
    ke = np.zeros((6, 6))
    ke[:3, :3] = k_local
    ke[:3, 3:] = -k_local
    ke[3:, :3] = -k_local
    ke[3:, 3:] = k_local
    return ke

# 3. Globale Steifigkeitsmatrix aufbauen
for start, end in elements:
    node_i = nodes[start]
    node_j = nodes[end]
    ke = element_stiffness(E, A, node_i, node_j)

    # Freiheitsgrade des Elements (3 pro Knoten)
    dof_i = np.arange(start*3, start*3+3)
    dof_j = np.arange(end*3, end*3+3)
    dof = np.concatenate([dof_i, dof_j])

    # in globale Matrix einfügen
    for ii in range(6):
        for jj in range(6):
            K_global[dof[ii], dof[jj]] += ke[ii, jj]

# 4. Randbedingungen (Basis fixiert)
fixed_dofs = np.array([
    0, 1, 2,     # Knoten 0
    3, 4, 5,     # Knoten 1
    6, 7, 8,     # Knoten 2
    9,10,11      # Knoten 3
])
all_dofs = np.arange(n_dof)
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

# 5. Kraftvektor (z-Richtung an der Spitze)
F = np.zeros(n_dof)
F[14] = -1000  # z-Kraft auf Knoten 4 (Index 4 → 3*4+2 = 14)

# 6. Gleichungssystem lösen
K_ff = K_global[np.ix_(free_dofs, free_dofs)]
F_f = F[free_dofs]
U_f = np.linalg.solve(K_ff, F_f)

U = np.zeros(n_dof)
U[free_dofs] = U_f

# 7. Verformte Knotenpositionen
skala = 1000  # Sichtbarkeitsfaktor für Verformung
nodes_verformt = nodes + U.reshape(-1, 3) * skala

# 8. Visualisierung
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Originalstruktur (schwarz)
for start, end in elements:
    x = [nodes[start, 0], nodes[end, 0]]
    y = [nodes[start, 1], nodes[end, 1]]
    z = [nodes[start, 2], nodes[end, 2]]
    ax.plot(x, y, z, 'k-', label='Original' if start == 0 and end == 1 else "")

# Verformte Struktur (rot)
for start, end in elements:
    x = [nodes_verformt[start, 0], nodes_verformt[end, 0]]
    y = [nodes_verformt[start, 1], nodes_verformt[end, 1]]
    z = [nodes_verformt[start, 2], nodes_verformt[end, 2]]
    ax.plot(x, y, z, 'r--', label='Verformt' if start == 0 and end == 1 else "")

# Knotenpunkte
ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='black', label='Knoten (Original)')
ax.scatter(nodes_verformt[:, 0], nodes_verformt[:, 1], nodes_verformt[:, 2],
           c='red', marker='^', label='Knoten (Verformt)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D-Pyramide aus Stäben – Original vs. Verformt')
ax.legend()
plt.tight_layout()
plt.show()
