import numpy as np
import math

L_0 = 1000
A_0 = 1000

E = 2.1 * 100000

Knotenzahl = 5

A = np.array([2,0.5,1,2,1]) * A_0

x = np.array([8,4,0,0]) * L_0

y = np.array([3,0,3,0]) * L_0

KI = np.array([[1,2],
              [3,1],
              [3,2],
              [4,2],
              [3,4]])

L = np.zeros((5))

phy = np.zeros((5))

for i in range(Knotenzahl):
    Knoten0 = KI[i, 0]
    Knoten1 = KI[i, 1]
    x0 = x[Knoten0-1]
    y0 = y[Knoten0-1]
    x1 = x[Knoten1-1]
    y1 = y[Knoten1-1]
    phyi = math.atan2(y1-y0, x1-x0)
    Li = math.sqrt((y1-y0)**2+(x1-x0)**2)
    L[i] = Li
    phy[i] = phyi

def elementsteifigkeitsmatrix(E, A, L, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    k = (E * int(A)) / L
    print(k)
    ke = k * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    
    return ke

Ke_1 = elementsteifigkeitsmatrix(E, A[0], L[0], phy[0])
print(Ke_1)

Ke_2 = elementsteifigkeitsmatrix(E, A[1], L[1], phy[1])
print(Ke_2)

Ke_3 = elementsteifigkeitsmatrix(E, A[2], L[2], phy[2])
print(Ke_3)

Ke_4 = elementsteifigkeitsmatrix(E, A[3], L[3], phy[3])
print(Ke_4)

Ke_5 = elementsteifigkeitsmatrix(E, A[4], L[4], phy[4])
print(Ke_5)

ke_dict = {}

for i in range(Knotenzahl * 4):
    name = f"ke_{i}"
    ke_dict[name] = np.zeros((2, 2))

print(ke_dict)