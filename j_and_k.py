import numpy as np
import pandas as pd
import sympy as sp
from sympy import Matrix, symbols
from scipy.integrate import nquad
from concurrent.futures import ProcessPoolExecutor
import vegas

x, y, z = sp.symbols('x y z')
x1, y1, z1, x2, y2, z2 = sp.symbols('x1 y1 z1 x2 y2 z2')

## Molecule (e.g. water)

# N.B. distances are in Bohr radii (this will produce a final ground state energy in Hartrees)

Molecule = {
    "Atom_label": ["O", "H", "H"],
    "Z": [8, 1, 1],
    "x": [0.00000, 0.00000, 0.00000],
    "y": [0.00000, 1.42757, -1.42757],
    "z": [0.00000, -1.11294, -1.11294]
}

df = pd.DataFrame(Molecule)
print(df)


##Orbitals

#Define symbolic coordinates for integration via Sympy

x, y, z = sp.symbols('x y z')
r = (x, y, z)

def sto_3G_1s(r, coeffs, alphas, center=(0, 0, 0)):
  x, y, z = r
  Rx, Ry, Rz = center
  orbital_func = 0
  for i in range(3):
    orbital_func += coeffs[i]*((2*alphas[i])/sp.pi)**0.75 * sp.exp(-alphas[i] *((x - Rx)**2 + (y - Ry)**2 + (z - Rz)**2))
  return orbital_func

def sto_3G_2s(r, coeffs, alphas, center=(0, 0, 0)):
  x, y, z = r
  Rx, Ry, Rz = center
  orbital_func = 0
  for i in range(3):
    orbital_func += coeffs[i]*((2*alphas[i])/sp.pi)**0.75 * sp.exp(-alphas[i]*((x - Rx)**2 + (y - Ry)**2 + (z - Rz)**2))
  return orbital_func

def sto_3G_2px(r, coeffs, alphas, center=(0, 0, 0)):
  x, y, z = r
  Rx, Ry, Rz = center
  orbital_func = 0
  for i in range(3):
    orbital_func += coeffs[i]*((2**1.5*alphas[i]**2.5)/(sp.pi)**1.5)**0.5*(x - Rx)*sp.exp(-alphas[i]*((x - Rx)**2 + (y - Ry)**2 + (z - Rz)**2))
  return orbital_func

def sto_3G_2py(r, coeffs, alphas, center=(0, 0, 0)):
  x, y, z = r
  Rx, Ry, Rz = center
  orbital_func = 0
  for i in range(3):
    orbital_func += coeffs[i]*((2**1.5*alphas[i]**2.5)/(sp.pi)**1.5)**0.5*(y - Ry)*sp.exp(-alphas[i]*((x - Rx)**2 + (y - Ry)**2 + (z - Rz)**2))
  return orbital_func

def sto_3G_2pz(r, coeffs, alphas, center=(0, 0, 0)):
  x, y, z = r
  Rx, Ry, Rz = center
  orbital_func = 0
  for i in range(3):
    orbital_func += coeffs[i]*((2**1.5*alphas[i]**2.5)/(sp.pi)**1.5)**0.5*((z - Rz))*sp.exp(-alphas[i]*((x - Rx)**2 + (y - Ry)**2 + (z - Rz)**2))
  return orbital_func

n_electrons = sum(Molecule["Z"])
print("number of electrons =", n_electrons)

basis_set = [sto_3G_1s(r, center=(Molecule['x'][0], Molecule['y'][0], Molecule['z'][0]),
        coeffs = [0.15432896, 0.53532814, 0.44463454], alphas = [130.709321, 23.80886605, 6.44360831]),
             sto_3G_2s(r, center=(Molecule['x'][0], Molecule['y'][0], Molecule['z'][0]),
        coeffs = [-0.0999672, 0.39951282, 0.70011546], alphas = [5.0331513, 1.1695961, 0.38038896]),
             sto_3G_2px(r, center=(Molecule['x'][0], Molecule['y'][0], Molecule['z'][0]),
        coeffs = [0.15591627, 0.60768371, 0.39195739], alphas = [5.0331513, 1.1695961, 0.38038896]),
             sto_3G_2py(r, center=(Molecule['x'][0], Molecule['y'][0], Molecule['z'][0]),
        coeffs = [0.15591627, 0.60768371, 0.39195739], alphas = [5.0331513, 1.1695961, 0.38038896]),
             sto_3G_2pz(r, center=(Molecule['x'][0], Molecule['y'][0], Molecule['z'][0]),
        coeffs = [0.15591627, 0.60768371, 0.39195739], alphas = [5.0331513, 1.1695961, 0.38038896]),
             sto_3G_1s(r, center=(Molecule['x'][1], Molecule['y'][1], Molecule['z'][1]),
        coeffs = [0.15432896, 0.53532814, 0.44463454], alphas = [3.4252509, 0.62391372, 0.16885540]),
             sto_3G_1s(r, center=(Molecule['x'][2], Molecule['y'][2], Molecule['z'][2]),
        coeffs = [0.15432896, 0.53532814, 0.44463454], alphas = [3.4252509, 0.62391372, 0.16885540])
]

# N.B. Approximate alpha coefficients taken from https://www.basissetexchange.org/


P = [
    [4.4, -0.6, -0.6, -0.6, -0.6, 0.0, 0.0],
    [-0.6, 1.4, 0.4, 0.4, 0.4, 0.0, 0.0],
    [-0.6, 0.4, 1.4, 0.4, 0.4, 0.0, 0.0],
    [-0.6, 0.4, 0.4, 1.4, 0.4, 0.0, 0.0],
    [-0.6, 0.4, 0.4, 0.4, 1.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.22044604925e-16],
    [0.0, 0.0, 0.0, 0.0, 0.0, -2.22044604925e-16, 2.0]
]

x1, y1, z1, x2, y2, z2 = sp.symbols('x1 y1 z1 x2 y2 z2')

# J = SIGMA(lambda sigma)(P_{lambda sigma} * (mu nu | lambda sigma))
# First define the 4-index integral (mu nu | lambda sigma):

def r12_inv(x1, y1, z1, x2, y2, z2):
    return (1 / np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2))

numeric_J_integrand = np.empty((len(basis_set),)*4, dtype=object)
numeric_K_integrand = np.empty((len(basis_set),)*4, dtype=object)

def integrand_JK(x1, y1, z1, x2, y2, z2, bf1, bf2, bf3, bf4):
    r12 = sp.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return bf1.subs({x: x1, y: y1, z: z1}) * bf2.subs({x: x1, y: y1, z: z1}) * (1 / r12) * bf3.subs({x: x2, y: y2, z: z2}) * bf4.subs({x: x2, y: y2, z: z2})
    
def integrand_JK_symbolic(bf1, bf2, bf3, bf4):
    return bf1 * bf2 * (1 / sp.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)) * bf3 * bf4

numeric_J_integrand = np.empty((len(basis_set),)*4, dtype=object)
numeric_K_integrand = np.empty((len(basis_set),)*4, dtype=object)

for i in range(len(basis_set)):
    for j in range(len(basis_set)):
        for k in range(len(basis_set)):
            for l in range(len(basis_set)):
                symbolic_J_integrand = integrand_JK_symbolic(
                    basis_set[i].subs({x: x1, y: y1, z: z1}),
                    basis_set[j].subs({x: x1, y: y1, z: z1}),
                    basis_set[k].subs({x: x2, y: y2, z: z2}),
                    basis_set[l].subs({x: x2, y: y2, z: z2}),
                )
                numeric_J_integrand[i,j,k,l] = sp.lambdify((x1,y1,z1,x2,y2,z2), symbolic_J_integrand, 'numpy')

                symbolic_K_integrand = integrand_JK_symbolic(
                    basis_set[i].subs({x: x1, y: y1, z: z1}),
                    basis_set[k].subs({x: x1, y: y1, z: z1}),
                    basis_set[j].subs({x: x2, y: y2, z: z2}),
                    basis_set[l].subs({x: x2, y: y2, z: z2}),
                )
                numeric_K_integrand[i,j,k,l] = sp.lambdify((x1,y1,z1,x2,y2,z2), symbolic_K_integrand, 'numpy')


# Use vegas monte carlo integrator (faster than analytic integrators)
# Re-initialise the integrator on each run so that its monte carlo adaptive sampling doesnt carry over



def j_ij(i, j):
  j_term = 0
  integrator = vegas.Integrator([[-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2]])
  for k in range(len(basis_set)):
    for l in range(len(basis_set)):
      j_term += P[k][l] * integrator(numeric_J_integrand[i,j,k,l], nitn=10, neval=10000).mean
  return j_term


def k_ij(i, j):
  k_term = 0
  integrator = vegas.Integrator([[-2,2], [-2,2], [-2,2], [-2,2], [-2,2], [-2,2]])
  for k in range(len(basis_set)):
    for l in range(len(basis_set)):
      k_term += P[k][l] * integrator(numeric_K_integrand[i,k,j,l], nitn=10, neval=10000).mean
  return k_term


# P is just a coefficient matrix so we can multiply it by the integrands pre-integration.

def parallel_J_matrix(basis_set):
    n = len(basis_set)
    J = np.zeros((n, n))
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [[executor.submit(j_ij, i, j) for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                J[i, j] = futures[i][j].result()
    return J

def parallel_K_matrix(basis_set):
    n = len(basis_set)
    K = np.zeros((n, n))
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [[executor.submit(k_ij, i, j) for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                K[i, j] = futures[i][j].result()
    return K

J = parallel_J_matrix(basis_set)
K = parallel_K_matrix(basis_set)

F = np.zeros((n, n))
F[i, j] = T[i, j] + V[i, j] + J[i, j] - 0.5 * K[i, j]

sp.pprint(F)

