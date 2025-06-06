import numpy as np
import pandas as pd
import sympy as sp
from sympy import Matrix, symbols
from scipy.integrate import nquad
from concurrent.futures import ProcessPoolExecutor
import vegas

x, y, z = sp.symbols("x y z")
x1, y1, z1, x2, y2, z2 = sp.symbols("x1 y1 z1 x2 y2 z2")
bounds = [[-3,3]] * 6


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

nbf = len(basis_set)

x1, y1, z1, x2, y2, z2 = sp.symbols('x1 y1 z1 x2 y2 z2')

# J = SIGMA(lambda sigma)(P_{lambda sigma} * (mu nu | lambda sigma))
# First define the 4-index integral (mu nu | lambda sigma):

def make_integrand(bf1, bf2, bf3, bf4):
    expr = bf1 * bf2 * (1 / sp.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)) * bf3 * bf4
    return sp.lambdify((x1, y1, z1, x2, y2, z2), expr, "numpy")

def compute_eri_entry(i, j, k, l):
    bf1 = basis_set[i].subs({x: x1, y: y1, z: z1})
    bf2 = basis_set[j].subs({x: x1, y: y1, z: z1})
    bf3 = basis_set[k].subs({x: x2, y: y2, z: z2})
    bf4 = basis_set[l].subs({x: x2, y: y2, z: z2})

    integrand = make_integrand(bf1, bf2, bf3, bf4)
    integrator = vegas.Integrator(bounds)
    result = integrator(lambda pts: integrand(*pts), nitn=10, neval=10000).mean
    return (i, j, k, l, result)

eri_tensor = np.zeros((nbf, nbf, nbf, nbf))

# Use vegas monte carlo integrator (faster than analytic integrators)
# Re-initialise the integrator on each run so that its monte carlo adaptive sampling doesnt carry over

if __name__ == "__main__":
    eri_tensor = np.zeros((nbf, nbf, nbf, nbf))
    computed_indices = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []

        for i in range(nbf):
            for j in range(i + 1):
                for k in range(nbf):
                    for l in range(k + 1):
                        # Ensure (i,j,k,l) is unique under 8-fold symmetry
                        ij = i * (i + 1) // 2 + j
                        kl = k * (k + 1) // 2 + l
                        if ij >= kl:
                            futures.append(executor.submit(compute_eri_entry, i, j, k, l))
                            computed_indices.append((i, j, k, l))

        # Fill in all symmetric permutations
        for future, (i, j, k, l) in zip(futures, computed_indices):
            val = future.result()[-1]

            perms = {
                (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)
            }
            for a, b, c, d in perms:
                eri_tensor[a, b, c, d] = val
        
np.set_printoptions(threshold=np.inf)
print(eri_tensor)
