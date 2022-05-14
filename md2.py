#!/usr/bin/python3

import numpy as np

class Atom:
    def __init__(self, x, y, z):
        self.x = np.array([x, y, z])
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.old_a = np.zeros(3)
        self.mass = 1.
        self.epsilon = 1.
        self.sigma = 1.

def cubic_lattice(n, dx):
    atoms = []
    for i in range(n):  
        for j in range(n):
            for k in range(n):
                atom = Atom(i*dx, j*dx, k*dx)
                atoms.append(atom)
    return atoms

def create_velocities_random(atoms,target_temp):
    for atom in atoms:
        atom.v = np.random.rand(3) - 0.5

    instant_temp = 0.
    for atom in atoms:
        instant_temp += atom.mass * np.dot(atom.v,atom.v) / 1.0
    instant_temp /= 3*len(atoms) - 3

    for atom in atoms:
        atom.v *= np.sqrt(target_temp/instant_temp)

def create_velocities_gaussian(atoms,target_temp):
    for atom in atoms:
        atom.v = np.random.normal(size=3)

    instant_temp = 0.
    for atom in atoms:
        instant_temp += atom.mass * np.dot(atom.v,atom.v) / 1.0
    instant_temp /= 3*len(atoms) - 3

    for atom in atoms:
        atom.v *= np.sqrt(target_temp/instant_temp)

def calculate_forces(atoms):
    for atom in atoms:
        atom.a = np.zeros(3)

    for atom_i in atoms:
        for atom_j in atoms:
            if atom_i != atom_j:
                dx = atom_i.x - atom_j.x
                r = np.sqrt(np.dot(dx,dx))
                dx = dx / r
                force = 24. * atom_i.epsilon / r**2 * ( 2 * (atom_i.sigma/r)**12 - (atom_i.sigma/r)**6 )
                atom_i.a += force * dx / atom_i.mass

def integrate(atoms, dt):
    # step 1 from wikipedia https://en.wikipedia.org/wiki/Verlet_integration
    for atom in atoms:
        atom.x = atom.x + atom.v * dt + 0.5 * atom.a * dt**2        
        atom.old_a = atom.a

    # step 2
    calculate_forces(atoms)

    # step 3
    for atom in atoms:
        atom.v = atom.v + 0.5 * (atom.a + atom.old_a) * dt


def write_xyz(filename, atoms):
    out = open(filename,'a')
    out.write("{}\n\n".format(len(atoms)))
    for atom in atoms:
        out.write("H {} {} {}\n".format(atom.x[0],atom.x[1],atom.x[2]))
    out.close()



atoms = cubic_lattice(3, 1.)
create_velocities_gaussian(atoms, 0.1)
calculate_forces(atoms)

out = open("output/out.xyz",'w')
out.close()

for step in range(1000):
    print("{}/1000".format(step))  
    integrate(atoms, 0.01)
    write_xyz("out.xyz", atoms)




