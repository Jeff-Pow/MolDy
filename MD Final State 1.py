#MD Final



# MD attempt

# import neccesary libraries:

import numpy as np
import matplotlib.pyplot as plt
import time

# length of crystal side
n = 4
N = n**3
m = 39.9 # mass in amu 

Boltz = 1
Sigma = 3.405 #Angstrom
Epsilon = 119.8 # E/Kb (k)
rhostar = 0.6 #unsure of unit, reduced density
rho = rhostar/(Sigma**3) #Unsure of unit, density
Tstar = 1.24 #unsure of unit, reduced temperature
L = (N/rho)**(1/3) #cell length
rcut = L/2 #cutoff
target_Temp = Tstar * Epsilon # Kelvin (?)
Mass = m*10/6.022169/1.380662 #KPsA mass (u, or 1.66053906660 x 10**-27 kg)

# length of crystal side (total atoms = n **3)


KE = []
PE = []
TE = []
I_temp = []

# define what an atom is:

class Atom:
  def __init__(self, x, y, z):
    self.x = np.array([x, y, z])
    self.v = np.zeros(3)
    self.a = np.zeros(3)
    self.a0 = np.zeros(3)
    self.mass = 32. * 10./6.022169/1.380662 # reduced mass
    self.omega = 1.
    self.epsilon = 119.8 # lennard-jones parameter by Rowley, Nicholson and Parsonage
    self.sigma = 3.405 # lennard-jones parameter by Rowley, Nicholson and Parsonage
    self.x0 = np.array([x, y, z])
    self.element = "Ar"

enerlrc = (8/3) * Epsilon * np.pi * N
enerlrc = enerlrc * rhostar * ((1/3) * (Sigma/rcut)**9 - (Sigma/rcut)**3)


# define the lattice of atoms: (this one is simple cubic)

def cubic_lattice (n, dx):
  atoms = []
  #for i in range(n):
  #  for j in range(n):
  #    for k in range(n):
  #      atom = Atom(i*dx, j*dx, k*dx)
  #      atoms.append(atom)
  atoms.append(Atom(0, 0, 0))
  atoms.append(Atom(dx, 0, 0))
  return atoms

# define velocity...stuff:

def velocities_gaussian(atoms, target_temp):
  for atom in atoms:
    atom.v = np.random.normal(size=3)

  instant_temp = 0.
  for atom in atoms:
    instant_temp += atom.mass * np.dot(atom.v, atom.v) / Boltz
  instant_temp /= 3*len(atoms) - 3

  for atom in atoms:
    atom.v *= np.sqrt(target_temp/instant_temp)


def get_temp():
  instant_temp = 0.
  for atom in atoms:
    instant_temp += atom.mass * np.dot(atom.v, atom.v) / Boltz
  instant_temp /= 3*len(atoms) - 3
  return instant_temp

def temp_scaling(atoms, target_temp):
  instant_temp = 0.
  for atom in atoms:
    instant_temp += atom.mass * np.dot(atom.v, atom.v) / Boltz
  instant_temp /= 3*len(atoms) - 3


  for atom in atoms:
    atom.v *= np.sqrt(target_temp/instant_temp)
    
  


# define force calculations:

def forces(atoms, force):
  K_E = 0
  P_E = 0
  for atom in atoms:
    atom.a = np.zeros(3)
    v_sum =atom.v[0]**2 + atom.v[1]**2 + atom.v[2]**2
    K_E += 0.5 * (atom.mass) * v_sum #new line

  for atom_i in atoms:
    for atom_j in atoms:
      if atom_i != atom_j:
        dx = atom_i.x - atom_j.x
        dx = dx - L * np.round(dx / L) # PERIODIC BOUNDARY CONDITIONS IN FORCE
        r = np.sqrt(np.dot(dx,dx))
        if r < rcut:
          dx = dx/r
          f= -24*(atom.epsilon/r)*((atom.sigma/r)**6)*(1-2*(atom.sigma/r)**6)
          #force = f*atom.sigma/atom.epsilon # reduced force DOES WIERD THINGS WHEN FORCE ISN'T REDUCED
          force.write("{} \n".format(f))
          atom_i.a += f*dx/atom_i.mass

          P_E += 0.5 * 4 * atom.epsilon * ( np.power(atom.sigma/r, 12) - np.power(atom.sigma/r, 6) )
        #P_E += 0.5 * 4 * atom.epsilon * (np.power(atom.sigma/r, 12) - np.power(atom.sigma/r, 6)) #parenthesis around the np.powers make it all flip


#  for atom in atoms:
#    v_sum = atom.v[0]**2 + atom.v[1]**2 + atom.v[2]**2
#    K_E += 0.5 * (n**3 * atom.mass) * v_sum

  KE.append(K_E)
  PE.append(P_E)
  TE.append(K_E+P_E) # unreduced energy


# define velocity verlet...stuff:

def integrate(atoms, dt, force):
  #step 1
  for atom in atoms:
    atom.x = atom.x + atom.v * dt + 0.5 * atom.a * dt**2
    atom.a0 = atom.a

  # step 2
  forces(atoms, force)

  # step 3
  for atom in atoms:
    atom.v = atom.v + 0.5 * (atom.a + atom.a0) * dt
    
  
# define the output file:

def write_file(filename, atoms):
    for atom in atoms:
        if atom.x[0] > L:
            atom.x[0] = atom.x[0] - L
        if atom.x[1] > L:
            atom.x[1] = atom.x[1] - L
        if atom.x[2] > L:
            atom.x[2] = atom.x[2] - L
        if atom.x[0] < 0:
            atom.x[0] = atom.x[0] + L
        if atom.x[1] < 0:
            atom.x[1] = atom.x[1] + L
        if atom.x[2] < 0:
            atom.x[2] = atom.x[2] + L
    
    out = open(filename, 'a')
    out.write("{}\n\n".format(len(atoms)))
    for atom in atoms:
        out.write("Ar {} {} {}\n".format(atom.x[0], atom.x[1], atom.x[2]))
    out.close()

# Initialize:

# sets starting positions

atoms = cubic_lattice(n, L/n) #changed distance from 1 to 3.6

#sets starting velocities

# velocities_gaussian(atoms, Tstar*Epsilon)

#sets starting acceleration and old acceleration
force = open("forces.txt", "w")
forces(atoms, force)

#creates the file

out = open("argon-final.xyz",'w')
out.close()


# do the thing:
nsteps = 1000
print(L)
start_time = time.time()
for step in range(nsteps):
  if step!= 0 and step%50==0 and step<nsteps/2:
    temp_scaling(atoms, Tstar*Epsilon)
  integrate(atoms, 0.01, force)
  write_file("argon-final.xyz", atoms)
  I_temp.append(get_temp())
  print(step)
 
  
  
tim = (time.time() - start_time) / 60
tim  = "{:.2f}".format(tim)
#print("time elapsed (minutes):", tim)
   
write_file("argon-final.xyz", atoms)

#print('KE:',KE)
#print('PE:',PE)
#print('TE:',TE)

#PE = U
Ave_U = sum(PE[int(step/2):])/(len(PE) /2)
Red_U = (Ave_U + enerlrc)/(Epsilon * N)
Ave_LRC = enerlrc/(Epsilon * N)

print("U*.N =", Red_U)
print("LRC*/N", Ave_LRC)

# Graph energy vs time
#plt.plot(range(nsteps+1),KE,label = "KE")
#plt.plot(range(nsteps+1),PE,label = "PE")
#plt.plot(range(nsteps+1),TE,label = "TE")
#plt.plot(range(nsteps),I_temp,label = "Temp")
#plt.legend()
#plt.show()

