import numpy as np

kB = 1.38064852e-23 # J / K
Na = 6.022e-23 # Atoms per mol

numTimeSteps = 1000  # Parameters to change for simulation
n = 4
timeStep = .001 # dt_star

# N = n ** 3
N = 2
SIGMA = 3.405  # Angstroms
EPSILON = 1.6540e-21  # Joules
EPS_STAR = EPSILON / kB # ~120 Kelvin

rhostar = .6  # Dimensionless density of gas
rho = rhostar / (SIGMA ** 3)  # density
L = ((N / rho) ** (1 / 3))  # unit cell length
rCutoff = L / 2
TARGET_TEMP = 1.24 * EPS_STAR
cutoff = 2.5 * SIGMA  # Kelvin
# MASS = 39.9 * 10 / 6.022169 / 1.380662
# MASS = 48.07 / Na # ???
MASS_amu = 39.9 # amu
MASS_kg = MASS_amu * 1.660539e-27 # kg
# MASS = 6.63588 * 10 ** -26 # kg / atom
# MASS = 6.63588 * 10 ** -23 # g / atom
# timeStep *= np.sqrt((MASS * SIGMA ** 2) / EPS_STAR) # Picoseconds ???
#timeStep *= 1e-12 # picoseconds

def main():
    atomList = []
    # Sigma is the distance between atoms in one dimension
    #for i in range(n):
    #    for j in range(n):
    #        for k in range(n):
    #            atom = Atom(i * SIGMA, j * SIGMA, k * SIGMA)
    #            atomList.append(atom)
    atomList.append(Atom(0., 0., 0.))

    atomList.append(Atom(SIGMA/2, 0., 0.))

    text_file = open("out.xyz", "w")
    energyFile = open("Energy.txt", "w")

    for i in range(numTimeSteps):
        print(i)
        text_file.write("%i \n \n" % len(atomList))
        energyFile.write("Time: {} \n".format(i))

        for atom in atomList:  # Print locations of all molecules

            energyFile.write("Positions: %f %f %f\n" % tuple(atom.positions))
            energyFile.write("Velocities: %f %f %f\n" % tuple(atom.velocities))
            energyFile.write("Accelerations: %f %f %f\n - \n" % tuple(atom.accelerations))
            text_file.write("A %f %f %f\n" % tuple(atom.positions))

        for atom in atomList: # Find new position
            for k in range(3):
                atom.positions[k] += atom.velocities[k] * timeStep + .5 * atom.accelerations[k] * timeStep ** 2
                atom.oldAccelerations[k] = atom.accelerations[k]


        for atom in atomList: # Keep atoms inside an L by L by L dimension box
            for k in range(3):
                atom.positions[k] += -L * np.floor(atom.positions[k] / L)

        calcForces(atomList, energyFile)  # Update accelerations

        netVelocity = np.zeros(3) #

        if False and i < numTimeSteps / 2. and i != 0. and i % 5. == 0.:
            gaussianVelocities(atomList, TARGET_TEMP)

        for j in range(len(atomList)):  # update velocities
            atomList[j].velocities += (.5 * (atomList[j].accelerations + atomList[j].oldAccelerations)) * timeStep
            netVelocity += atomList[j].velocities

        # if i > numtimesteps / 2:
        # energyfile.write("time: {} \n".format(i))

        # v = np.dot(netvelocity, netvelocity)

        # netke = (.5 * mass * v)
        # energyfile.write("ke: {} \n".format(netke))

        # energyfile.write("pe: {} \n".format(netpotential))
        # energyfile.write("total energy: {} \n".format(netpotential +
        # netke))
        energyFile.write("------------------------------------ \n")

    text_file.close()
    energyFile.close()


def gaussianVelocities(atomList, targetTemp):
    instantTemp = 0
    for i in range(len(atomList)):
        # Dot product of velocity vectors
        dot = np.dot(atomList[i].velocities, atomList[i].velocities) / N
        instantTemp += MASS_kg * dot

    instantTemp /= (3 * len(atomList) - 3)
    instantTemp = np.sqrt(targetTemp / instantTemp)

    for i in range(len(atomList)):  # V = lambda * V
        atomList[i].velocities *= instantTemp


def calcForces(atomList, energyFile):
    netPotential_star = 0
    for i in range(len(atomList)):  # Set all accelerations to zero
        atomList[i].accelerations = [0., 0., 0.]

    # Iterate over all atoms in atomList
    for i in range(len(atomList)):
        for j in range(len(atomList)):
            if i != j:
                distArr = np.zeros(3)  # Position of an atom relative to another atom
                distArr = atomList[i].positions - atomList[j].positions
                # Boundary conditions - Interact through walls if closer
                distArr = distArr - L * np.round(distArr / L)
                #for k in range(3):
                #    while distArr[k] >= .5 * L:
                #        distArr[k] -= L
                #    while distArr[k] < -.5 * L:
                #        distArr[k] += L

                dot = np.dot(distArr, distArr)
                r = np.sqrt(dot)  # Distance vector magnitude
                r_star = r / SIGMA

                # potential in Joules per EPSILON
                netPotential_star += 4 * (r_star ** -12 - r_star ** -6)

                # Force in Newton sigmas per epsilon
                force_star = 24 / r_star * (2 * r_star ** -12) - (r_star ** -6)
                force = force_star * EPSILON / SIGMA
                energyFile.write("{} on {}: {} \n".format(i, j, force_star))
                # energyFile.write("----------------- \n")
                atomList[i].accelerations += (force_star * distArr / r_star)

    return netPotential_star * EPSILON


class Atom:
    def __init__(self, positionX, positionY, positionZ):
        self.positions = np.array([positionX, positionY, positionZ])
        #self.velocities = np.random.normal(size=3)
        self.velocities = np.zeros(3)
        self.accelerations = np.zeros(3)
        self.oldAccelerations = np.zeros(3)


if __name__ == "__main__":
    main()
