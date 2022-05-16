import numpy as np

kB = 1.38064852e-23 # J / K
Na = 6.022e-23 # Atoms per mol

numTimeSteps = 5000  # Parameters to change for simulation
n = 3
timeStep = .001 # dt_star

N = n ** 3
SIGMA = 3.405  # Angstroms
EPSILON = 1.6540e-21  # Joules
EPS_STAR = EPSILON / kB # ~120 Kelvin

rhostar = .8  # Dimensionless density of gas
rho = rhostar / (SIGMA ** 3)  # density
L = ((N / rho) ** (1 / 3))  # unit cell length
rCutoff = L / 2
TARGET_TEMP = 1.24 * EPS_STAR
cutoff = 2.5 * SIGMA  # Kelvin
MASS = 39.9 * 10 / 6.022169 / 1.380662
# MASS = 48.07 / Na # ???
MASS_amu = 39.9 # amu
MASS_kg = MASS_amu * 1.660539e-27 # kg
# MASS = 6.63588 * 10 ** -26 # kg / atom
# MASS = 6.63588 * 10 ** -23 # g / atom
timeStep *= np.sqrt((MASS * SIGMA ** 2) / EPS_STAR) # Picoseconds ???
#timeStep *= 1e-12 # picoseconds

def main():
    atomList = []
    # Sigma is the distance between atoms in one dimension
    for i in range(n):
        for j in range(n):
            for k in range(n):
                atom = Atom(i * SIGMA, j * SIGMA, k * SIGMA)
                atomList.append(atom)
    thermostat(atomList, TARGET_TEMP)
    #atomList.append(Atom(0., 0., 0.))

    #atomList.append(Atom(SIGMA, 0., 0.))

    text_file = open("out.xyz", "w")
    energyFile = open("Energy.txt", "w")
    conserved = open("Conservation.txt", "w")

    for i in range(numTimeSteps):
        print(i)
        text_file.write("%i \n \n" % len(atomList))
        energyFile.write("Time: {} \n".format(i))

        for atom in atomList:  # Print locations of all molecules

            energyFile.write("Positions: %f %f %f\n" % tuple(atom.positions))
            energyFile.write("Velocities: %f %f %f\n" % tuple(atom.velocities))
            energyFile.write("Accelerations: %f %f %f\n - \n" % tuple(atom.accelerations))
            text_file.write("A %f %f %f\n" % tuple(atom.positions))

        netPotential = calcForces(atomList,energyFile)

        for atom in atomList: # Find new position
            for k in range(3):
                atom.positions[k] += atom.velocities[k] * timeStep + .5 * atom.accelerations[k] * timeStep ** 2
                atom.oldAccelerations[k] = atom.accelerations[k]

        for atom in atomList: # Keep atoms inside an L by L by L dimension box
            for k in range(3):
                atom.positions[k] += -L * np.floor(atom.positions[k] / L)

        # calcForces(atomList, energyFile)  # Update accelerations

        netVelocity = np.zeros(3)



        for atom in atomList:  # update velocities
            atom.velocities += (.5 * (atom.accelerations + atom.oldAccelerations)) * timeStep
            netVelocity += atom.velocities

        if i < numTimeSteps / 2. and i != 0. and i % 5. == 0.:
            thermostat(atomList, TARGET_TEMP)

        if i > numTimeSteps / 2:
            conserved.write("time: {} \n".format(i))

            v2 = np.dot(netVelocity, netVelocity)

            netKE = (.5 * MASS * v2)
            conserved.write("KE: {} \n".format(netKE))

            conserved.write("PE: {} \n".format(netPotential))
            conserved.write("Total energy: {} \n".format(netPotential + netKE))
            conserved.write("------------------------------------ \n")

    text_file.close()
    energyFile.close()
    conserved.close()


def thermostat(atomList, targetTemp):
    instantTemp = 0
    for atom in atomList:
        # Dot product of velocity vectors
        dot = np.dot(atom.velocities, atom.velocities)
        instantTemp += MASS * dot
    instantTemp = instantTemp
    instantTemp /= (3 * len(atomList) - 3)
    instantTemp = np.sqrt(targetTemp / instantTemp)

    for atom in atomList:  # V = lambda * V
        atom.velocities *= instantTemp


def calcForces(atomList, energyFile):
    netPotential = 0
    for atom in atomList:  # Set all accelerations to zero
        atom.accelerations = [0., 0., 0.]

    # Iterate over all atoms in atomList
    for i in range(len(atomList) - 1):
        for j in range(i + 1, len(atomList)):
            distArr = np.zeros(3)  # Position of an atom relative to another atom
            distArr = atomList[i].positions - atomList[j].positions
            # Boundary conditions - Interact through walls if closer
            distArr = distArr - L * np.round(distArr / L)
            for k in range(3):
                while distArr[k] >= .5 * L:
                    distArr[k] -= L
                while distArr[k] < -.5 * L:
                   distArr[k] += L

            dot = np.dot(distArr, distArr)
            r = np.sqrt(dot)  # Distance vector magnitude

            if r < rCutoff:

                force = 24 * EPS_STAR / dot * ((2 * (SIGMA / r) ** 12) -
                                                           (SIGMA / r) ** 6)

                netPotential += 4 * EPS_STAR / dot * ((SIGMA / r) ** 12 - (
                                                            SIGMA / r) ** 6)
                energyFile.write("{} on {}: {} \n".format(i, j, force))
                # energyFile.write("----------------- \n")
                atomList[i].accelerations += (force * distArr / MASS)
                atomList[j].accelerations -= (force * distArr / MASS)

    return netPotential




class Atom:
    def __init__(self, positionX, positionY, positionZ):
        self.positions = np.array([positionX, positionY, positionZ])
        self.velocities = np.random.normal(size=3)
        # self.velocities = np.zeros(3)
        self.accelerations = np.zeros(3)
        self.oldAccelerations = np.zeros(3)


if __name__ == "__main__":
    main()
