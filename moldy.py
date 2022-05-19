import time

import matplotlib.pyplot as plt
import numpy as np

kB = 1.38064852e-23 # J / K
Na = 6.022e23 # Atoms per mol

numTimeSteps = 10000  # Parameters to change for simulation
n = 2
timeStep = .001 # dt_star

N = n ** 3
SIGMA = 3.405  # Angstroms
EPSILON = 1.6540e-21  # Joules
EPS_STAR = EPSILON / kB # ~119.8 Kelvin

rhostar = .7  # Dimensionless density of gas
rho = rhostar / (SIGMA ** 3)  # density
L = ((N / rho) ** (1 / 3))  # unit cell length
rCutoff = SIGMA * 2.5
TARGET_TEMP = 1.1 * EPS_STAR
#
MASS = 39.9 * 10 / Na / kB # K * ps^2 / A^2
timeStep *= np.sqrt((MASS * SIGMA ** 2) / EPS_STAR) # Picoseconds



def main():
    text_file = open("out.xyz", "w")
    energyFile = open("Energy.txt", "w")
    conserved = open("Conservation.txt", "w")
    plt.style.use('classic')

    KE = []
    PE = []
    netE = []

    atomList = []
    # Sigma is the distance between atoms in one dimension
    for i in range(n):
        for j in range(n):
            for k in range(n):
                atom = Atom(i * SIGMA, j * SIGMA, k * SIGMA)
                atomList.append(atom)

    conserved.write("Temp Scalar: {} \n \n".format(thermostat(atomList, TARGET_TEMP)))


    count = .05
    currTime = time.time()
    for i in range(numTimeSteps):
        # if i == count * numTimeSteps:
        #     print("{}% \n".format(i / numTimeSteps * 100))
        #     count += .05
        #     count = round(count, 3)
        print(i)


        text_file.write("%i \n \n" % N)
        energyFile.write("Time: {} \n".format(i))

        for atom in atomList:  # Print locations of all molecules

            energyFile.write("Positions: %f %f %f\n" % tuple(atom.positions))
            energyFile.write("Velocities: %f %f %f\n" % tuple(atom.velocities))
            energyFile.write("Accelerations: %f %f %f\n - \n" % tuple(atom.accelerations))
            text_file.write("A %f %f %f\n" % tuple(atom.positions))


        totalVelSquared = 0

        for atom in atomList: # Find new position
            atom.positions += atom.velocities * timeStep + 0.5 * atom.accelerations * timeStep * timeStep
            atom.positions += -L * np.floor(atom.positions / L) # Keep atom inside box
            atom.oldAccelerations = atom.accelerations.copy()

        netPotential = calcForces(atomList, energyFile) # Update accelerations

        for atom in atomList: # Update velocities
            atom.velocities += (.5 * (atom.accelerations + atom.oldAccelerations)) * timeStep
            totalVelSquared += np.dot(atom.velocities, atom.velocities)

        if i < numTimeSteps / 2. and i != 0. and i % 5 == 0:
            thermostat(atomList, TARGET_TEMP)

        if i > numTimeSteps / 2:
            conserved.write("Time: {} \n".format(i))

            netKE = .5 * MASS * totalVelSquared

            conserved.write("KE: {} \n".format(netKE))
            KE.append(netKE)
            conserved.write("PE: {} \n".format(netPotential))
            PE.append(netPotential)
            conserved.write("Total energy: {} \n".format(netPotential + netKE))
            netE.append(netPotential + netKE)
            conserved.write("------------------------------------ \n")

    T = (time.time() - currTime) 
    print("Time elapsed: \n {} sec \n".format(T))
    print("{} mins".format(T / 60))
    plt.plot(range(5000, 9999), KE, label = "KE")
    plt.plot(range(5000, 9999), PE, label = "PE")
    plt.plot(range(5000, 9999), netE, label = "Net Energy")
    text_file.close()
    energyFile.close()
    conserved.close()
    plt.show()


def thermostat(atomList, targetTemp):
    instantTemp = 0
    for atom in atomList:
        instantTemp += MASS * np.dot(atom.velocities, atom.velocities)

    instantTemp /= (3 * N - 3)
    tempScalar = np.sqrt(targetTemp / instantTemp)

    for atom in atomList:  # V = lambda * V
        atom.velocities *= tempScalar
    return tempScalar


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

            dot = np.dot(distArr, distArr)
            r = np.sqrt(dot)  # Distance vector magnitude

            if r <= rCutoff:
                sor = SIGMA / r # sor = sigma over r

                force_over_r = 24 * EPS_STAR / dot * ((2 * sor ** 12 - sor ** 6))
                netPotential += 4 * EPS_STAR * (sor ** 12 - sor ** 6)

                energyFile.write("{} on {}: {} \n".format(i, j, force_over_r * r))
                # energyFile.write("----------------- \n")
                atomList[i].accelerations += (force_over_r * distArr / MASS)
                atomList[j].accelerations -= (force_over_r * distArr / MASS)

    return netPotential




class Atom:
    def __init__(self, positionX, positionY, positionZ):
        self.positions = np.array([positionX, positionY, positionZ])
        self.velocities = np.random.normal(size=3)
        self.accelerations = np.zeros(3)
        self.oldAccelerations = np.zeros(3)


if __name__ == "__main__":
    main()
