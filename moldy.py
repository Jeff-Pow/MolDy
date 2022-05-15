import math
import numpy as np
import random

kB = 1.38064852 * 10 ** -23
Na = 6.022 * 10 ** -23

numTimeSteps = 1000  # Parameters to change for simulation
n = 4
timeStep = .01

# N = n ** 3
N = 2
SIGMA = 3.405  # Angstroms
EPSILON = 1.6540 * 10 ** -21  # Kelvin
EPS = EPSILON / (1.38 * 10 ** -23)

rhostar = .6  # Dimensionless density of gas

rho = rhostar / (SIGMA ** 3)  # density
L = ((N / rho) ** (1 / 3))  # unit cell length
rCutoff = L / 2
TARGET_TEMP = 1.24 * EPS
cutoff = 2.5 * SIGMA  # Kelvin
# MASS = 39.9 * 10 / 6.022169 / 1.380662
MASS = 48.07
#timeStep *= (EPSILON / (MASS * SIGMA ** 2)) ** .5


def main():
    atomList = []
    # Sigma is the distance between atoms in one dimension
    #for i in range(n):
    #    for j in range(n):
    #        for k in range(n):
    #            atom = Atom(i * SIGMA, j * SIGMA, k * SIGMA)
    #            atomList.append(atom)
    atomList.append(Atom(0, 0, 0))

    atomList.append(Atom(SIGMA, 0, 0))

    text_file = open("out.xyz", "w")
    energyFile = open("Energy.txt", "w")

    for i in range(numTimeSteps):
        print(i)
        text_file.write("%i \n \n" % len(atomList))
        energyFile.write("Time: {} \n".format(i))

        for j in range(len(atomList)):  # Print locations of all molecules

            energyFile.write("Positions: %f %f %f \n" % (atomList[
                                                             j].positions[0],
                                                         atomList[j].positions[
                                                             1],
                                                         atomList[j].positions[
                                                             2]))
            energyFile.write("Velocities: %f %f %f \n" % (atomList[
                                                              j].velocities[0],
                                                          atomList[
                                                              j].velocities[
                                                              1],
                                                          atomList[
                                                              j].velocities[
                                                              2]))
            energyFile.write("Accelerations: %f %f %f \n - \n" % (atomList[
                                                                      j].accelerations[
                                                                      0],
                                                                  atomList[
                                                                      j].accelerations[
                                                                      1],
                                                                  atomList[
                                                                      j].accelerations[
                                                                      2]))
            text_file.write(("A %f %f %f \n" % (atomList[j].positions[0],
                                                atomList[j].positions[1],
                                                atomList[j].positions[2])))

        for j in range(len(atomList)): # Find new position
            for k in range(3):
                atomList[j].positions[k] += atomList[j].velocities[k] * timeStep + .5 * atomList[j].accelerations[k] * timeStep ** 2
                atomList[j].oldAccelerations[k] = atomList[j].accelerations[k]


        #for j in range(len(atomList)):  # Boundaries of simulation
        #    for k in range(3):
        #        if atomList[j].positions[k] > L:
        #            atomList[j].positions[k] -= L
#
#                if atomList[j].positions[k] < 0:
#                    atomList[j].positions[k] += L

        calcForces(atomList, energyFile)  # Update accelerations

        netVelocity = np.zeros(3)

        #if i < numTimeSteps / 2 and i != 0 and i % 5 == 0:
        #    gaussianVelocities(atomList, TARGET_TEMP)

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
        instantTemp += MASS * dot

    instantTemp /= (3 * len(atomList) - 3)
    instantTemp = np.sqrt(instantTemp)

    for i in range(len(atomList)):  # V = lambda * V
        atomList[i].velocities *= instantTemp


def calcForces(atomList, energyFile):
    netPotential = 0
    for i in range(len(atomList)):  # Set all accelerations to zero
        atomList[i].accelerations = [0, 0, 0]

    # Iterate over all atoms in atomList
    for i in range(len(atomList)):
        for j in range(len(atomList)):
            if i != j:
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
                distArr /= r  # Find unit direction of force

                netPotential += 4 * EPS * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)

                force = 24 * EPS / dot * (2 * (SIGMA / dot) ** 12 - (SIGMA / dot) ** 6)

                energyFile.write("{} on {}: {} \n".format(i, j, force))
                # energyFile.write("----------------- \n")
                #for k in range(3):
                atomList[i].accelerations += (force * distArr / MASS)

    return netPotential


class Atom:
    def __init__(self, positionX, positionY, positionZ):
        self.positions = np.array([positionX, positionY, positionZ])
        #self.velocities = np.random.normal(size=3)
        self.velocities = np.zeros(3)
        self.accelerations = np.zeros(3)
        self.oldAccelerations = np.zeros(3)


if __name__ == "__main__":
    main()
