import math
import numpy as np
import random

kB = 1.38064852 * 10 ** -23
Na = 6.022 * 10 ** -23

numTimeSteps = 10000 # Parameters to change for simulation
n = 2
timeStep = .01

N = n ** 3
SIGMA = 3.405 # Angstroms
EPSILON = 1.6540 * 10 ** -21 # Kelvin
EPS = EPSILON / (1.38 * 10 ** -23)


rhostar = .6 # Dimensionless density of gas


rho = rhostar/(SIGMA**3) # density
L = ((N / rho)**(1/3)) # unit cell length
rCutoff = L / 2
TARGET_TEMP = 1.24 * EPS
cutoff = 2.5 * SIGMA # Kelvin
# MASS = 39.9 * 10 / 6.022169 / 1.380662
MASS = 48.07
timeStep *= ((MASS * SIGMA ** 2) / EPSILON) ** .5



def main():
    atomList = []
    dx = SIGMA # Distance between atoms in one dimension
    #for i in range(n):
    #   for j in range(n):
    #       for k in range(n):
    #           atom = Atom(i * dx, j * dx, k * dx)
    #           atomList.append(atom)
    atomList.append(Atom(0, 0, 0))

    atomList.append(Atom(dx, 0, 0))
    timeStep = timeStep
    timeStep *= (EPSILON / (MASS) * SIGMA ** 2) ** .5

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
                atomList[j].positions[k] += atomList[j].velocities[k] * \
                                            timeStep + .5 * \
                                     atomList[j].accelerations[k] * timeStep**2

                atomList[j].oldAccelerations[k] = atomList[j].accelerations[k]

        for j in range(len(atomList)): # Boundaries of simulation
           for k in range(3):
                if atomList[j].positions[k] > L:
                    atomList[j].positions[k] -= L

                if atomList[j].positions[k] < 0:
                    atomList[j].positions[k] += L



        calcForces(atomList, energyFile) # Update accelerations

        netVelocity = np.zeros(3)


        if i < numTimeSteps / 2 and i != 0 and i % 50 == 0:
            gaussianVelocities(atomList, TARGET_TEMP)
        for j in range(len(atomList)): # Update velocities
            for k in range(3):
                atomList[j].velocities[k] += (.5 * (atomList[j].accelerations
                                [k] + atomList[j].oldAccelerations[k])) * \
                    timeStep
                netVelocity[k] += atomList[j].velocities[k]



        #if i > numTimeSteps / 2:
            #energyFile.write("Time: {} \n".format(i))

            #v = np.dot(netVelocity, netVelocity)

            #netKE = (.5 * MASS * v)
            #energyFile.write("KE: {} \n".format(netKE))

            #energyFile.write("PE: {} \n".format(netPotential))
            #energyFile.write("Total energy: {} \n".format(netPotential +
            # netKE))
        energyFile.write("------------------------------------ \n")




    text_file.close()
    energyFile.close()


def gaussianVelocities(atomList, targetTemp):

    instantTemp = 0
    for i in range(len(atomList)):
        dot = 0 # Dot product of velocity vectors of each atom
        for j in range(3):
            dot += atomList[i].velocities[j] * atomList[i].velocities[j]
        instantTemp += MASS * dot

    instantTemp /= (3 * len(atomList) - 3)

    for i in range(len(atomList)): # V = lambda * V
        for j in range(3):
            atomList[i].velocities[j] *= np.sqrt(targetTemp / instantTemp)


def calcForces(atomList, energyFile):
    netPotential = 0
    for i in range(len(atomList)): # Set all accelerations to zero
        atomList[i].accelerations = [0, 0, 0]

    # Iterate over all atoms in atomList
    for i in range(len(atomList)):
        for j in range(len(atomList)):
            if i != j:
                dx = [0, 0, 0] # Position of an atom relative to another atom
                dot = 0
                for k in range(3):
                    dx[k] = atomList[i].positions[k] - atomList[j].positions[k]
                    dx[k] = dx[k] - L * np.round(dx[k] / L)
                    #while dx[k] >= .5 * L:
                        # If atom is closer to another specific atom via the
                            #border rather than through the middle of the
                        # plane, calculate force through the border
                        #dx[k] -= L

                    #while dx[k] < -.5 * L:
                        #dx[k] += L

                    dot += dx[k] * dx[k] # Dot product






                r = np.sqrt(dot) # Vector magnitude
                for k in range(3): # Find unit vector
                   dx[k] = dx[k] / r

                netPotential += 4 * EPS * ((SIGMA / r) ** 12 - (SIGMA
                                                                    / r) ** 6)
                #if r < .25 * L * L:
                force = 24 * EPS / dot * (2 * (SIGMA / dot) ** 12 - (
                 SIGMA
                                                             / dot) ** 6)

                energyFile.write("{} on {}: {} \n".format(i, j, force))
                # energyFile.write("----------------- \n")
                for k in range (3):
                    atomList[i].accelerations[k] += (force * dx[k] /
                                                     MASS)


    return netPotential



class Atom:
    def __init__(self, positionX, positionY, positionZ):
        self.positions = [positionX, positionY, positionZ]
        self.velocities = np.random.normal(size = 3)
        #self.velocities = [0, 0, 0]
        self.accelerations = [0, 0, 0]
        self.oldAccelerations = [0, 0, 0]


if __name__ == "__main__":
    main()
