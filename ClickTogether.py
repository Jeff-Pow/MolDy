import math
import random
import numpy as np


SIGMA = 3.405 # Angstroms
EPSILON = 119.8 # Kelvin
n = 4
rhostar = .6
rho = rhostar/(SIGMA**3) # density
L = (n**3 / rho)**(1/3) # unit cell length
MASS = 39.9 # AMU
TARGET_TEMP = .1 # Kelvin



def main():
    atomList = []
    dx = SIGMA # Distance between atoms in one dimension
    for i in range(n):
        for j in range(n):
            for k in range(n):
                atom = Atom(i * dx, j * dx, k * dx)
                atomList.append(atom)



    numTimeSteps = 1000
    timeStep = .01 # float(finalTime - time) / numTimeSteps

    text_file = open("Interactions.xyz", "w")


    for i in range(numTimeSteps):
        print(i)
        text_file.write("%i \n \n" % len(atomList))

        for j in range(len(atomList)): #Print locations of all molecules
            text_file.write("A %f %f %f \n" % (atomList[j].positions[0],
                                             atomList[j].positions[1],
                                             atomList[j].positions[2]))

        for j in range(len(atomList)): # Find new position
            for k in range(3):
                # 3
                atomList[j].positions[k] += atomList[j].velocities[k] * \
                                            timeStep + .5 *\
                                     atomList[j].accelerations[k] * timeStep**2

                atomList[j].oldAccelerations[k] = atomList[j].accelerations[k]

        netPotential = calcForces(atomList) / n ** 3 # Update acclerations

        netVelocity = 0

        if i % 5 == 0 and 500 > i > 0:
            gaussianVelocities(atomList, TARGET_TEMP)
        else:
            for j in range(len(atomList)): # Update velocities
                for k in range(3):
                    atomList[j].velocities[k] += .5 * (atomList[j].accelerations \
                                [k] + atomList[j].oldAccelerations[k]) * timeStep
                    netVelocity += atomList[j].velocities[k]


        




    text_file.close()


def calcEnergy(mass, velocity, omega, position):
    return .5 * mass * velocity * velocity + .5 * mass * omega * omega * \
           position * position

def gaussianVelocities(atomList, targetTemp):

    instantTemp = 0
    for i in range(len(atomList)):
        dot = 0
        for k in range(3):
            dot += atomList[i].velocities[k] * atomList[i].velocities[k]
        instantTemp += atomList[0].mass + dot
    instantTemp /= 3 * len(atomList) - 3

    for i in range(len(atomList)):
        for j in range(3):
            atomList[i].velocities[j] *= np.sqrt(targetTemp / instantTemp)


def calcForces(atomList):
    netPotential = 0
    for i in range(len(atomList)): # Set all accelerations to zero
        atomList[i].accelerations = [0, 0, 0]

    # Iterate over all atoms in atomList
    for i in range(len(atomList)):
        for j in range(len(atomList)):
            if i != j:
                dx = [0, 0, 0] # Position of an atom relative to another atom
                for k in range(3):
                    dx[k] = atomList[i].positions[k] - atomList[j].positions[k]


                dot = 0 # Dot product of distance between two atoms
                for k in range (3):
                    dot += dx[k] * dx[k]

                r = math.sqrt(dot) # Vector magnitude
                for k in range(3): # Find unit vector
                   dx[k] /= r

                netPotential += 4 * EPSILON * ((SIGMA / dot) ** 12 - (SIGMA
                                                                    / dot) ** 6)
                force = 24 * EPSILON / r**2 * (2 * (SIGMA / dot) ** 12 - (SIGMA
                                                            / dot) ** 6)
                for k in range (3):
                    atomList[i].accelerations[k] += force * dx[k] / atomList[
                    i].mass


    return netPotential



class Atom:
    def __init__(self, positionX, positionY, positionZ):
        self.positions = [positionX, positionY, positionZ]
        self.velocities = np.random.normal(size=3)
        self.accelerations = [0, 0, 0]
        self.oldAccelerations = [0, 0, 0]
        self.mass = .5


if __name__ == "__main__":
    main()
