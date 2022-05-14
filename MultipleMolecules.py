def main():
    molecule1 = Molecule(.1, .55, .34, -.66, .37, .46, .89, .03)
    molecule2 = Molecule(.15, .34, -.72, -.78, .08, -.2, .77, .05)
    molecule3 = Molecule(.26, .86, -.47, .86, -.42, .38, .66, .1)
    moleculeList = [molecule1, molecule2, molecule3]

    time = 0
    finalTime = 500
    numTimeSteps = 1000
    timeStep = (finalTime - time) / numTimeSteps

    text_file = open("output/MultipleMolecules.xyz", "w")

    for i in range(0, numTimeSteps):

        text_file.write("%i \n \n" % len(moleculeList))

        for j in range(len(moleculeList)):
            text_file.write("A %f %f %f \n" % (moleculeList[j].positionX,
                                             moleculeList[j].positionY,
                                             moleculeList[j].positionZ))


            # 3
            moleculeList[j].positionX += moleculeList[j].velocityX * timeStep\
                                         + .5 * timeStep\
                                         * timeStep * \
                         moleculeList[j].accelerationX
            moleculeList[j].positionY += moleculeList[j].velocityY * timeStep\
                                         + .5 * timeStep * timeStep * \
                         moleculeList[j].accelerationY
            moleculeList[j].positionZ += moleculeList[j].velocityZ * timeStep + .5 * timeStep * timeStep * \
                         moleculeList[j].accelerationZ

            # 4
            moleculeList[j].velocityX += .5 * timeStep * moleculeList[j].accelerationX
            moleculeList[j].velocityY += .5 * timeStep * moleculeList[j].accelerationY
            moleculeList[j].velocityZ += .5 * timeStep * moleculeList[j].accelerationZ

            # 5
            moleculeList[j].accelerationX = -1 * moleculeList[j].omega * moleculeList[j].omega * moleculeList[j].positionX
            moleculeList[j].accelerationY = -1 * moleculeList[j].omega * moleculeList[j].omega * moleculeList[j].positionY
            moleculeList[j].accelerationZ = -1 * moleculeList[j].omega * moleculeList[j].omega * moleculeList[j].positionZ

        # 6
            moleculeList[j].velocityX += .5 * timeStep * moleculeList[j].accelerationX
            moleculeList[j].velocityY += .5 * timeStep * moleculeList[j].accelerationY
            moleculeList[j].velocityZ += .5 * timeStep * moleculeList[j].accelerationZ

        time += timeStep

    text_file.close()


def calcEnergy(mass, velocity, omega, position):
    return .5 * mass * velocity * velocity + .5 * mass * omega * omega * \
           position * position


class Molecule:
    def __init__(self, omega, positionX, positionY, positionZ, velocityX,
                 velocityY, velocityZ, mass):
        self.omega = omega
        self.mass = mass
        self.positionX = positionX
        self.positionY = positionY
        self.positionZ = positionZ
        self.velocityX = velocityX
        self.velocityY = velocityY
        self.velocityZ = velocityZ
        self.accelerationX = omega * omega * positionX
        self.accelerationY = omega * omega * positionY
        self.accelerationZ = omega * omega * positionZ


if __name__ == "__main__":
    main()
