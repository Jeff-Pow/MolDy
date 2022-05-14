def main():
    omega = .1

    positionX = .55
    positionY = .34
    positionZ = -.66
    velocityX = .37
    velocityY = .46
    velocityZ = .89

    initTime = 0
    finalTime = 1000000
    numTimeSteps = 1000000
    timeStep = 1

    time = initTime

    
    accelerationX = -1 * omega * omega * positionX
    accelerationY = -1 * omega * omega * positionY
    accelerationZ = -1 * omega * omega * positionZ


    text_file = open("output/Verlet3DOutput.xyz", "w")

    for i in range(0, numTimeSteps):
        text_file.write("1 \n \n")
        text_file.write("H %f %f %f" % (positionX, positionY, positionZ))

        # 3
        positionX += velocityX * timeStep + .5 * timeStep * timeStep * \
                     accelerationX
        positionY += velocityY * timeStep + .5 * timeStep * timeStep * \
                     accelerationY
        positionZ += velocityZ * timeStep + .5 * timeStep * timeStep * \
                     accelerationZ

        # 4
        velocityX += .5 * timeStep * accelerationX
        velocityY += .5 * timeStep * accelerationY
        velocityZ += .5 * timeStep * accelerationZ

        # 5
        accelerationX = -1 * omega * omega * positionX
        accelerationY = -1 * omega * omega * positionY
        accelerationZ = -1 * omega * omega * positionZ

        # 6
        velocityX += .5 * timeStep * accelerationX
        velocityY += .5 * timeStep * accelerationY
        velocityZ += .5 * timeStep * accelerationZ

        time += timeStep
        text_file.write("\n")


    text_file.close()


def calcEnergy(mass, velocity, omega, position):
    return .5 * mass * velocity * velocity + .5 * mass * omega * omega * \
           position * position
        
        
if __name__ == "__main__":
    main()
