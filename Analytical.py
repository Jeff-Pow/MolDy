import math


def main():
    omega = int(input("Enter omega: "))
    initPosition = int(input("Enter initial position: "))
    initVelocity = int(input("Enter initial velocity: "))
    initTime = int(input("Enter initial time: "))
    finalTime = int(input("Enter final time: "))
    numTimeSteps = int(input("Input number of time steps to evaluate: "))
    timeStep = float((finalTime - initTime) / numTimeSteps)
    mass = int(input("Enter mass: "))
    time = initTime

    text_file = open("output/AnalyticalOutput.txt", "w")
    text_file.write("Initial energy: %f \n \n" % calcEnergy(mass, initVelocity, omega, initPosition))
    for i in range(0, numTimeSteps):
        text_file.write("Time: %f \n" % time)
        text_file.write("Position: %f \n" % calcPosition(initPosition, omega, time, initVelocity))
        text_file.write("Velocity: %f \n" % calcVelocity(initPosition, omega, time, initVelocity))
        text_file.write("Acceleration: %f \n" % calcAcceleration(initPosition, omega, time, initVelocity))
        time += timeStep
        text_file.write("------------------------------------------------------------ \n")

    text_file.write("Final energy: %f \n" % calcEnergy(mass, calcVelocity(initPosition, omega, time, initVelocity),
                                                       omega, calcPosition(initPosition, omega, time, initVelocity)))

    text_file.close()


def calcPosition(initPosition, omega, time, initVelocity):
    return initPosition * math.cos(omega * time) + initVelocity / omega * math.sin(omega * time)


def calcVelocity(initPosition, omega, time, initVelocity):
    return -1 * initPosition * omega * math.sin(omega * time) + initVelocity * math.cos(omega * time)


def calcAcceleration(initPosition, omega, time, initVelocity):
    return -1 * initPosition * omega * omega * math.cos(omega * time) - initVelocity * omega * math.cos(omega * time)


def calcEnergy(mass, velocity, omega, position):
    return .5 * mass * velocity * velocity + .5 * mass * omega * omega * position * position


if __name__ == "__main__":
    main()
