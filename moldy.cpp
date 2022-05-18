//
// Created by jeff on 5/18/2022.
//

#include "moldy.h"
#include <iostream>
#include <math.h>
#include <ctime>
#include <random>

const double Kb = 1.38064582 * math.pow(10, -23); // J / K
const double Na = 6.022 * math.pow(10, 23); // Atoms per mole

const int numTimeSteps = 10000; // Parameters to change for simulation
const int n = 6;
const double dt_star= .001;

const int N = math.pow(n, 3);
const double SIGMA = 3.405; // Angstroms
const double EPSILON = 1.6540 * math.pow(10, -21); // Joules
const double EPS_STAR = EPSILON / Kb; // ~ 119.8 K

const double rhostar = .7; // Dimensionless density of gas
const double rho = rhostar / math.pow(SIGMA, 3); // Density of gas
const double L = math.pow((N / rho), 3); // Unit cell length
const double rCutoff = SIGMA * 2.5;
const double TARGET_TEMP = 1.1 * EPS_STAR;
// 39.9 is amu mass of argon, 10 is a conversion between the missing units :)
const double MASS = 39.9 * 10 / Na / Kb; // K * ps^2 / A^2
const int timeStep *= math.sqrt(MASS * SIGMA * SIGMA / EPS_STAR); // Convert time step to picoseconds

double dot(double x, double y, double z);
void thermostat(std::vector<Atom> atomList, double targetTemp);
double calcForces(std::vector<Atom> atomList, std::ofstream &energyFile) {

int main() {
    ofstream text_file;
    ofstream energyFile;
    ofstream conserved;
    text_file.open("out.xyz");
    energyFile.open("Energy.txt");
    conserved.open("Conservation.txt");

    std::vector<double> KE;
    std::vector<double> PE;
    std::vector<double> netE;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<Atom> atomList;
    // Sigma is the distance between atoms at resting state
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                atomList.push_back(Atom(i * SIGMA, j * SIGMA, k * SIGMA));
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j) {
            atomList[i].velocities[j] = distribution(generator);
        }
    }
    thermostat(atomList, TARGET_TEMP);


    double totalVelSquared;
    double netPotential;
    clock_t begin = clock();
    for (int i = 0; i < numTimeSteps; ++i) {
        std::cout << i << "\n";

        text_file << N << "\n \n";
        energyFile << "Time: " << i << "\n";

        for (Atom atom : atomList) {
                energyFile << "Positions: " << atom.positions[0] << " " << atom.positions[1] << " " << atom.positions[2] << "\n";
                energyFile << "Velocities: " << atom.velocities[0] << " " << atom.velocities[1] << " " << atom.velocities[2] << "\n";
                energyFile << "Accelerations: " << atom.accelerations[0] << " " << atom.accelerations[1] << " " << atom.accelerations[2] << "\n";
                text_file << "A " << atom.positions[0] << " " << atom.positions[1] << " " << atom.positions[2] << "\n";
        }

        for (Atom atom : atomList) { // Update positions
            for (int j = 0; j < 3; ++j) {
               atom.positions[j] += atom.velocities[j] * timeStep + .5 * atom.accelerations[j] * timeStep * timeStep;
               atom.positions[j] += -L * math.floor(atom.positions[j] / L); // Keep atom inside box
               atom.oldAccelerations[j] = atom.accelerations[j];
            }
        }

        netPotential = calcForces(atomList, energyFile); // Update accelerations and return potential of system

        totalVelSquared = 0;
        for (Atom atom : atomList) { // Update velocities
            for (int j = 0; j < 3; ++j) {
                atom.velocities[j] += .5 * (atom.accelerations[j] + atom.oldAccelerations[j]) * timeStep;
                totalVelSquared += atom.velocities[j] * atom.velocities[j];
            }
        }

        if (i < numTimeSteps / 2 && i != 0 && i % 5 == 0) {
            thermostat(atomList, TARGET_TEMP);
        }

        // TODO: Conservation of energy file
    }





    clock_t end = clock();
    double time = double(end - begin) / (double)CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time << std::endl;

    text_file.close();
    energyFile.close();
    conserved.close();

    return 0;
}

double dot (double x, double y, double z) {
    return x * x + y * y + z * z;
}

void thermostat(std::vector<Atom> atomList, double targetTemp) {
    double instantTemp = 0;
    for (Atom atom : atomList) {
        instantTemp += MASS * dot(atom.velocities[0], atom.velocities[1], atom.velocities[2]);
    }

    instantTemp /= (3 * N - 3);
    double tempScalar = math.sqrt(targetTemp / instantTemp);

    for (Atom atom : atomList) {
        for (int j = 0; j < 3; ++j) {
            atom.velocities[j] *= tempScalar;
        }
    }
}

double calcForces(std::vector<Atom> atomList, std::ofstream &energyFile) {
    double netPotential = 0;
    for (Atom atom : atomList) {
        atom.accelerations = {0, 0, 0};
    }

    // Iterate over all pairs of atoms once in atom list
    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double distArr[3];
            for (int k = 0; k < 3; ++k) {
                distArr[k] = atomList[i].positions[k] - atomList[j].positions[k];
                distArr[k] -= L * math.round(distArr[k] / L); // Boundary conditions
            }

            double r2 = dot(distArr[0], distArr[1], distArr[2]);
            double r = math.sqrt(r2);

            if (r <= rCutoff) {
                double sor = SIGMA / r;
                double sor6 = math.pow(sor, 6);
                double sor12 = r6 * r6;

                double forceOverR = 24 * EPS_STAR / r2 * (2 * sor12 - sor6);
                netPotential += 4 * EPS_STAR * (sor12 - sor6);

                energyFile << i << " on " << j << ": " << forceOverR * r << "\n";

                for (int k = 0; k < 3; ++k) {
                    atomList[i].accelerations[k] += forceOverR * distArr[k] / MASS;
                    atomList[j].accelerations[k] -= forceOverR * distArr[k] / MASS;
                }
            }

        }
    }
    return netPotential;
}

class Atom(x, y, z) {
    public:
        double position[3] = {x, y, z};
        double velocities[3];
        double accelerations[3];
        double oldAccelerations[3];
}