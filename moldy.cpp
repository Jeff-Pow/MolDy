//
// Created by jeff on 5/18/2022.
//

#include <iostream>
#include <math.h>
#include <time.h>
#include <random>
#include <fstream>

class Atom {
public:
    double positions[3];
    double velocities[3];
    double accelerations[3];
    double oldAccelerations[3];

    Atom(double x, double y, double z) {
        positions[0] = x;
        positions[1] = y;
        positions[2] = z;
    }
};

const double Kb = 1.38064582 * std::pow(10, -23); // J / K
const double Na = 6.022 * std::pow(10, 23); // Atoms per mole

const int numTimeSteps = 10000; // Parameters to change for simulation
const double dt_star= .001;

const int N = 500; // Number of atoms in simulation
const double SIGMA = 3.405; // Angstroms
const double EPSILON = 1.6540 * std::pow(10, -21); // Joules
const double EPS_STAR = EPSILON / Kb; // ~ 119.8 K

const double rhostar = .6; // Dimensionless density of gas
const double rho = rhostar / std::pow(SIGMA, 3); // Density of gas
const double L = std::cbrt(N / rho); // Unit cell length
const double rCutoff = SIGMA * 2.5; // Forces are negligible past this distance, not worth calculating
const double tStar = 1.24; // Reduced units of temperature
const double TARGET_TEMP = tStar * EPS_STAR;
// 39.9 is mass of argon in amu, 10 is a conversion between the missing units :)
const double MASS = 39.9 * 10 / Na / Kb; // K * ps^2 / A^2
const double timeStep = dt_star * std::sqrt(MASS * SIGMA * SIGMA / EPS_STAR); // Convert time step to picoseconds

double dot(double x, double y, double z);
void thermostat(std::vector<Atom> &atomList);
double calcForces(std::vector<Atom> &atomList);
std::vector<Atom> faceCenteredCell();
std::vector<Atom> simpleCubicCell();
double radialDistribution(std::ifstream &xyzFile);

int main() {
    std::ofstream positionFile;
    std::ofstream debug;
    std::ofstream energyFile;
    positionFile.open("out.xyz");
    debug.open("debug.dat");
    energyFile.open("Energy.dat");

    std::vector<double> KE;
    std::vector<double> PE;
    std::vector<double> netE;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);


    std::vector<Atom> atomList = faceCenteredCell();
    

    for (int i = 0; i < N; ++i) { // Randomize velocities
         for (int j = 0; j < 3; ++j) {
             atomList[i].velocities[j] = distribution(generator);
         }
    }
   
    thermostat(atomList); // Make velocities more accurate


    double totalVelSquared;
    double netPotential;
    clock_t begin = clock();

    std::cout << "Starting program \n";
    double count = .05;
    for (int i = 0; i < numTimeSteps; ++i) { // Main loop handles integration and printing to files
        if (i > count * numTimeSteps) {
            std::cout << count * 100 << " % \n";
            count += .05;
        }
        
        positionFile << N << "\n \n";
        // debug << "Time: " << i << "\n";


        for (int j = 0; j < N; ++j) { // Write positions to xyz file
            positionFile << "A " << atomList[j].positions[0] << " " << atomList[j].positions[1] << " " << atomList[j].positions[2] << "\n";

            // debug << "Positions: " << atomList[j].positions[0] << " " << atomList[j].positions[1] << " " << atomList[j].positions[2] << "\n";
            // debug << "Velocities: " << atomList[j].velocities[0] << " " << atomList[j].velocities[1]  << " " << atomList[j].velocities[2] << "\n";
            // debug << "Accelerations: " << atomList[j].accelerations[0] << " " << atomList[j].accelerations[1]  << " " << atomList[j].accelerations[2] << "\n";
            // debug << "- \n";
        }
        // debug << "------------------------------------------------- \n";

        for (int k = 0; k < N; ++k) { // Update positions
            for (int j = 0; j < 3; ++j) {
                atomList[k].positions[j] += atomList[k].velocities[j] * timeStep 
                        + .5 * atomList[k].accelerations[j] * timeStep * timeStep;
                atomList[k].positions[j] += -L * std::floor(atomList[k].positions[j] / L); // Keep atom inside box
                atomList[k].oldAccelerations[j] = atomList[k].accelerations[j];
            }
        }
        

        netPotential = calcForces(atomList); // Update accelerations and return potential of system

        totalVelSquared = 0;
        for (int k = 0; k < N; ++k) { // Update velocities
            for (int j = 0; j < 3; ++j) {
                atomList[k].velocities[j] += .5 * (atomList[k].accelerations[j] + atomList[k].oldAccelerations[j]) * timeStep;
                totalVelSquared += atomList[k].velocities[j] * atomList[k].velocities[j];
            }
        }

        if (i < numTimeSteps / 2 && i != 0 && i % 5 == 0) { // Apply velocity modifications for first half of sample
            thermostat(atomList);
        }

        if (i > numTimeSteps / 2) { // Record energies to arrays and file
            energyFile << "Time: " << i << "\n";
            
            double netKE = .5 * MASS * totalVelSquared;

            //energyFile << "KE: " << netKE << "\n";
            KE.push_back(netKE);
            //energyFile << "PE: " << netPotential << "\n";
            PE.push_back(netPotential);
            //energyFile << "Total energy: " << netPotential + netKE << "\n";
            //energyFile << "------------------------------------------ \n";
            netE.push_back(netPotential + netKE);
        }
    }

    double avgPE = 0; // Average PE is average of PE array
    for (int i = 0; i < PE.size(); i++) {
        avgPE += PE[i];
    }
    avgPE /= PE.size();

    double SoLo2 = SIGMA / (L / 2); // Sigma over L over 2
    double Ulrc = (8.0 / 3.0) * M_PI * N * rhostar * EPS_STAR; // Long range correction to potential formula
    double temp = 1.0 / 3.0 * std::pow(SoLo2, 9.0);
    double temp1 = std::pow(SoLo2, 3.0);
    Ulrc *= (temp - temp1);
    double PEstar = ((avgPE + Ulrc) / N) / EPS_STAR; // Reduced potential energy

    std::cout << " Reduced potential with long range correction: " << PEstar << std::endl;

    clock_t end = clock();
    double time = double(end - begin) / (double)CLOCKS_PER_SEC;
    std::cout << "Time elapsed: \n";
    std::cout << time << " seconds \n";
    std::cout << time / 60 << " minutes" << std::endl;

    positionFile.close();
    debug.close();
    energyFile.close();

    std::ifstream xyzFile;
    xyzFile.open("out.xyz");
    std::cout << "Finding radial distribution \n";
    std::cout << radialDistribution(xyzFile) << std::endl;

    return 0;
}

double dot (double x, double y, double z) { // Returns dot product of a vector
    return x * x + y * y + z * z;
}

void thermostat(std::vector<Atom> &atomList) {
    double instantTemp = 0;
    for (int i = 0; i < N; i++) {
        instantTemp += MASS * dot(atomList[i].velocities[0], atomList[i].velocities[1], atomList[i].velocities[2]);
    }

    instantTemp /= (3 * N - 3);
    double tempScalar = std::sqrt(TARGET_TEMP / instantTemp);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; ++j) {
            atomList[i].velocities[j] *= tempScalar; // V = V * lambda
        }
    }
}

double calcForces(std::vector<Atom> &atomList) {

    double netPotential = 0;

    for (int j = 0; j < N; j++) { // Set all accelerations in every atom equao to zero
        for (int i = 0; i < 3; ++i) {
            atomList[j].accelerations[i] = 0;
        }
    }

    double distArr[3];
    // Iterate over all pairs of atoms once in atom list
    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {

            for (int k = 0; k < 3; ++k) { // Find distance between atoms 
                distArr[k] = atomList[i].positions[k] - atomList[j].positions[k];
                // Boundary conditions - Calculate interaction through wall if thats closest distance between molecules
                distArr[k] = distArr[k] - L * std::round(distArr[k] / L); 
            }

            double r2 = dot(distArr[0], distArr[1], distArr[2]);
            double r = std::sqrt(r2); // Magnitude of distance between the atoms

            if (r <= rCutoff) {
                double sor = SIGMA / r; // SIGMA over r
                double sor6 = std::pow(sor, 6);
                double sor12 = sor6 * sor6;

                double forceOverR = 24 * EPS_STAR / r2 * (2 * sor12 - sor6);
                netPotential += 4 * EPS_STAR * (sor12 - sor6);


                for (int k = 0; k < 3; ++k) {
                    // force divided by radius divided by the magnitude of the distance vector in direction k
                    // gives the force multiplied by the unit vector direction
                    atomList[i].accelerations[k] += (forceOverR * distArr[k] / MASS);
                    atomList[j].accelerations[k] -= (forceOverR * distArr[k] / MASS);
                }
            }

        }
    }
    return netPotential;
}

std::vector<Atom> simpleCubicCell() {
    double n = std::cbrt(N);

    std::vector<Atom> atomList;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                atomList.push_back(Atom(i * SIGMA, j * SIGMA, k * SIGMA));
            }
        }
    }
    return atomList;
}

std::vector<Atom> faceCenteredCell() {
    // Each face centered unit cell has four atoms
    // Method creates a cube of face centered unit cells

    double n = std::cbrt(N / 4.0); // Number of unit cells in each direction
    double dr = L / n; // Distance between two corners in a unit cell
    double dro2 = dr / 2.0; // dr over 2

    std::vector<Atom> atomList;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                atomList.push_back(Atom(i * dr, j * dr, k * dr));
                atomList.push_back(Atom(i * dr + dro2, j * dr + dro2, k * dr));
                atomList.push_back(Atom(i * dr + dro2, j * dr, k * dr + dro2));
                atomList.push_back(Atom(i * dr, j * dr + dro2, k * dr + dro2));
            }
        }
    }
    return atomList;
}


double radialDistribution(std::ifstream &xyzFile) {
    
    int i, j, n;
    int row;
    double dx, dy, dz, r, rij;
    double g[100];
    double x[N], y[N], z[N];

    double dr = L / 2.0 / 100;
    
    
    return -1.0;
}

