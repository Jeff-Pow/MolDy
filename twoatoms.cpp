//
// Created by Jeff Powell on 5/18/2022.
//

#include <cstdio>
#include <iostream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <string>
#include <array>


struct Atom {
public:
    std::array<double, 3> positions;
    std::array<double, 3> velocities;
    std::array<double, 3> accelerations;
    std::array<double, 3> oldAccelerations;
    char atomType;
    double mass;

    Atom(double x, double y, double z, char element, double m) {
        positions[0] = x;
        positions[1] = y;
        positions[2] = z;
        accelerations = {0, 0, 0};
        oldAccelerations = {0, 0, 0};
        atomType = element;
        mass = m;
    }
};

/****
 * Sigma between two atoms is one half of the sum
 * Epsilon between two atoms is the square root of the product
 * https://www.hindawi.com/journals/jther/2013/828620/tab1/
 ****/


const double Kb = 1.38064582 * std::pow(10, -23); // J / K
const double Na = 6.022 * std::pow(10, 23); // Atoms per mole

const int numTimeSteps = 15000; // Parameters to change for simulation
const double dt_star= .001;

const int N = 864;
const double SIGMA_NE = 2.775; // Angstroms
const double EPSILON_NE = 1.6540 * std::pow(10, -21); // Joules
const double EPS_STAR_NE = 36.831;
const double SIGMA_XE = 4.055;
const double EPS_STAR_XE = 218.18;

const double MIXED_SIGMA = .5 * (SIGMA_NE + SIGMA_XE);
const double MIXED_EPS = std::sqrt(EPS_STAR_NE * EPS_STAR_XE);

const double rhostar = .6; // Dimensionless density of gas
const double rho = rhostar / std::pow(MIXED_SIGMA, 3); // Density of gas
const double L = std::cbrt(N / rho); // Unit cell length
const double rCutoff = SIGMA_XE * 2.5; // Forces are negligible past this distance, not worth calculating
const double rCutoffSquared = rCutoff * rCutoff;
double tStar; // Reduced units of temperature
double TARGET_TEMP = tStar * MIXED_EPS;
// 39.9 is mass of argon in amu, 10 is a conversion between the missing units :)
const double NE_MASS = 20.18 * 10 / Na / Kb; // Kelvin * ps^2 / A^2
const double XE_MASS = 131.29 * 10 / Na / Kb;
const double AVG_MASS = .5 * (NE_MASS + XE_MASS);
const double timeStep = dt_star * std::sqrt((.5 * (XE_MASS + NE_MASS)) * MIXED_SIGMA * MIXED_SIGMA / MIXED_EPS); // Convert time step to picoseconds

double dot(double x, double y, double z);
void thermostat(std::vector<Atom> &atomList);
double calcForces(std::vector<Atom> &atomList, std::ofstream &debug);
std::vector<Atom> faceCenteredCell();
std::vector<Atom> simpleCubicCell();
void radialDistribution();

int main(int argc, char* argv[]) {
    // .1 Creates a liquid in 15000 timesteps
    tStar = std::stod(argv[1]);
    TARGET_TEMP = tStar * MIXED_EPS;
    std::ofstream positionFile("out.xyz");
    std::ofstream debug("debug.dat");
    std::ofstream energyFile("Energy.dat");

    // Arrays to hold energy values at each step of the process
    std::vector<double> KE;
    std::vector<double> PE;
    std::vector<double> netE;

    std::random_device rd;
    std::default_random_engine generator(3); // (rd())
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

    double count = .01;
    for (int i = 0; i < numTimeSteps; ++i) { // Main loop handles integration and printing to files

        if (i > count * numTimeSteps) { // Percent progress
            std::cerr << count * 100 << "% \n";
            count += .01;
        }
        
            positionFile << N << "\nTime: " << i << "\n";
            //debug << "Time: " << i << "\n";

            for (int j = 0; j < N; ++j) { // Write positions to xyz file
                positionFile << atomList[j].atomType << " " << atomList[j].positions[0] << " " << 
                                atomList[j].positions[1] << " " << atomList[j].positions[2] << "\n";
                //debug << "Atom number: " << j << "\n";
                //debug << "Positions: " << atomList[j].positions[0] << " " << atomList[j].positions[1] << " " << atomList[j].positions[2] << "\n";
                //debug << "Velocities: " << atomList[j].velocities[0] << " " << atomList[j].velocities[1]  << " " << atomList[j].velocities[2] << "\n";
                //debug << "Accelerations: " << atomList[j].accelerations[0] << " " << atomList[j].accelerations[1]  << " " << atomList[j].accelerations[2] << "\n";
                //debug << "- \n";
            }
            //debug << "------------------------------------------------- \n";

        for (int k = 0; k < N; ++k) { // Update positions
            for (int j = 0; j < 3; ++j) {
                atomList[k].positions[j] += atomList[k].velocities[j] * timeStep 
                    + .5 * atomList[k].accelerations[j] * timeStep * timeStep;
                atomList[k].positions[j] += -L * std::floor(atomList[k].positions[j] / L); // Keep atom inside box
                atomList[k].oldAccelerations[j] = atomList[k].accelerations[j];
            }
        }
        
        netPotential = calcForces(atomList, debug); // Update accelerations and return potential of system

        totalVelSquared = 0;
        for (int k = 0; k < N; ++k) { // Update velocities
            for (int j = 0; j < 3; ++j) {
                atomList[k].velocities[j] += .5 * (atomList[k].accelerations[j] + atomList[k].oldAccelerations[j]) * timeStep;
                totalVelSquared += atomList[k].velocities[j] * atomList[k].velocities[j];
            }
        }

        if (i < numTimeSteps / 2 && i % 5 == 0) { // Apply velocity modifications for first half of sample
            thermostat(atomList);
        }

        if (i > numTimeSteps / 2) { // Record energies to arrays and file after half of time has passed
            double netKE = .5 * AVG_MASS * totalVelSquared;
            KE.push_back(netKE);
            PE.push_back(netPotential);
            netE.push_back(netPotential + netKE);
            //energyFile << "Time: " << i << "\n";
            //energyFile << "KE: " << netKE << "\n";
            //energyFile << "PE: " << netPotential << "\n";
            //energyFile << "Total energy: " << netPotential + netKE << "\n";
            //energyFile << "------------------------------------------ \n";
            energyFile << netKE << "," <<  netPotential << "," << netPotential + netKE << "\n";
        }
    }

    double avgPE = 0; // Average PE array
    for (int i = 0; i < PE.size(); i++) {
        avgPE += PE[i];
    }
    avgPE /= PE.size();

    double avgE = 0;
    for (int i = 0; i < netE.size(); i++) {
        avgE += netE[i];
    }
    avgE /= netE.size();
    std::cout << 1 / tStar << " " << avgE << std::endl;
    // std::cout << "Average energy: " << avgE << std::endl;

    double SoLo2 = MIXED_SIGMA / (L / 2); // Sigma over L over 2
    double Ulrc = (8.0 / 3.0) * M_PI * N * rhostar * MIXED_EPS; // Potential sub lrc (long range corrections)
    double temp = 1.0 / 3.0 * std::pow(SoLo2, 9.0);
    double temp1 = std::pow(SoLo2, 3.0);
    Ulrc *= (temp - temp1);
    double PEstar = ((avgPE + Ulrc) / N) / MIXED_EPS; // Reduced potential energy

    std::cerr << "Reduced potential with long range correction: " << PEstar << std::endl;

    positionFile.close();
    debug.close();
    energyFile.close();

    // std::cout << "Finding radial distribution \n";
    // radialDistribution(); // Comment out function to reduce runtime

    return 0;
}

double dot (double x, double y, double z) { // Returns dot product of a vector
    return x * x + y * y + z * z;
}

void thermostat(std::vector<Atom> &atomList) {
    double instantTemp = 0;
    for (int i = 0; i < N; i++) { // Add kinetic energy of each molecule to the temperature
        instantTemp += atomList[i].mass * dot(atomList[i].velocities[0], atomList[i].velocities[1], atomList[i].velocities[2]);
    }
    instantTemp /= (3 * N - 3);
    double tempScalar = std::sqrt(TARGET_TEMP / instantTemp);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; ++j) {
            atomList[i].velocities[j] *= tempScalar; // V = V * lambda
        }
    }
}

double calcForces(std::vector<Atom> &atomList, std::ofstream &debug) { // Cell pairs method to calculate forces

    double netPotential = 0;
    const double targetCellLength = rCutoff;
    const int numCellsPerDirection = std::floor(L / targetCellLength);
    double cellLength = L / numCellsPerDirection; // Side length of each cell
    int numCellsYZ = numCellsPerDirection * numCellsPerDirection; // Number of cells in one plane
    const int numCellsXYZ = numCellsYZ * numCellsPerDirection; // Number of cells in the simulation
    std::array<int, N> pointerArr; // Array pointing to the next lowest atom in the cell
    int header[numCellsXYZ]; // Array pointing at the highest numbered atom in each cell
    std::array<int, 3> mc; // Array to keep track of coordinates of a cell
    std::array<int, 3> mc1; // Array to keep track of the coordinates of a neighboring cell
    std::array<int, 3> shiftedNeighbor; // Boundary conditions
    int c, c1; // Convert coordinates of a cell into an index
    std::array<double, 3> distArr; // Array for distance between atoms


    for (int j = 0; j < N; j++) { // Set all accelerations in every atom equal to zero
        for (int i = 0; i < 3; ++i) {
            atomList[j].accelerations[i] = 0;
        }
    }

    for (c = 0; c < numCellsXYZ; c++) { // Initialize all cells in header array to empty
        header[c] = -1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) { 
            mc[j] = atomList[i].positions[j] / cellLength; // Find the coordinates of a cell an atom belongs to
        }
        // Turn coordinates of cell into a cell index for the header array
        c = mc[0] * numCellsYZ + mc[1] * numCellsPerDirection + mc[2];
        // Link current atom to previous occupant
        pointerArr[i] = header[c];
        // Current atom is the highest in its cell, so it goes in the header
        header[c] = i;
    }

    for (mc[0] = 0; mc[0] < numCellsPerDirection; (mc[0])++) { // Calculate coordinates of a cell to work in
        for (mc[1] = 0; mc[1] <  numCellsPerDirection; (mc[1])++) {
            for (mc[2] = 0; mc[2] < numCellsPerDirection; (mc[2])++) {
                // Calculate index of the current cell we're working in
                c = mc[0] * numCellsYZ + mc[1] * numCellsPerDirection + mc[2];

                // Scan neighbor cells including the one currently active
                for (mc1[0] = mc[0] - 1; mc1[0] < mc[0] + 2; mc1[0]++) {
                    for (mc1[1] = mc[1] - 1; mc1[1] < mc[1] + 2; mc1[1]++) {
                        for (mc1[2] = mc[2] - 1; mc1[2] < mc[2] + 2; mc1[2]++) {

                            for (int k = 0; k < 3; k++) { // Boundary conditions
                                shiftedNeighbor[k] = (mc1[k] + numCellsPerDirection) % numCellsPerDirection;
                            }
                            // Scalar index of neighboring cell
                            c1 = shiftedNeighbor[0] * numCellsYZ + shiftedNeighbor[1] * numCellsPerDirection + shiftedNeighbor[2];

                            int i = header[c]; // Find the highest numbered atom in each cell
                            double r2; // Dot product between two atoms
                            while (i > -1) {
                                int j = header[c1]; // Scan atom with the largest index in neighboring cell c1
                                while (j > -1) {
                                    if (i < j) { // Don't double count atoms (if i > j its already been counted)
                                        for (int k = 0; k < 3; k++) {
                                            // Apply boundary conditions
                                            distArr[k] = atomList[i].positions[k] - atomList[j].positions[k];
                                            distArr[k] = distArr[k] - L * std::round(distArr[k] / L);
                                        }
                                        r2 = dot(distArr[0], distArr[1], distArr[2]); // Dot of distance vector between the two atoms
                                        if (r2 < rCutoffSquared) {
                                            double sig, eps;
                                            if (atomList[i].atomType == 'A' && atomList[j].atomType == 'A') {
                                                sig = SIGMA_NE;
                                                eps = EPS_STAR_NE;
                                            }
                                            else if (atomList[i].atomType == 'X' && atomList[j].atomType == 'X') {
                                                sig = SIGMA_XE;
                                                eps = EPS_STAR_XE;
                                            }
                                            else {
                                                sig = MIXED_SIGMA;
                                                eps = MIXED_EPS;
                                            }

                                            double s2or2 = sig * sig / r2; // Sigma squared over r squared
                                            double sor6 = s2or2 * s2or2 * s2or2; // Sigma over r to the sixth
                                            double sor12 = sor6 * sor6; // Sigma over r to the twelfth

                                            double forceOverR = 24 * eps / r2 * (2 * sor12 - sor6);
                                            netPotential += 4 * eps * (sor12 - sor6);
                                            // debug << i << " on " << j << ": " << forceOverR << "\n";

                                            for (int k = 0; k < 3; k++) {
                                                atomList[i].accelerations[k] += (forceOverR * distArr[k] / atomList[i].mass);
                                                atomList[j].accelerations[k] -= (forceOverR * distArr[k] / atomList[j].mass);
                                            }
                                        } 
                                    }
                                    j = pointerArr[j];
                                }
                                i = pointerArr[i];
                            }
                        }
                    }
                }
            }
        }
    }
    return netPotential;
}

std::vector<Atom> simpleCubicCell() {
    double n = std::cbrt(N); // Number of atoms in each dimension
    int count = 0;

    std::vector<Atom> atomList;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if (count % 2 == 0) {
                    atomList.emplace_back(i * MIXED_SIGMA, j * MIXED_SIGMA, k * MIXED_SIGMA, 'A', NE_MASS);
                }
                else {
                    atomList.emplace_back(i * MIXED_SIGMA, j * MIXED_SIGMA, k * MIXED_SIGMA, 'X', XE_MASS);
                }
                count++;
            }
        }
    }
    return atomList;
}

std::vector<Atom> faceCenteredCell() {
    // Each face centered unit cell has four atoms
    // Method creates a cubic arrangement of face centered unit cells

    double n = std::cbrt(N / 4.0); // Number of unit cells in each direction
    double dr = L / n; // Distance between two corners in a unit cell
    double dro2 = dr / 2.0; // dr over 2

    std::vector<Atom> atomList;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                atomList.emplace_back(i * dr, j * dr, k * dr, 'A', NE_MASS);
                atomList.emplace_back(i * dr + dro2, j * dr + dro2, k * dr, 'X', XE_MASS);
                atomList.emplace_back(i * dr + dro2, j * dr, k * dr + dro2, 'A', NE_MASS);
                atomList.emplace_back(i * dr, j * dr + dro2, k * dr + dro2, 'X', XE_MASS);
            }
        }
    }
    return atomList;
}


void radialDistribution() {
    
    std::string line;
    std::string s;

    int numDataPts = 100;
    double data[numDataPts];
    std::array<double, N> x;
    std::array<double, N> y;
    std::array<double, N> z;
    // Arrays hold coordinates of each atom at each step
    double dr = L / 2.0 / 100;

    std::ifstream xyz ("out.xyz");

    for (int i = 0; i < numTimeSteps; i++) {

        std::getline(xyz, line); // Skips line with number of molecules
        std::getline(xyz, line); // Skips comment line

        for (int row = 0; row < N; row++) {
            std::getline(xyz, line);
            std::istringstream iss( line );

            iss >> s >> x[row] >> y[row] >> z[row]; // Drop atom type, store coordinates of each atom
        }
        

        if (i >= numTimeSteps / 2) {
            for (int j = 0; j < N - 1; j++) {
                for (int k = j + 1; k < N; k++) {
                    double xDif = x[j] - x[k]; // Distance between atoms in x direction
                    xDif = xDif - L * std::round(xDif / L); // Boundary conditions
                    double yDif = y[j] - y[k];
                    yDif = yDif - L * std::round(yDif / L);
                    double zDif = z[j] - z[k];
                    zDif = zDif - L * std::round(zDif / L);
                    
                    double r = std::sqrt(dot(xDif, yDif, zDif));

                    if (r < L/2.0) {
                        data[(int)(r / dr)] += 2.0;
                    }
                }
            }
        }
    }
    xyz.close();
    std::ofstream radialData("Radial_Data.dat");

    radialData << "r \t \t g(r) \n";
    for (int i = 0; i < numDataPts; i++) {
        double r = (i + .5) * dr;
        data[i] /= (numTimeSteps / 2.0);
        data[i] /= 4.0 * M_PI / 3.0 * (std::pow(i + 1, 3.0) - std::pow(i, 3.0)) * std::pow(dr, 3.0) * rho;
        radialData << r << " , " << data[i] / N << "\n";
    }
    radialData.close();
}