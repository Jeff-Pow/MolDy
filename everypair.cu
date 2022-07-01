/*
Host: CPU
Device: GPU
    __global__ - Runs on the GPU, called from the CPU or the GPU*. Executed with <<<dim3>>> arguments.
    __device__ - Runs on the GPU, called from the GPU. Can be used with variabiles too.
    __host__ - Runs on the CPU, called from the CPU.
     __global__ functions can be called from other __global__ functions starting compute capability 3.5.
*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuPrintf.cu"
#include "cuPrintf.cuh"
#include <cuda.h>

#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <fstream>
#include <array>
#include <string>

struct Atom {
    double positions[3];
    double velocities[3] = {0,0,0};
    double accelerations[3] = {0,0,0};
    double oldAccelerations[3] = {0,0,0};

    Atom(double x, double y, double z) {
        positions[0] = x;
        positions[1] = y;
        positions[2] = z;
    }
    Atom() {

    }
};

const double Kb = 1.38064582 * std::pow(10, -23); // J / K
const double Na = 6.022 * std::pow(10, 23); // Atoms per mole

const int numTimeSteps = 5000; // Parameters to change for simulation
const double dt_star= .001;

const int N = 32; // Number of atoms in simulation
const double SIGMA = 3.405; // Angstroms
const double EPSILON = 1.6540 * std::pow(10, -21); // Joules
const double EPS_STAR = EPSILON / Kb; // ~ 119.8 K

const double rhostar = .45; // Dimensionless density of gas
const double rho = rhostar / std::pow(SIGMA, 3); // Density of gas
const double L = std::cbrt(N / rho); // Unit cell length
const double rCutoff = SIGMA * 2.5; // Forces are negligible past this distance, not worth calculating
const double rCutoffSquared = rCutoff * rCutoff;
const double tStar = 1.24; // Reduced units of temperature
const double TARGET_TEMP = tStar * EPS_STAR;
// 39.9 is mass of argon in amu, 10 is a conversion between the missing units :)
const double MASS = 39.9 * 10 / Na / Kb; // Kelvin * ps^2 / A^2
const double timeStep = dt_star * std::sqrt(MASS * SIGMA * SIGMA / EPS_STAR); // Convert time step to picoseconds

// Declarations
void thermostat(Atom *atomList);
double calcForces(Atom *atomList);
thrust::host_vector<Atom> faceCenteredCell();
std::vector<Atom> simpleCubicCell();
void radialDistribution();



__host__
void writePositions(Atom *atomList, std::ofstream &positionFile, int i) {
    positionFile << N << "\nTime: " << i << "\n";
    for (int j = 0; j < N; ++j) { // Write positions to xyz file
        positionFile << "A " << atomList[j].positions[0] << " " << atomList[j].positions[1] << " " << atomList[j].positions[2] << "\n";
    }
}

int main() {

    std::ofstream positionFile("out.xyz");
    //std::ofstream debug("debug.dat");

    // Arrays to hold energy values at each step of the process
    std::vector<double> KE;
    std::vector<double> PE;
    std::vector<double> netE;

    std::random_device rd;
    std::default_random_engine generator(3); // (rd())
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    thrust::host_vector<Atom> atomList = faceCenteredCell();
    Atom atoms[N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            atoms[i].positions[j] = atomList[i].positions[j];
            atoms[i].velocities[j] = atomList[j].velocities[j];
        }
    }

    for (int i = 0; i < N; ++i) { // Randomize velocities
         for (int j = 0; j < 3; ++j) {
             atoms[i].velocities[j] = distribution(generator);
         }
    }
   
    thermostat(atoms); // Make velocities more accurate

    double totalVelSquared;
    double netPotential;

    double count = .01;
    for (int i = 0; i < numTimeSteps; ++i) { // Main loop handles integration and printing to files

        if (i > count * numTimeSteps) { // Percent progress
            std::cout << count * 100 << "% \n";
            count += .01;
        }

        writePositions(atoms, positionFile, i);

        for (int k = 0; k < N; ++k) { // Update positions
            for (int j = 0; j < 3; ++j) {
                atoms[k].positions[j] += atoms[k].velocities[j] * timeStep 
                    + .5 * atoms[k].accelerations[j] * timeStep * timeStep;
                atoms[k].positions[j] += -L * std::floor(atoms[k].positions[j] / L); // Keep atom inside box
                atoms[k].oldAccelerations[j] = atomList[k].accelerations[j];
            }
        }

        netPotential = calcForces(atoms); // Update accelerations and return potential of system

        totalVelSquared = 0;
        for (int k = 0; k < N; ++k) { // Update velocities
            for (int j = 0; j < 3; ++j) {
                atoms[k].velocities[j] += .5 * (atoms[k].accelerations[j] + atoms[k].oldAccelerations[j]) * timeStep;
                totalVelSquared += atoms[k].velocities[j] * atoms[k].velocities[j];
            }
        }

        if (i < numTimeSteps / 2 && i % 5 == 0) { // Apply velocity modifications for first half of sample
            thermostat(atoms);
        }

        if (i > numTimeSteps / 2) { // Record energies after half of time has passed
            double netKE = .5 * MASS * totalVelSquared;
            KE.push_back(netKE);
            PE.push_back(netPotential);
            netE.push_back(netPotential + netKE);
        }
    }

    double avgPE = 0; // Average PE array
    for (double i : PE) {
        avgPE += i;
    }
    avgPE /= PE.size();
    std::cout << "Avg PE: " << avgPE << std::endl;

    double SoLo2 = SIGMA / (L / 2); // Sigma over L over 2
    double Ulrc = (8.0 / 3.0) * M_PI * N * rhostar * EPS_STAR; // Potential sub lrc (long range corrections)
    double temp = 1.0 / 3.0 * std::pow(SoLo2, 9.0);
    double temp1 = std::pow(SoLo2, 3.0);
    Ulrc *= (temp - temp1);
    double PEstar = ((avgPE + Ulrc) / N) / EPS_STAR; // Reduced potential energy

    std::cout << "Reduced potential with long range correction: " << PEstar << std::endl;

    positionFile.close();
    //debug.close();

    // std::cout << "Finding radial distribution \n";
    // radialDistribution(); // Comment out function to reduce runtime

    return 0;
}

__device__
void dotForGPU(double x, double y, double z, double &r2) { // Returns dot product of a vector
    r2 = x * x + y * y + z * z;
}

__host__
double dotForCPU(double x, double y, double z) {
    return x * x + y * y + z * z;
}

__host__
void thermostat(Atom *atomList) {
    double instantTemp = 0;
    for (int i = 0; i < N; i++) { // Add kinetic energy of each molecule to the temperature
        instantTemp += MASS * dotForCPU(atomList[i].velocities[0], atomList[i].velocities[1], atomList[i].velocities[2]);
    }
    instantTemp /= (3 * N - 3);
    double tempScalar = std::sqrt(TARGET_TEMP / instantTemp);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; ++j) {
            atomList[i].velocities[j] *= tempScalar; // V = V * lambda
        }
    }
}

__global__
void calcForcesPerAtom(Atom *devAtoms, int atomidx, double L, double EPS_STAR, double MASS, double *netPotential) {
    double distArr[3]; // Record distance between atoms
    double localPotential = 0;
    double r2;

    for(int j = 0; j < N; j++) {
        for (int k = 0; k < 3; k++) {
            // Apply boundary conditions
            distArr[k] = devAtoms[atomidx].positions[k] - devAtoms[j].positions[k];
            distArr[k] -= L * std::round(distArr[k] / L);
        }
        r2 = distArr[0] * distArr[0] + distArr[1] * distArr[1] + distArr[2] * distArr[2]; // Dot product b/t atoms
        if (r2 < rCutoffSquared) {
            double s2or2 = SIGMA * SIGMA / r2; // Sigma squared over r squared
            double sor6 = s2or2 * s2or2 * s2or2; // Sigma over r to the sixth
            double sor12 = sor6 * sor6; // Sigma over r to the twelfth

            double forceOverR = 24 * EPS_STAR / r2 * (2 * sor12 - sor6);
            localPotential += 4 * EPS_STAR * (sor12 - sor6);
            for (int k = 0; k < 3; k++) {
                devAtoms[atomidx].accelerations[k] += (forceOverR * distArr[k] / MASS);
                devAtoms[j].accelerations[k] -= (forceOverR * distArr[k] / MASS);
            }
        }
    }
    *netPotential += localPotential;
}

__host__
double calcForces(Atom *atomList) { // Cell pairs method to calculate forces
    double *netPotential;
    cudaMallocManaged(&netPotential, sizeof(double));
    *netPotential = 0;

    for (int j = 0; j < N; j++) { // Set all accelerations equal to zero
        for (int i = 0; i < 3; ++i) {
            atomList[j].accelerations[i] = 0;
        }
    }
    Atom *devAtoms;
    cudaMallocManaged(&devAtoms, N * sizeof(Atom));
    cudaMemcpy(devAtoms, atomList, N * sizeof(Atom), cudaMemcpyHostToDevice);

    for (int c = 0; c < N; c++) {
         calcForcesPerAtom<<<1, 1>>>(devAtoms, c, L, EPS_STAR, MASS, netPotential);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(atomList, devAtoms, N * sizeof(Atom), cudaMemcpyDeviceToHost);
    cudaFree(devAtoms);
    double result = *netPotential;
    cudaFree(netPotential);
    return result;
}

__host__
std::vector<Atom> simpleCubicCell() {
    double n = std::cbrt(N); // Number of atoms in each dimension

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

__host__
thrust::host_vector<Atom> faceCenteredCell() {
    // Each face centered unit cell has four atoms
    // Method creates a cubic arrangement of face centered unit cells

    double n = std::cbrt(N / 4.0); // Number of unit cells in each direction
    double dr = L / n; // Distance between two corners in a unit cell
    double dro2 = dr / 2.0; // dr over 2

    thrust::host_vector<Atom> atomList;

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


__host__
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
                    
                    double r = std::sqrt(dotForCPU(xDif, yDif, zDif));

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
