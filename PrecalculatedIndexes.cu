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

#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <fstream>
#include <array>
#include <string>

struct Atom {
    std::array<double, 3> positions;
    std::array<double, 3> velocities;
    std::array<double, 3> accelerations;
    std::array<double, 3> oldAccelerations;

    Atom(double x, double y, double z) {
        positions[0] = x;
        positions[1] = y;
        positions[2] = z;
        accelerations = {0, 0, 0};
        oldAccelerations = {0, 0, 0};
    }
};

const double Kb = 1.38064582 * std::pow(10, -23); // J / K
const double Na = 6.022 * std::pow(10, 23); // Atoms per mole

const int numTimeSteps = 5000; // Parameters to change for simulation
const double dt_star= .001;

const int N = 4000; // Number of atoms in simulation
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

const double targetCellLength = rCutoff;
const int numCellsPerDirection = std::floor(L / targetCellLength);
const double cellLength = L / numCellsPerDirection; // Side length of each cell

double dot(double x, double y, double z);
void thermostat(thrust::host_vector<Atom> &atomList);
double calcForces(thrust::host_vector<Atom> &atomList, int (&cellInteractionIndexes)[343][14]);
thrust::host_vector<Atom> faceCenteredCell();
std::vector<Atom> simpleCubicCell();
void radialDistribution();



__host__
void writePositions(thrust::host_vector<Atom> &atomList, std::ofstream &positionFile, int i) {
    positionFile << N << "\nTime: " << i << "\n";
    for (int j = 0; j < N; ++j) { // Write positions to xyz file
        positionFile << "A " << atomList[j].positions[0] << " " << atomList[j].positions[1] << " " << atomList[j].positions[2] << "\n";
    }
}

__host__
int calcCellIndex(int x, int y, int z) {
    return x * numCellsPerDirection * numCellsPerDirection + y * numCellsPerDirection + z;
}

__host__
std::array<int, 3> calcCellFromIndex(int index) {
    std::array<int, 3> arr;
    int numCellsYZ = numCellsPerDirection * numCellsPerDirection;
    arr[0] = index / numCellsYZ;
    int remainder = index % numCellsYZ;
    arr[1] = remainder / numCellsPerDirection;
    arr[2] = remainder % numCellsPerDirection;
    return arr;
}

__host__
std::array<int, 3> moveCellInsideBox(int x, int y, int z) {
    std::array<int, 3> cell;
    cell[0] = (x + numCellsPerDirection) % numCellsPerDirection;
    cell[1] = (y + numCellsPerDirection) % numCellsPerDirection;
    cell[2] = (z + numCellsPerDirection) % numCellsPerDirection;
    return cell;
}

__host__
int processCell(int x, int y, int z) {
    std::array<int, 3> shiftedNeighbor = moveCellInsideBox(x, y, z);
    int index = calcCellIndex(shiftedNeighbor[0], shiftedNeighbor[1], shiftedNeighbor[2]);
    return index;
}

__host__
void calcCellInteractions(std::vector<std::vector<int>> &cellInteractionIndexes, int numCellsXYZ) {
    for (int i = 0; i < numCellsXYZ; i++) {
        std::vector<int> arr;
        std::array<int, 3> cell = calcCellFromIndex(i);

        arr.push_back(processCell(cell[0], cell[1], cell[2]));
        arr.push_back(processCell(cell[0], cell[1], cell[2] + 1));
        arr.push_back(processCell(cell[0], cell[1] + 1, cell[2] - 1));
        arr.push_back(processCell(cell[0], cell[1] + 1, cell[2]));
        arr.push_back(processCell(cell[0], cell[1] + 1, cell[2] + 1));

        // Next level above
        arr.push_back(processCell(cell[0] + 1, cell[1] - 1, cell[2] - 1));
        arr.push_back(processCell(cell[0] + 1, cell[1] - 1, cell[2]));
        arr.push_back(processCell(cell[0] + 1, cell[1] - 1, cell[2] + 1));
        arr.push_back(processCell(cell[0] + 1, cell[1], cell[2] - 1));
        arr.push_back(processCell(cell[0] + 1, cell[1], cell[2]));
        arr.push_back(processCell(cell[0] + 1, cell[1], cell[2] + 1));
        arr.push_back(processCell(cell[0] + 1, cell[1] + 1, cell[2] - 1));
        arr.push_back(processCell(cell[0] + 1, cell[1] + 1, cell[2]));
        arr.push_back(processCell(cell[0] + 1, cell[1] + 1, cell[2] + 1));

        cellInteractionIndexes[i] = arr;
    }
}

int main() {
    int numCellsYZ = numCellsPerDirection * numCellsPerDirection;
    int numCellsXYZ = numCellsPerDirection * numCellsYZ;
    std::vector<std::vector<int>> vectorIndicies;

    std::cout << "Cells per direction: " << numCellsPerDirection << std::endl;
    std::cout << "Simulation length: " << L << std::endl;
    std::cout << "Cell length: " << cellLength << std::endl;

    std::ofstream positionFile("out.xyz");
    //std::ofstream debug("debug.dat");
    //debug << "I \t J \t C \t C1 \t R2 \t forceOverR \n";

    calcCellInteractions(vectorIndicies, numCellsXYZ);

    int cellInteractionIndexes[numCellsXYZ][14]; // Interactions indexes in an array to make CUDA happy

    for (int i = 0; i < vectorIndicies.size(); i++) {
        for (int j = 0; j < vectorIndicies[i].size(); j++) {
            cellInteractionIndexes[i][j] = vectorIndicies[i][j];
        }
    }


    // Arrays to hold energy values at each step of the process
    std::vector<double> KE;
    std::vector<double> PE;
    std::vector<double> netE;

    std::random_device rd;
    std::default_random_engine generator(3); // (rd())
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    thrust::host_vector<Atom> atomList = faceCenteredCell();

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
            std::cout << count * 100 << "% \n";
            count += .01;
        }

        writePositions(atomList, positionFile, i);

        for (int k = 0; k < N; ++k) { // Update positions
            for (int j = 0; j < 3; ++j) {
                atomList[k].positions[j] += atomList[k].velocities[j] * timeStep 
                    + .5 * atomList[k].accelerations[j] * timeStep * timeStep;
                atomList[k].positions[j] += -L * std::floor(atomList[k].positions[j] / L); // Keep atom inside box
                atomList[k].oldAccelerations[j] = atomList[k].accelerations[j];
            }
        }

        netPotential = calcForces(atomList, cellInteractionIndexes); // Update accelerations and return potential of system

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
void thermostat(thrust::host_vector<Atom> &atomList) {
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
void calcForcesOnCell(int cellI, thrust::device_vector<Atom> &gpuAtom, int (&cellInteractionIndexes)[343][14], thrust::device_vector<thrust::device_vector<int>> &atomsInCells) {
    double distArr[3]; // Record distance between atoms
    double netPotential = 0;
    auto cellArr = atomsInCells[cellI];
    double r2;

    // Scan neighbor cells including the one currently active
    for (int cellJ : cellInteractionIndexes[cellI]) {
        auto neighborCellArr = atomsInCells[cellJ];

        for (int atomi : cellArr) {
            for (int atomj : neighborCellArr) {
                if (atomi < atomj || cellI != cellJ) { // Don't double count atoms (if i > j its already been counted)
                    for (int k = 0; k < 3; k++) {
                        // Apply boundary conditions
                        distArr[k] = gpuAtom[atomi].positions[k] - gpuAtom[atomj].positions[k];
                        distArr[k] -= L * std::round(distArr[k] / L);
                    }
                    dotForGPU(distArr[0], distArr[1], distArr[2], r2); // Dot of distance vector between the two atoms
                    if (r2 < rCutoffSquared) {
                        double s2or2 = SIGMA * SIGMA / r2; // Sigma squared over r squared
                        double sor6 = s2or2 * s2or2 * s2or2; // Sigma over r to the sixth
                        double sor12 = sor6 * sor6; // Sigma over r to the twelfth

                        double forceOverR = 24 * EPS_STAR / r2 * (2 * sor12 - sor6);
                        netPotential += 4 * EPS_STAR * (sor12 - sor6);
                        for (int k = 0; k < 3; k++) {
                            gpuAtom[atomi].accelerations[k] += (forceOverR * distArr[k] / MASS);
                            gpuAtom[atomj].accelerations[k] -= (forceOverR * distArr[k] / MASS);
                        }
                    }
                }
            }
        }
    }
}

__host__
double calcForces(thrust::host_vector<Atom> &atomList, int (&cellInteractionIndexes)[343][14]) { // Cell pairs method to calculate forces

    double netPotential = 0;
    int c; // Indexes of cell coordinates
    std::array<int, 3> cell; // Array to keep track of coordinates of a cell
    int numCellsYZ = numCellsPerDirection * numCellsPerDirection;
    int numCellsXYZ = numCellsPerDirection * numCellsPerDirection * numCellsPerDirection;
    thrust::device_vector<int> atomsInCells;
    atomsInCells.reserve(numCellsXYZ);
    thrust::device_vector<Atom> gpuAtoms = atomList;

    for (int j = 0; j < N; j++) { // Set all accelerations equal to zero
        for (int i = 0; i < 3; ++i) {
            atomList[j].accelerations[i] = 0;
        }
    }

    for (int i = 0; i < N; i++) { // Place atoms in cells
        for (int j = 0; j < 3; j++) {
            cell[j] = atomList[i].positions[j] / cellLength; // Find the coordinates of a cell an atom belongs to
        }
        // Turn coordinates of cell into a cell index for the header array
        c = cell[0] * numCellsYZ + cell[1] * numCellsPerDirection + cell[2];
        atomsInCells[c].push_back(i);
    }

    for (int c = 0; c < numCellsXYZ; c++) {
         calcForcesOnCell<<<1, 1>>>(c, gpuAtoms, cellInteractionIndexes, atomsInCells);
    }
    cudaDeviceSynchronize();
    atomList = gpuAtoms;
    return netPotential;
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
