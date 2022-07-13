/*
Host: CPU
Device: GPU
    __global__ - Runs on the GPU, called from the CPU or the GPU*. Executed with <<<dim3>>> arguments.
    __device__ - Runs on the GPU, called from the GPU. Can be used with variabiles too.
    __host__ - Runs on the CPU, called from the CPU.
     __global__ functions can be called from other __global__ functions starting compute capability 3.5.
*/

#include <cuda.h>
#include "helper_math.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <fstream>
#include <array>

const float Kb = 1.38064582e-23; // J / K
const float Na = 6.022e23; // Atoms per mole

const int numTimeSteps = 5000; // Parameters to change for simulation
const float dt_star= .001;

const int N = 4000; // Number of atoms in simulation
const float SIGMA = 3.405; // Angstroms
const float EPSILON = 1.6540e-21; // Joules
const float EPS_STAR = EPSILON / Kb; // ~ 119.8 K

const int numThreadsPerBlock = 1024;
const int numBlocks = ceil(N / numThreadsPerBlock) + 1;

const float rhostar = .45; // Dimensionless density of gas
const float rho = rhostar / std::pow(SIGMA, 3); // Density of gas
const float L = std::cbrt(N / rho); // Unit cell length
const float rCutoff = SIGMA * 2.5; // Forces are negligible past this distance, not worth calculating
const float rCutoffSquared = rCutoff * rCutoff;
const float tStar = 1.24; // Reduced units of temperature
const float TARGET_TEMP = tStar * EPS_STAR;
// 39.9 is mass of argon in amu, 10 is a conversion between the missing units :)
const float MASS = 39.9 * 10 / Na / Kb; // Kelvin * ps^2 / A^2
const float timeStep = dt_star * std::sqrt(MASS * SIGMA * SIGMA / EPS_STAR); // Convert time step to picoseconds

const int targetCellLength = rCutoff;
const int cellsPerDirection = floor(L / targetCellLength);
const float cellLength = L / cellsPerDirection;
const int cellsYZ = cellsPerDirection * cellsPerDirection;
const int cellsXYZ = cellsYZ * cellsPerDirection;

__host__ 
float calcVelocityDotProd(float3 *velocities) {
    float num = 0;
    for (int i = 0; i < N; i++) {
        num += dot(velocities[i], velocities[i]);
    }
    return num;
}

__host__
void faceCenteredCell(float3 *pos) {
    // Each face centered unit cell has four atoms
    // Method creates a cubic arrangement of face centered unit cells

    float n = std::cbrt(N / 4.0); // Number of unit cells in each direction
    float dr = L / n; // Distance between two corners in a unit cell
    float dro2 = dr / 2.0; // dr over 2

    int idx = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                pos[idx++] = make_float3(i * dr, j * dr, k * dr);
                pos[idx++] = make_float3(i * dr + dro2, j * dr + dro2, k * dr);
                pos[idx++] = make_float3(i * dr + dro2, j * dr, k * dr + dro2);
                pos[idx++] = make_float3(i * dr, j * dr + dro2, k * dr + dro2);
            }
        }
    }
}

__host__
void writePositions(float3 *positions, std::ofstream &positionFile, int i) {
    positionFile << N << "\nTime: " << i << "\n";
    for (int j = 0; j < N; ++j) { // Write positions to xyz file
        positionFile << "A " << positions[j].x << " " << positions[j].y << " " << positions[j].z << "\n";
    }

}

__host__
void writeToDebugFile(float3 *positions, float3 *velocities, float3 *accelerations, float3 *oldAccelerations, std::ofstream &debug, int i) {
    debug << "Time: " << i << "\n";
    for (int j = 0; j < N; j++) {
        debug << "Atom number: " << j << "\n";
        debug << "Positions: " << positions[j].x << " " << positions[j].y << " " << positions[j].z << "\n";
        debug << "Velocities: " << velocities[j].x << " " << velocities[j].y << " " << velocities[j].z << "\n";
        debug << "Accelerations: " << accelerations[j].x << " " << accelerations[j].y << " " << accelerations[j].z << "\n";
        debug << "Old accelerations: " << oldAccelerations[j].x << " " << oldAccelerations[j].y << " " << oldAccelerations[j].z << "\n";
        debug << "- \n";
    }
    debug << "------------------------------------------------- \n";
}

__global__ 
void thermostat(float3 *velocities, float totalVelSquared) {
    int k = blockIdx.x * numThreadsPerBlock + threadIdx.x;
    if (k >= N) return;
    float instantTemp = MASS * totalVelSquared;
    instantTemp /= (3 * N - 3);
    float tempScalar = std::sqrt(TARGET_TEMP / instantTemp);
    velocities[k] *= tempScalar; // V = V * lambda
}

__global__ 
void firstStep(float3 *positions, float3 *velocities, float3 *accelerations, float3 *oldAccelerations, float L, float timeStep) {
    int k = blockIdx.x * numThreadsPerBlock + threadIdx.x;
    if (k >= N) return;
    positions[k].x += velocities[k].x * timeStep + .5 * accelerations[k].x * timeStep * timeStep;
    positions[k].y += velocities[k].y * timeStep + .5 * accelerations[k].y * timeStep * timeStep;
    positions[k].z += velocities[k].z * timeStep + .5 * accelerations[k].z * timeStep * timeStep;
    positions[k].x += -L * floor(positions[k].x / L); // Keep atom inside box
    positions[k].y += -L * floor(positions[k].y / L); // Keep atom inside box
    positions[k].z += -L * floor(positions[k].z / L); // Keep atom inside box
    oldAccelerations[k] = accelerations[k];
    accelerations[k] = make_float3(0, 0, 0);
}

__global__ 
void thirdStep(float3 *velocities, float3 *accelerations, float3 *oldAccelerations, float L, float *totalVelSquared, float timeStep) {
    int k = blockIdx.x * numThreadsPerBlock + threadIdx.x;
    if (k >= N) return;
    velocities[k] += .5 * (accelerations[k] + oldAccelerations[k]) * timeStep;
    totalVelSquared[k] = dot(velocities[k], velocities[k]);
}

__global__ 
void sortAtoms(int *pointerArr, int *header, float3 *positions, int *cellLocation, float L) {
    int targetCellLength = rCutoff;
    int cellsPerDirection = floor(L / targetCellLength);
    float cellLength = L / cellsPerDirection;
    int cellsYZ = cellsPerDirection * cellsPerDirection;
    int cellsXYZ = cellsYZ * cellsPerDirection;
    for (int i = 0; i < cellsXYZ; i++) {
        header[i] = -1;
    }

    int3 cell;
    for (int i = 0; i < N; i++) {
        cell.x = positions[i].x / cellLength;
        cell.y = positions[i].y / cellLength;
        cell.z = positions[i].z / cellLength;
        // Turn coordinates of cell into index
        int c = cell.x * cellsYZ + cell.y * cellsPerDirection + cell.z;
        // Link current atom to previous occupant
        pointerArr[i] = header[c];
        // Current atom is highest # in cell, so its placed in header
        header[c] = i;

        cellLocation[i] = c;
    }
}

__device__ 
int3 indexToCell(int idx, float L) {
    int3 arr;
    int targetCellLength = rCutoff;
    int cellsPerDirection = floor(L / targetCellLength);
    float cellLength = L / cellsPerDirection;
    int cellsYZ = cellsPerDirection * cellsPerDirection;
    int cellsXYZ = cellsYZ * cellsPerDirection;
    arr.x = idx / cellsYZ;
    int remainder = idx % cellsYZ;
    arr.y = remainder / cellsPerDirection;
    arr.z = remainder % cellsPerDirection;
    return arr;
}

__global__
void calcForcesPerAtom(float3 *positions, float3 *accelerations, float *netPotential, float L, int *pointerArr, int *header, int *cellLocation) {
    int atomidx = blockIdx.x * numThreadsPerBlock + threadIdx.x;
    if (atomidx >= N) return;
    int targetCellLength = rCutoff;
    int cellsPerDirection = floor(L / targetCellLength);
    float cellLength = L / cellsPerDirection;
    int cellsYZ = cellsPerDirection * cellsPerDirection;
    int cellsXYZ = cellsYZ * cellsPerDirection;

    float3 distArr; // Record distance between atoms
    float localPotential = 0;
    float r2;
    int3 cellArr = indexToCell(cellLocation[atomidx], L);
    int3 mc1, shiftedNeighbor;

    for (mc1.x = cellArr.x - 1; mc1.x < cellArr.x + 2; mc1.x++) {
        for (mc1.y = cellArr.y - 1; mc1.y < cellArr.y + 2; mc1.y++) {
            for (mc1.z = cellArr.z - 1; mc1.z < cellArr.z + 2; mc1.z++) {
                shiftedNeighbor.x = (mc1.x + cellsPerDirection) % cellsPerDirection;
                shiftedNeighbor.y = (mc1.y + cellsPerDirection) % cellsPerDirection;
                shiftedNeighbor.z = (mc1.z + cellsPerDirection) % cellsPerDirection;
                int c1 = shiftedNeighbor.x * cellsYZ + shiftedNeighbor.y + cellsPerDirection + shiftedNeighbor.z;
                int i = atomidx;
                while (i > -1) {
                    int j = header[c1];
                    while (j > -1) {
                        if (i < j) {
                            // Apply boundary conditions
                            distArr = positions[atomidx] - positions[j];
                            distArr.x -= L * round(distArr.x / L);
                            distArr.y -= L * round(distArr.y / L);
                            distArr.z -= L * round(distArr.z / L);
                            r2 = dot(distArr, distArr);
                            if (r2 < rCutoffSquared) {
                                float s2or2 = SIGMA * SIGMA / r2; // Sigma squared over r squared
                                float sor6 = s2or2 * s2or2 * s2or2; // Sigma over r to the sixth
                                float sor12 = sor6 * sor6; // Sigma over r to the twelfth

                                float forceOverR = 24 * EPS_STAR / r2 * (2 * sor12 - sor6);
                                localPotential += 4 * EPS_STAR * (sor12 - sor6);
                                //printf("%i on %i: %f \t r2: %f\n", atomidx, j, forceOverR, r2);
                                atomicAdd(&accelerations[atomidx].x, (forceOverR * distArr.x / MASS));
                                atomicAdd(&accelerations[atomidx].y, (forceOverR * distArr.y / MASS));
                                atomicAdd(&accelerations[atomidx].z, (forceOverR * distArr.z / MASS));
                                atomicAdd(&accelerations[j].x, (-forceOverR * distArr.x / MASS));
                                atomicAdd(&accelerations[j].y, (-forceOverR * distArr.y / MASS));
                                atomicAdd(&accelerations[j].z, (-forceOverR * distArr.z / MASS));
                            }                        
                        }
                        j = pointerArr[j];
                    }
                    i = pointerArr[i];
                }
            }
        }
    }
    netPotential[atomidx] = localPotential;
}


int main() {
    std::cout << "Simulation length: " << L << std::endl;
    std::cout << "Cell length: " << cellLength << std::endl;
    std::cout << "Cells per direction: " << cellsPerDirection << std::endl;
    std::cout << "Num cuda blocks: " << numBlocks << std::endl;
    std::cout << "Threads per block: " << numThreadsPerBlock << std::endl;
    std::cout << "Blocks * Threads per block: " << numBlocks * numThreadsPerBlock << std::endl;
    std::ofstream positionFile("outoldarrays.xyz");
    std::ofstream energyFile("energyoldarrays.dat");
    std::ofstream debug("debugoldarrays.dat");

    // Arrays to hold energy values at each step of the process
    std::vector<float> KE;
    std::vector<float> PE;
    std::vector<float> netE;

    float3 positions[N];
    float3 velocities[N];
    float3 accelerations[N];
    float3 oldAccelerations[N];

    faceCenteredCell(positions);

    std::random_device rd;
    std::default_random_engine generator(3); // (rd())
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (int i = 0; i < N; ++i) { // Randomize velocities
        velocities[i].x = distribution(generator);
        velocities[i].y = distribution(generator);
        velocities[i].z = distribution(generator);
        accelerations[i] = make_float3(0, 0, 0);
        oldAccelerations[i] = make_float3(0, 0, 0);
    }
    float3 *devPos, *devVel, *devAccel, *devOldAccel;
    int *pointerArr, *header, *cellLocation;
    cudaMalloc(&devPos, N * sizeof(float3));
    cudaMemcpy(devPos, positions, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&devVel, N * sizeof(float3));
    cudaMemcpy(devVel, velocities, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&devAccel, N * sizeof(float3));
    cudaMemcpy(devAccel, accelerations, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&devOldAccel, N * sizeof(float3));
    cudaMemcpy(devOldAccel, oldAccelerations, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc(&pointerArr, N * sizeof(int));
    cudaMalloc(&header, cellsXYZ * sizeof(int));
    cudaMalloc(&cellLocation, N * sizeof(int));
    thermostat<<<numBlocks, numThreadsPerBlock>>>(devVel, calcVelocityDotProd(velocities)); // Make velocities more accurate
    cudaDeviceSynchronize();

    float count = .05;
    for (int i = 0; i < numTimeSteps; ++i) { // Main loop handles integration and printing to files
        if (i > count * numTimeSteps) { // Percent progress
            std::cout << count * 100 << "% \n";
            count += .05;
        }

        cudaMemcpy(positions, devPos, N * sizeof(float3), cudaMemcpyDeviceToHost);
        writePositions(positions, positionFile, i);

        /*
        cudaMemcpy(velocities, devVel, N * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(accelerations, devAccel, N * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(oldAccelerations, devOldAccel, N * sizeof(float3), cudaMemcpyDeviceToHost);
        writeToDebugFile(positions, velocities, accelerations, oldAccelerations, debug, i);
        */

        firstStep<<<numBlocks, numThreadsPerBlock>>>(devPos, devVel, devAccel, devOldAccel, L, timeStep); // Update position and write currect accel to old accel
        cudaDeviceSynchronize();

        float *netPotential;
        cudaMallocManaged(&netPotential, N * sizeof(float));
        for (int r = 0; r < N; r++) {
            netPotential[r] = 0;
        }
        sortAtoms<<<1, 1>>>(pointerArr, header, devPos, cellLocation, L);
        cudaDeviceSynchronize();
        int test[N];
        cudaError_t err = cudaMemcpy(test, cellLocation, cellsXYZ * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "bruh" << std::endl;
            exit(20);
        }
        for (int j = 0; j < cellsXYZ; j++) {
            std::cout << header[j] << std::endl;
        }
        exit(1);
        cudaDeviceSynchronize();
        calcForcesPerAtom<<<numBlocks, numThreadsPerBlock>>>(devPos, devAccel, netPotential, L, pointerArr, header, cellLocation); // Update accelerations and return potential of system
        cudaDeviceSynchronize();
        float result = 0;
        for (int j = 0; j < N; j++) {
            result += netPotential[j];
        }
        cudaFree(netPotential);

        float *totalVelSquared;
        cudaMallocManaged(&totalVelSquared, N * sizeof(float));
        thirdStep<<<numBlocks, numThreadsPerBlock>>>(devVel, devAccel, devOldAccel, L, totalVelSquared, timeStep); // Modify velocity a second time based off new forces
        cudaDeviceSynchronize();
        float vel = 0;
        for (int j = 0; j < N; j++) {
            vel += totalVelSquared[j];
        }
        cudaFree(totalVelSquared);

        if (i < numTimeSteps / 2 && i % 5 == 0) { // Apply velocity modifications for first half of sample
            thermostat<<<numBlocks, numThreadsPerBlock>>>(devVel, vel);
            cudaDeviceSynchronize();
        }

        if (i > numTimeSteps / 2) { // Record energies after half of time has passed
            float netKE = .5 * MASS * vel;
            KE.push_back(netKE);
            PE.push_back(result);
            netE.push_back(result + netKE);
            energyFile << "Time: " << i << "\n";
            energyFile << "KE: " << netKE << "\n";
            energyFile << "PE: " << result << "\n";
            energyFile << "Total energy: " << result + netKE << "\n";
            energyFile << "----------------------------------\n";
        }
    }
    
    cudaFree(devPos);
    cudaFree(devVel);
    cudaFree(devAccel);
    cudaFree(devOldAccel);
    cudaFree(pointerArr);
    cudaFree(header);
    cudaFree(cellLocation);

    float avgPE = 0; // Average PE array
    for (float i : PE) {
        avgPE += i;
    }
    avgPE /= PE.size();
    std::cout << "Avg PE: " << avgPE << std::endl;

    float SoLo2 = SIGMA / (L / 2); // Sigma over L over 2
    float Ulrc = (8.0 / 3.0) * M_PI * N * rhostar * EPS_STAR; // Potential sub lrc (long range corrections)
    float temp = 1.0 / 3.0 * std::pow(SoLo2, 9.0);
    float temp1 = std::pow(SoLo2, 3.0);
    Ulrc *= (temp - temp1);
    float PEstar = ((avgPE + Ulrc) / N) / EPS_STAR; // Reduced potential energy

    std::cout << "Reduced potential with long range correction: " << PEstar << std::endl;
    debug << "Avg PE: " << avgPE << std::endl;
    debug << "Reduced potential: " << PEstar << std::endl;

    positionFile.close();
    energyFile.close();
    debug.close();

    return 0;
}
