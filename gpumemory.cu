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

#include <cuda.h>

#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <fstream>
#include <array>

const double Kb = 1.38064582e-23; // J / K
const double Na = 6.022e23; // Atoms per mole

const int numTimeSteps = 500; // Parameters to change for simulation
const double dt_star= .001;

const int N = 32; // Number of atoms in simulation
const double SIGMA = 3.405; // Angstroms
const double EPSILON = 1.6540e-21; // Joules
const double EPS_STAR = EPSILON / Kb; // ~ 119.8 K

const double rhostar = .45; // Dimensionless density of gas
const double rho = rhostar / (SIGMA * SIGMA * SIGMA); // Density of gas
const double L = std::cbrt(N / rho); // Unit cell length
const double rCutoff = SIGMA * 2.5; // Forces are negligible past this distance, not worth calculating
const double rCutoffSquared = rCutoff * rCutoff;
const double tStar = 1.24; // Reduced units of temperature
const double TARGET_TEMP = tStar * EPS_STAR;
// 39.9 is mass of argon in amu, 10 is a conversion between the missing units :)
const double MASS = 39.9 * 10 / Na / Kb; // Kelvin * ps^2 / A^2
const double timeStep = dt_star * std::sqrt(MASS * SIGMA * SIGMA / EPS_STAR); // Convert time step to picoseconds


__host__
void faceCenteredCell(float3 *pos) {
    // Each face centered unit cell has four atoms
    // Method creates a cubic arrangement of face centered unit cells

    double n = std::cbrt(N / 4.0); // Number of unit cells in each direction
    double dr = L / n; // Distance between two corners in a unit cell
    double dro2 = dr / 2.0; // dr over 2

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

__global__ 
void thermostat(float3 *velocities) {
    double instantTemp = 0;
    for (int i = 0; i < N; i++) { // Add kinetic energy of each molecule to the temperature
        instantTemp += MASS * velocities[i].x * velocities[i].x + velocities[i].y * velocities[i].y + velocities[i].z * velocities[i].z;
    }
    instantTemp /= (3 * N - 3);
    double tempScalar = std::sqrt(TARGET_TEMP / instantTemp);
    for (int i = 0; i < N; i++) {
        velocities[i].x *= tempScalar; // V = V * lambda
        velocities[i].y *= tempScalar; // V = V * lambda
        velocities[i].z *= tempScalar; // V = V * lambda
    }
}

__global__ 
void firstStep(float3 *positions, float3 *velocities, float3 *accelerations, float3 *oldAccelerations, double L, double timeStep) {
    for (int k = 0; k < N; ++k) { // Update positions
        positions[k].x += velocities[k].x * timeStep + .5 * accelerations[k].x * timeStep * timeStep;
        positions[k].y += velocities[k].y * timeStep + .5 * accelerations[k].y * timeStep * timeStep;
        positions[k].z += velocities[k].z * timeStep + .5 * accelerations[k].z * timeStep * timeStep;
        positions[k].x += -L * std::floor(positions[k].x / L); // Keep atom inside box
        positions[k].y += -L * std::floor(positions[k].y / L); // Keep atom inside box
        positions[k].z += -L * std::floor(positions[k].z / L); // Keep atom inside box
        oldAccelerations[k].x = accelerations[k].x;
        oldAccelerations[k].y = accelerations[k].y;
        oldAccelerations[k].z = accelerations[k].z;
    }
}

__global__ 
void thirdStep(float3 *velocities, float3 *accelerations, float3 *oldAccelerations, double L, float *totalVelSquared, double timeStep) {
    for (int k = 0; k < N; ++k) { // Update velocities
        velocities[k].x += .5 * (accelerations[k].x + oldAccelerations[k].x) * timeStep;
        velocities[k].y += .5 * (accelerations[k].y + oldAccelerations[k].y) * timeStep;
        velocities[k].z += .5 * (accelerations[k].z + oldAccelerations[k].z) * timeStep;
        totalVelSquared[k] += velocities[k].x * velocities[k].x;
        totalVelSquared[k] += velocities[k].y * velocities[k].y;
        totalVelSquared[k] += velocities[k].z * velocities[k].x;
    }
}

__device__
void calcForcesPerAtom(float3 *positions, float3 *accelerations, int atomidx, float *netPotential, double L) {
    float distArr[3]; // Record distance between atoms
    float localPotential = 0;
    float r2;

    for (int j = atomidx + 1; j < N; j++) {
        // Apply boundary conditions
        distArr[0] = positions[atomidx].x - positions[j].x;
        distArr[1] = positions[atomidx].y - positions[j].y;
        distArr[2] = positions[atomidx].z - positions[j].z;
        distArr[0] -= L * std::round(distArr[0] / L);
        distArr[1] -= L * std::round(distArr[1] / L);
        distArr[2] -= L * std::round(distArr[2] / L);
        r2 = distArr[0] * distArr[0] + distArr[1] * distArr[1] + distArr[2] * distArr[2]; // Dot product b/t atoms
        if (r2 < rCutoffSquared) {
            float s2or2 = SIGMA * SIGMA / r2; // Sigma squared over r squared
            float sor6 = s2or2 * s2or2 * s2or2; // Sigma over r to the sixth
            float sor12 = sor6 * sor6; // Sigma over r to the twelfth

            float forceOverR = 24 * EPS_STAR / r2 * (2 * sor12 - sor6);
            localPotential += 4 * EPS_STAR * (sor12 - sor6);
            accelerations[atomidx].x += (forceOverR * distArr[0] / MASS);
            accelerations[atomidx].y += (forceOverR * distArr[1] / MASS);
            accelerations[atomidx].z += (forceOverR * distArr[2] / MASS);
            accelerations[j].x -= (forceOverR * distArr[0] / MASS);
            accelerations[j].y -= (forceOverR * distArr[1] / MASS);
            accelerations[j].z -= (forceOverR * distArr[2] / MASS);
        }
    }
    netPotential[atomidx] = localPotential;
}

__global__
void calcForces(float3 *positions, float3 *accelerations, float *netPotential, double L) { // Cell pairs method to calculate forces
    for (int j = 0; j < N; j++) { // Set all accelerations equal to zero
        for (int i = 0; i < 3; ++i) {
            accelerations[j] = make_float3(0, 0, 0);
        }
    }
    for (int c = 0; c < N - 1; c++) {
         calcForcesPerAtom(positions, accelerations, c, netPotential, L);
    }
}


int main() {

    std::cout << "Cell length: " << L << std::endl;
    std::ofstream positionFile("outmultithreadeverypairs.xyz");
    std::ofstream energyFile("energymultithreadeverypair.dat");
    std::ofstream debug("debugmultithreadeverypairs.dat");

    // Arrays to hold energy values at each step of the process
    std::vector<float> KE;
    std::vector<float> PE;
    std::vector<float> netE;

    float3 positions[N];
    float3 velocities[N];
    float3 accelerations[N];
    float3 oldAccelerations[N];

    std::cout << "test1" << std::endl;
    faceCenteredCell(positions);

    std::random_device rd;
    std::default_random_engine generator(3); // (rd())
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (int i = 0; i < N; ++i) { // Randomize velocities
        velocities[i].x = distribution(generator);
        velocities[i].y = distribution(generator);
        velocities[i].z = distribution(generator);
    }
    std::cout << "test2" << std::endl;
    float3 *devPos, *devVel, *devAccel, *devOldAccel;
    cudaMallocManaged(&devPos, N * sizeof(float3));
    cudaMallocManaged(&devVel, N * sizeof(float3));
    cudaMallocManaged(&devAccel, N * sizeof(float3));
    cudaMallocManaged(&devOldAccel, N * sizeof(float3));
    cudaMemcpy(devPos, positions, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(devAccel, accelerations, N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(devOldAccel, oldAccelerations, N * sizeof(float3), cudaMemcpyHostToDevice);
    std::cout << "test3" << std::endl;
    thermostat<<<1, 1>>>(velocities); // Make velocities more accurate

    double count = .01;
    for (int i = 0; i < numTimeSteps; ++i) { // Main loop handles integration and printing to files

        if (i > count * numTimeSteps) { // Percent progress
            std::cout << count * 100 << "% \n";
            count += .01;
        }

    std::cout << "test4" << std::endl;
        cudaMemcpy(positions, devPos, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        writePositions(positions, positionFile, i);
    std::cout << "test5" << std::endl;

        firstStep<<<1, 1>>>(devPos, devVel, devAccel, devOldAccel, L, timeStep); // Update position and write currect accel to old accel
        cudaDeviceSynchronize();
    std::cout << "test6" << std::endl;

        float *netPotential;
        cudaMallocManaged(&netPotential, N * sizeof(float));
    std::cout << "test7" << std::endl;
        calcForces<<<1, 1>>>(devPos, devAccel, netPotential, L); // Update accelerations and return potential of system
    std::cout << "test8" << std::endl;
        cudaDeviceSynchronize();
        std::cout << "test9" << std::endl;
        float result = 0;
        for (int j = 0; j < N; j++) {
            std::cout << "test10\n";
            result += netPotential[j];
        }
        cudaFree(netPotential);

        float *totalVelSquared;
        cudaMallocManaged(&totalVelSquared, N * sizeof(float));
        thirdStep<<<1, 1>>>(devVel, devAccel, devOldAccel, L, totalVelSquared, timeStep); // Modify velocity a second time based off new forces
        cudaDeviceSynchronize();
        std::cout << "test11" << std::endl;
        double vel = 0;
        for (int j = 0; j < N; j++) {
            vel += j[totalVelSquared];
        }
        cudaFree(totalVelSquared);

        if (i < numTimeSteps / 2 && i % 5 == 0) { // Apply velocity modifications for first half of sample
            thermostat<<<1, 1>>>(devVel);
        }

        if (i > -1) { // Record energies after half of time has passed
            energyFile << "Time: " << i << "\n";
            double netKE = .5 * MASS * vel;
            KE.push_back(netKE);
            PE.push_back(result);
            netE.push_back(result + netKE);
            /*
            energyFile << "KE: " << netKE << "\n";
            energyFile << "PE: " << netPotential << "\n";
            energyFile << "Total energy: " << netPotential + netKE << "\n";
            energyFile << "----------------------------------\n";
            */
        }
    }
    
    cudaFree(devPos);
    cudaFree(devVel);
    cudaFree(devAccel);
    cudaFree(devOldAccel);

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
    debug << "Avg PE: " << avgPE << std::endl;
    debug << "Reduced potential: " << PEstar << std::endl;

    positionFile.close();
    energyFile.close();
    debug.close();

    return 0;
}
