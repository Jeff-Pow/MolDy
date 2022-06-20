package org.drpowell.moldy;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.lang.Math;

public class MolDy {

    public static Random random = new Random(6345789);

    public static class Atom {
        public double positions[] = {0,0,0};
        public double accelerations[] = {0,0,0};
        public double oldAccelerations[] = {0,0,0};
        public double velocities[] = new double[3];
        public Atom(double x, double y, double z) {
            positions[0] = x;
            positions[1] = y;
            positions[2] = z;
            for (int i = 0; i < 3; i++) {
                velocities[i] = random.nextGaussian();
            }
        }
    }

    public static final double Kb = 1.38064582E-23; // J/K std::pow(10, -23); // J / K
    public static final double Na = 6.022E23; // Atoms per mole

    public static final int numTimeSteps = 5000; // Parameters to change for simulation
    public static final double dt_star = .001;

    public static final int N = 4000; // Number of atoms in simulation
    public static final double SIGMA = 3.405; // Angstroms
    public static final double EPSILON = 1.6540E-21; // Joules
    public static final double EPS_STAR = EPSILON / Kb; // ~ 119.8 K

    public static final double rhostar = .6; // Dimensionless density of gas
    public static final double rho = rhostar / (SIGMA*SIGMA*SIGMA); // Density of gas
    public static final double L = Math.cbrt(N / rho); // Unit cell length
    public static final double rCutoff = SIGMA * 2.5; // Forces are negligible past this distance, not worth calculating
    public static final double rCutoffSquared = rCutoff * rCutoff;
    public static final double tStar = 1.24; // Reduced units of temperature
    public static final double TARGET_TEMP = tStar * EPS_STAR;
    // 39.9 is mass of argon in amu, 10 is a conversion between the missing units :)
    public static final double MASS = 39.9 * 10 / Na / Kb; // Kelvin * ps^2 / A^2
    public static final double timeStep = dt_star * Math.sqrt(MASS * SIGMA * SIGMA / EPS_STAR); // Convert time step to picoseconds

    public static final double targetCellLength = rCutoff;
    public static final int numCellsPerDirection = (int) (L / targetCellLength);
    public static final double cellLength = L / numCellsPerDirection; // Side length of each cell
    public static final int numCellsYZ = numCellsPerDirection * numCellsPerDirection; // Number of cells in one plane
    public static final int numCellsXYZ = numCellsYZ * numCellsPerDirection; // Number of cells in the simulation

    public static ArrayList<Atom> faceCenteredCell(int totalAtoms, double unitCellLength) {
        double n = Math.cbrt(totalAtoms / 4); // Number of unit cells in each direction
        ArrayList<Atom> atoms = new ArrayList<Atom>(totalAtoms);
        double dr = unitCellLength / n; // distance between 2 corners in unit cell
        double dro2 = dr / 2;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    atoms.add(new Atom(i * dr, j * dr, k * dr));
                    atoms.add(new Atom(i * dr + dro2, j * dr + dro2, k * dr));
                    atoms.add(new Atom(i * dr + dro2, j * dr, k * dr + dro2));
                    atoms.add(new Atom(i * dr, j * dr + dro2, k * dr + dro2));
                }
            }
        }
        return atoms;
    }

    public static double dot(double [] a, double [] b) {
        // if (a.length != b.length) throw new IllegalArgumentException("Array lengths differ: " + a.length + " != " + b.length);
        double dot = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    public static void thermostat(ArrayList<Atom> atoms) {
        double instantTemp = 0;
        for (Atom atom : atoms) { // Add kinetic energy of each molecule to the temperature
            double [] velocities = atom.velocities;
            instantTemp += MASS * dot(velocities, velocities);
        }
        instantTemp /= (3 * N - 3);
        double tempScalar = Math.sqrt(TARGET_TEMP / instantTemp);
        for (Atom atom : atoms) {
            for (int i = 0; i < 3; ++i) {
                atom.velocities[i] *= tempScalar; // V = V * lambda
            }
        }
    }

    public static final int cellAddressToIndex(int address[]) {
        return address[0] * numCellsYZ + address[1] * numCellsPerDirection + address[2];
    }

    public static final int [] cellIndexToAddress(int index) {
        int [] response = new int[3];
        response[0] = index / numCellsYZ;
        int remainder = index % numCellsYZ;
        response[1] = remainder / numCellsPerDirection;
        response[2] = remainder % numCellsPerDirection;
        return response;
    }

    private static double calcForcesOnCell(int cellIndex, ArrayList<Atom> atoms, int[] header, int[] nextAtom) {
        double localPotential = 0;
        int [] cellAddress = cellIndexToAddress(cellIndex);
        int [] neighborAddress = new int[3];
        int [] shiftedNeighbor = new int[3];
        double distanceVector[] = new double[3];

        int pairsAffected = 0;

        // Scan neighbor cells including the one currently active
        for (neighborAddress[0] = cellAddress[0] - 1; neighborAddress[0] < cellAddress[0] + 2; neighborAddress[0]++) {
            for (neighborAddress[1] = cellAddress[1] - 1; neighborAddress[1] < cellAddress[1] + 2; neighborAddress[1]++) {
                for (neighborAddress[2] = cellAddress[2] - 1; neighborAddress[2] < cellAddress[2] + 2; neighborAddress[2]++) {

                    for (int k = 0; k < 3; k++) { // Boundary conditions
                        shiftedNeighbor[k] = (neighborAddress[k] + numCellsPerDirection) % numCellsPerDirection;
                    }
                    // Scalar index of neighboring cell
                    int neighborIndex = cellAddressToIndex(shiftedNeighbor);

                    int i = header[cellIndex]; // Find the highest numbered atom in each cell
                    while (i > -1) {
                        Atom atomi = atoms.get(i);
                        int j = header[neighborIndex]; // Scan atom with the largest index in neighboring cell c1
                        while (j > -1) {
                            if (i < j) { // Don't double count atoms (if i > j its already been counted)
                                Atom atomj = atoms.get(j);
                                for (int k = 0; k < 3; k++) {
                                    // Apply boundary conditions
                                    distanceVector[k] = atomi.positions[k] - atomj.positions[k];
                                    distanceVector[k] -= L * Math.round(distanceVector[k] / L);
                                }
                                double r2 = dot(distanceVector, distanceVector); // Dot of distance vector between the two atoms
                                if (r2 < rCutoffSquared) {
                                    double s2or2 = SIGMA * SIGMA / r2; // Sigma squared over r squared
                                    double sor6 = s2or2 * s2or2 * s2or2; // Sigma over r to the sixth
                                    double sor12 = sor6 * sor6; // Sigma over r to the twelfth

                                    double forceOverR = 24 * EPS_STAR / r2 * (2 * sor12 - sor6);
                                    localPotential += 4 * EPS_STAR * (sor12 - sor6);
                                    // debug << i << " on " << j << ": " << forceOverR << "\n";

                                    for (int k = 0; k < 3; k++) {
                                        atomi.accelerations[k] += (forceOverR * distanceVector[k] / MASS);
                                        atomj.accelerations[k] -= (forceOverR * distanceVector[k] / MASS);
                                    }
                                    pairsAffected++;
                                }
                            }
                            j = nextAtom[j];
                        }
                        i = nextAtom[i];
                    }
                }
            }
        }
        //System.err.println("For cell index " + cellIndex + " there were " + pairsAffected + " affected pairs");
        return localPotential;
    }

    public static double calcForces(ArrayList<Atom> atoms) {
        int cellIndex;
        int [] cellAddress = new int[3];
        double netPotential = 0;

        // header and nextAtom together make a linked list (possibly more performant than LinkedList?)
        int [] header = new int[numCellsXYZ];
        int [] nextAtom = new int[N];
        for (cellIndex = 0; cellIndex < numCellsXYZ; cellIndex++) {
            header[cellIndex] = -1;
        }

        int atomNum = 0;
        for (Atom a: atoms) {
            for (int i = 0; i < 3; i++) {
                cellAddress[i] = (int) (a.positions[i] / cellLength);
            }
            cellIndex = cellAddressToIndex(cellAddress);
            nextAtom[atomNum] = header[cellIndex];
            header[cellIndex] = atomNum;
            atomNum++;
        }

        for (cellIndex = 0; cellIndex < numCellsXYZ ; cellIndex++) {
            netPotential += calcForcesOnCell(cellIndex, atoms, header, nextAtom);
        }
        return netPotential;
    }

    public static void writePositions(PrintStream out, ArrayList<Atom> atoms, int timeStep) {
        out.println(atoms.size() + "\nTime: " + timeStep);
        for (Atom a : atoms) {
            out.println("A " + a.positions[0] + " " + a.positions[1] + " " + a.positions[2]);
        }
    }

    public static void main(String[] args) throws FileNotFoundException {
        long startTime = System.currentTimeMillis();
        PrintStream outputStream = new PrintStream(new BufferedOutputStream(new FileOutputStream(new File("out.xyz"))), false);
        ArrayList<Double> KE = new ArrayList<>(numTimeSteps / 2);
        ArrayList<Double> PE = new ArrayList<>(numTimeSteps / 2);
        ArrayList<Double> netE = new ArrayList<>(numTimeSteps / 2);

        ArrayList<Atom> atoms = faceCenteredCell(N, L);
        thermostat(atoms);
        for (int t = 0; t < numTimeSteps; t++) {
            writePositions(outputStream, atoms, t);
            outputStream.flush();

            // update positions
            for (Atom atom: atoms) {
                for (int i = 0; i < 3; i++) {
                    atom.positions[i] += atom.velocities[i] * timeStep + .5 * atom.accelerations[i] * timeStep * timeStep;
                    atom.positions[i] -= L * Math.floor(atom.positions[i] / L); // keep atom inside box
                    atom.oldAccelerations[i] = atom.accelerations[i];
                }
            }
            double netPotential = 0;
            netPotential += calcForces(atoms);

            double totalVelSquared = 0;
            for (Atom atom: atoms) {
                for (int i = 0; i < 3; i++) {
                    atom.velocities[i] += (atom.accelerations[i] + atom.oldAccelerations[i]) * timeStep / 2;
                    totalVelSquared += atom.velocities[i] * atom.velocities[i];
                    atom.oldAccelerations[i] = atom.accelerations[i];
                    atom.accelerations[i] = 0;
                }
            }

            if ((t * 100) % numTimeSteps == 0) {
                System.err.println((t*100)/numTimeSteps);
            }

            if (t < numTimeSteps / 2 && t % 5 == 0 && t != 0) {
                // apply velocity modifications for 20% of first half of timesteps
                thermostat(atoms);
            }

            if (t > numTimeSteps / 2) {
                // record energies after half of time has passed
                double netKE = MASS * totalVelSquared / 2;
                KE.add(netKE);
                PE.add(netPotential);
                netE.add(netPotential + netKE);
            }

        }

        double avgPE = 0; // Average PE array
        for (int i = 0; i < PE.size(); i++) {
            avgPE += PE.get(i);
        }
        avgPE /= PE.size();

        double SoLo2 = SIGMA / (L / 2); // Sigma over L over 2
        double Ulrc = (8.0 / 3.0) * Math.PI * N * rhostar * EPS_STAR; // Potential sub lrc (long range corrections)
        double temp = 1.0 / 3.0 * Math.pow(SoLo2, 9.0);
        double temp1 = Math.pow(SoLo2, 3.0);
        Ulrc *= (temp - temp1);
        double PEstar = ((avgPE + Ulrc) / N) / EPS_STAR;
        System.out.println("Reduced potential with long range corrections: " + PEstar);
        System.out.println("Total time: " + ((System.currentTimeMillis() - startTime) / 1000.0) + " seconds");
    }
}
