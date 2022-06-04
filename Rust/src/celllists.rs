use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;
use rand::Rng;

const KB: f64 = 1.38e-23;
const NA: f64 = 6.022e23;

const NUM_TIME_STEPS: i32 = 10000;
const DT_STAR: f64 = 0.001;

const N: i32 = 256;
const SIGMA: f64 = 3.405;
const EPSILON: f64 = 1.654e-21;
const EPS_STAR: f64 = EPSILON / KB;

const RHOSTAR: f64 = 0.6;
const RHO: f64 = RHOSTAR / (SIGMA * SIGMA * SIGMA);
const R_CUTOFF: f64 = SIGMA * 2.5;
const T_STAR: f64 = 1.24;
const TARGET_TEMP: f64 = T_STAR * EPS_STAR;
const MASS: f64 = 39.9 * 10. / NA / KB;


macro_rules! timeStep {
    () => { 
        (DT_STAR * f64::sqrt(MASS * SIGMA * SIGMA / EPS_STAR))
    };
}

macro_rules! L {
    () => {
        (f64::cbrt(N as f64 / RHO))
    };
}

macro_rules! rCutoffSquared {
    () => {
        (R_CUTOFF * R_CUTOFF)
    };
}

macro_rules! cellsxyz {
    () => {
        (cells_per_side * cells_per_side * cells_per_side)
    };
}

struct Atom {
    positions: [f64; 3],
    velocities: [f64; 3],
    accelerations: [f64; 3],
    old_accelerations: [f64; 3],
}

fn main() {
    let mut position_file = File::create("out.xyz").expect("Unable to open");
    
    let mut ke = Vec::new();
    let mut pe = Vec::new();
    let mut net_e = Vec::new();

    let mut atoms: Vec<Atom> = face_centered_cell();

    let mut rng = rand::thread_rng();
    for atom in atoms.iter_mut() {
        for j in 0..3 {
            atom.velocities[j] = rng.gen_range(-1.0, 1.0);
        }
    } 

    thermostat(&mut atoms);

    let mut total_vel_squared: f64;
    let mut net_potential: f64;

    println!("Starting program");
    let mut count: f64 = 0.01;
    for i in 0..NUM_TIME_STEPS {
        if i as f64 > count * NUM_TIME_STEPS as f64 {
            println!("{}", count * 100.);
            count += 0.01;
        }

        // position_file.write("{} \n \n", N);
        write!(&mut position_file, "{} \n \n", N).expect("TODO: panic message");
        for atom in atoms.iter_mut() {
            write!(&mut position_file, "A {} {} {} \n", atom.positions[0], atom.positions[1], atom.positions[2]).expect("TODO: panic message");
            // position_file.write_all(b"A {} {} {} \n", atom.positions[0], atom.positions[1], atom.positions[2]);
        }

        for atom in atoms.iter_mut() {
            for k in 0..3 {
                atom.positions[k] += atom.velocities[k] * timeStep!() + 0.5 * atom.accelerations[k] * timeStep!() * timeStep!();
                atom.positions[k] += -1. * L!() * f64::floor(atom.positions[k] / L!());
            }
        }

        net_potential = calc_forces(&mut atoms);

        total_vel_squared = 0.;

        for atom in atoms.iter_mut() {
            for k in 0..3 {
                atom.velocities[k] += 0.5 * (atom.accelerations[k] + atom.old_accelerations[k]) * timeStep!();
                total_vel_squared += atom.velocities[k] * atom.velocities[k];
            }
        }

        if i < NUM_TIME_STEPS / 2 && i % 5 == 0 {
            thermostat(&mut atoms);
        }

        if i > NUM_TIME_STEPS / 2 {
            let net_ke = 0.5 * MASS * total_vel_squared;

            ke.push(net_ke);
            pe.push(net_potential);
            net_e.push(net_ke + net_potential);
        }
    }

    let mut avg: f64 = 0.;
    for i in &pe {
        avg += i;
    }
    avg /= pe.len() as f64;

    let SoLo2 = SIGMA / (L!() / 2.);
    let mut Ulrc = (8.0 / 3.0) * PI as f64 * N as f64 * RHOSTAR * EPS_STAR;
    let temp = 1.0 / 3.0 * f64::powf(SoLo2, 9.);
    let temp1 = f64::powf(SoLo2, 3.);
    Ulrc *= temp - temp1;
    let pestar = ((avg + Ulrc) / N as f64) / EPS_STAR;
    println!("Reduced potential: {}", pestar);

}

fn dot(x: f64, y: f64, z: f64) -> f64 { x * x + y * y + z * z }

fn thermostat(atoms: &mut Vec<Atom>) {
    let mut instant_temp: f64 = 0.;
    for atom in atoms.iter_mut() {
        instant_temp += MASS * dot(atom.velocities[0], atom.velocities[1], atom.velocities[2]);
    }
    instant_temp /= (3 * N - 3) as f64;
    let temp_scalar = f64::sqrt(TARGET_TEMP / instant_temp);
    for atom in atoms.iter_mut() {
        for j in 0..3 {
            atom.velocities[j] *= temp_scalar;
        }
    } 
}

fn calc_forces(atoms: &mut Vec<Atom>) -> f64 {
    let mut net_potential: f64 = 0.;
    const target_cell_length: f64 = R_CUTOFF;
    const cells_per_side: i32 = f64::floor(L as i32 / targetCellLength);
    const cell_length: f64 = L / cells_per_side as f64;
    let cellsxy = cells_per_side * cells_per_side;
    let cellsxyz = cellsxy * cells_per_side;
    let mut pointer_arr: [f64, N] = [0.; N];
    let mut header: [f64, cellsxyz!()];
    let mut mc: [f64, 3];
    let mut mc1: [f64, 3];
    let mut neighbor: [i32, 3];
    let mut distArr: [f64, 3];


    for atom in atoms.iter_mut() {
        for j in 0..3 {
            atom.accelerations[j] = 0.;
        }
    }

    for c in &header {
        c = -1;
    }

    for i in 0..N {
        for j in 0..3 {
            mc[j as usize] = atoms[i as usize].positions[j as usize] / cell_length;
        }
        c = mc[0] * cellsyz + mc[1] * cells_per_side + mc[2];
        pointer_arr[i as usize] = header[c as usize];
        header[c as usize] = i;
    }



    let mut dist_arr: [f64; 3] = [0., 0., 0.];

    for i in 0..(N-1){
        for j in i..N {
            if i != j {
                for k in 0..3 {
                    dist_arr[k] = atoms[i as usize].positions[k] - atoms[j as usize].positions[k];
                    dist_arr[k] -= L!() * f64::round(dist_arr[k] / L!());
                }

                let r2: f64 = dot(dist_arr[0], dist_arr[1], dist_arr[2]);

                if r2 <= rCutoffSquared!() {
                    let s2or2 = SIGMA * SIGMA / r2;
                    let sor6 = s2or2 * s2or2 * s2or2;
                    let sor12 = sor6 * sor6;

                    let force_over_r = 24. * EPS_STAR / r2 * (2. * sor12 - sor6);
                    net_potential += 4. * EPS_STAR * (sor12 - sor6);

                    for k in 0..3 {
                        atoms[i as usize].accelerations[k] += force_over_r * dist_arr[k] / MASS;
                        atoms[j as usize].accelerations[k] -= force_over_r * dist_arr[k] / MASS;
                    }
                }
            }
        }
    }
    return net_potential;
}

fn simple_cubic_cell() -> Vec<Atom> {
    let n = f64::cbrt(N as f64) as i32;
    let mut atoms: Vec<Atom> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let atom = Atom{positions: [i as f64 * SIGMA, j as f64 * SIGMA, k as f64 * SIGMA], velocities: [0., 0., 0.], 
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]};
                atoms.push(atom); 
            }
        }
    }
    return atoms;
}

fn face_centered_cell() -> Vec<Atom> {
    let n: i32 = f64::cbrt(N as f64 / 4.) as i32;
    let dr: f64 = L!() / n as f64;
    let dro2: f64 = dr / 2.0;

    let mut atoms: Vec<Atom> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                atoms.push(Atom{positions:[i as f64 * dr, j as f64 * dr, k as f64 * dr], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
                atoms.push(Atom{positions:[i as f64 * dr + dro2, j as f64 * dr + dro2, k as f64 * dr], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
                atoms.push(Atom{positions:[i as f64 * dr + dro2, j as f64 * dr, k as f64 * dr + dro2], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
                atoms.push(Atom{positions:[i as f64 * dr, j as f64 * dr + dro2, k as f64 * dr + dro2], velocities: [0., 0., 0.],
                    accelerations: [0., 0., 0.], old_accelerations: [0., 0., 0.]});
            }
        }
    }
    return atoms;
}
