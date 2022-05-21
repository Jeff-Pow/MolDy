#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define kB 1.38064852e-23						//Bolztmann's constant (J/K)
#define NA 6.02214e23							//Avogadro's constant

#define N_STEPS 10000							//Number of simulation steps
#define BIN 100								//Binning number for radial distribution histogram
#define DIMENSIONS 3							//Number of coordinates (x,y,z)
#define N 32								//Number of particles
#define s 3.405								//Sigma of gas (in Angstroms)
#define p_star 0.7							//Density of gas in (dimentionless)
#define T_star 1.1							//Temperature of gas (dimensionless)
#define dt_star 0.001							//Time step (dimensionless)
#define MASS 39.948							//Mass of gas (amu)
#define epsilon 1.65401693e-21						//Epsilon of gas (J)
#define THERMO (N_STEPS / 2.0)						//Thermostat for specified portion of N_STEPS (i.e. first quarter, half, etc.)
#define APPLY_T 5							//Apply thermostat every X number of steps
static char *atom1 = "Ar";						//Atom type

/* THE ABOVE PARAMETERS CAN BE ADJUSTED TO CHANGE THE PROPERTIES OF THE GAS IN THE MACROS BELOW							*
 * Lennard-Jones parameters (epsilon and sigma) taken from: https://www.sciencedirect.com/science/article/pii/002199917590042X?via%3Dihub 	*/

#define p (p_star / pow(s,3.0))						//Density (A^-3)
#define L pow((N/p),1.0/3.0)						//Box length (A)
#define e (epsilon / kB)						//Energy of gas (K)
#define m (MASS * 10.0 / NA / kB)					//Conversion of amu to K*ps^2/A^2
#define T (T_star * e)							//Conversion of temperature to K
#define dt (dt_star * sqrt((m*pow(s,2.0)) / e))				//Conversion of time step to ps

static double r[N][DIMENSIONS];
static double v[N][DIMENSIONS];
static double a[N][DIMENSIONS];

//New array declaration for multicomponent system for sigma and epsilon parameters
//static double sigma[N], epsilon[N];

void crystalLattice();
void wrapToBox();
void initializeVelocities();
double calculateAcceleration();
void velocityVerlet();
double potentialEnergy();
double kineticEnergy();
double generateGaussian();
double meanSquaredVelocity();
void thermostat();
void radialDist(FILE *fp);
void MSD(FILE *fp);
//void VACF(FILE *fp);

int main()
{
	int i,j,k,n;
	int progress;
	double Temp,Press,Pavg,Tavg,V,PE,KE,sqPE,sqKE,PEavg,KEavg,mv,ulrc,plrc,cvPE,cvKE,vSum,v2;

	clock_t start,end;
	double sim_time;

        FILE *ftraj, *fvel, *fener, *fmv, *ftemp, *fpress;
        ftraj = fopen("traj.xyz","w");
	fvel = fopen("velocities.xyz","w");
        fener = fopen("energy.dat","w");
        fmv = fopen("momentum.dat","w");
        ftemp = fopen("temp.dat","w");
        fpress = fopen("pressure.dat","w");

	start = clock();

	crystalLattice();
	calculateAcceleration();
	v2 = meanSquaredVelocity();

	Pavg = 0;
	Tavg = 0;
	PEavg = 0;
	KEavg = 0;
	sqPE = 0;
	sqKE = 0;
	vSum = 0.0;

	progress = floor(N_STEPS / 10);

	for(n=0;n<=N_STEPS;n++)
	{
		wrapToBox();

		if(n == progress)
			printf("[ 10 |");
		else if(n == 2*progress)
			printf(" 20 |");
		else if(n == 3*progress)
			printf(" 30 |");
		else if(n == 4*progress)
			printf(" 40 |");
		else if(n == 5*progress)
			printf(" 50 |");
		else if(n == 6*progress)
			printf(" 60 |");
		else if(n == 7*progress)
			printf(" 70 |");
		else if(n == 8*progress)
			printf(" 80 |");
		else if(n == 9*progress)
			printf(" 90 |");
		else if(n == 10*progress)
			printf(" 100 ]\n");
		fflush(stdout);

		V = calculateAcceleration();

		//Apply thermostat
		if(n != 0 && n % APPLY_T == 0 && n < THERMO)
			thermostat();

		//Momentum at each step
		for(i=0;i<N;i++)
			for(j=0;j<3;j++)
				vSum += v[i][j] / N;
		mv = m * vSum;
	       	fprintf(fmv,"%d\t %lf\n",n,mv);

		//Write atom position to trajectory file
	        for(i=0;i<1;i++)
                {
                        fprintf(ftraj,"%d\n\n",N);
                        for(j=0;j<N;j++)
                        {
				if(j % 5 == 0)
	                                fprintf(ftraj,"%s\t",atom1);
				else
					fprintf(ftraj,"%s\t",atom1);

                                for(k=0;k<3;k++)
                                        fprintf(ftraj,"%lf\t",r[j][k]);
                                fprintf(ftraj,"\n");
                        }
                }
		
		velocityVerlet();

		//Write atom velocities to file
                for(i=0;i<1;i++)
                {
                        fprintf(fvel,"%d\n\n",N);
                        for(j=0;j<N;j++)
                        {
				if(j % 5 == 0)
	                                fprintf(fvel,"%s\t",atom1);
				else
					fprintf(fvel,"%s\t",atom1);

                                for(k=0;k<3;k++)
                                        fprintf(fvel,"%lf\t",v[j][k]);
                                fprintf(fvel,"\n");
                        }
                }

		//Collect simulation parameters after thermostat switches off
		if(n >= THERMO)
		{
			PE = potentialEnergy();
			KE = kineticEnergy();
			fprintf(fener,"%lf\t %lf\t %lf\n",PE,KE,PE+KE);
			sqPE += PE*PE;
			sqKE += KE*KE;
			PEavg += PE;
                        KEavg += KE;

			//Temperature from kinetic theory of gases (kB = 1 in reduced units)
			Temp = (2.0/3.0) * KE;
			fprintf(ftemp,"%d\t %lf\n",n,Temp);
			Tavg += Temp;

			//Pressure from virial theorem (kB = 1 in reduced units)
			Press = (p*s*s*s)/N/3.0 * (v2 + V/e);
			fprintf(fpress,"%d\t %lf\n",n,Press);
			Pavg += Press;
		}
	}
	fclose(ftraj);
	fclose(fvel);

        printf("*****************************************************************************\n");
        printf("Calculating Radial Distribution...\n");
//        radialDist(ftraj);
        printf("Done!\n");

        printf("*****************************************************************************\n");
        printf("Calculating Mean Squared Displacement...\n");
//        MSD(ftraj);
        printf("Done!\n");
/*
	printf("*****************************************************************************\n");
        printf("Calculating Velocity Autocorrelation...\n");
        VACF(fvel);
        printf("Done!\n");
*/
	sqPE /= (N_STEPS - THERMO);
	sqKE /= (N_STEPS - THERMO);
	PEavg /= (N_STEPS - THERMO);
	KEavg /= (N_STEPS - THERMO);
	Tavg /= (N_STEPS - THERMO);
	Pavg /= (N_STEPS - THERMO);

	//Specific heat from fluctuation in average potential/kinetic energy and average temp
	cvPE = (3.0/2.0) * pow(1.0 - 2*N*(sqPE-pow(PEavg,2.0))/3.0/pow(Tavg,2.0),-1.0);
	cvKE = (3.0/2.0) * pow(1.0 - 2*N*(sqKE-pow(KEavg,2.0))/3.0/pow(Tavg,2.0),-1.0);

	//Long-range correction to pressure (from Comp. Sim. of Liquids - Allen/Tildesly, pg. 65)
	plrc = (16.0/3.0) * M_PI * pow(p_star,2.0) * ((2.0/3.0) * pow(s/L/2.0,9.0) - pow(s/L/2.0,3.0));

	//Long-range correction to energy (from Comp. Sim. of Liquids - Allen/Tildesly, pg. 65)
	ulrc = (8.0/3.0) * M_PI * N * p_star * e * ((1.0/3.0) * pow(s/L/2.0,9.0) - pow(s/L/2.0,3.0));

	printf("*****************************************************************************\n");
	printf("Momentum: %lf\n",mv);
	printf("*****************************************************************************\n");
	printf("Starting Temperature (K): %lf\n",T);
        printf("Average Temperature (K): %lf\n",Tavg);
	printf("Percent Difference: %.2lf \n",(Tavg-T)/T*100.0);
	printf("*****************************************************************************\n");
	printf("--- PRESSURE ---\n");
	printf("Average Reduced Pressure: %lf\n",Pavg);
	printf("Pressure from Long-Range Correction: %lf\n",plrc);
	printf("Average Reduced Pressure with Long-Range Correction: %lf\n",(Pavg+plrc));
	printf("*****************************************************************************\n");
	printf("--- ENERGY ---\n");
	printf("Average Reduced Potential Energy: %lf\n",PEavg/e);
	printf("Energy from Long-Range Correction per Particle: %lf\n",ulrc/e/N);
	printf("Average Reduced Potential Energy with Long-Range Correction: %lf\n",(PEavg+ulrc/N)/e);
	printf("*****************************************************************************\n");
	printf("---SPECIFIC HEAT AT CONSTANT VOLUME---\n");
	printf("Specific Heat from Fluctuations in Potenial Energy: %lf\n",cvPE);
	printf("Specific Heat from Fluctuations in Kinetic Energy: %lf\n",cvKE);
	printf("*****************************************************************************\n");

	end = clock();

	sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("SIMULATION RUNTIME: %.2lf seconds\n",sim_time);
	printf("*****************************************************************************\n");

	return(0);
}

/**
//  UNCOMMENT FOR SIMPLE CUBIC
void crystalLattice()
{
	int i,j,k,n;

	//Number of atoms in each x,y,z direction rounded up to the nearest whole number
	// n = ceil(pow(N,1.0/3.0));
    n = 27;

	//Atom spacing in given x,y,z direction
	double dr = s;

	//Index for total number of atoms, N
	int index = 0;

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			for(k=0;k<n;k++)
			{
				r[index][0] = i * dr;
				r[index][1] = j * dr;
				r[index][2] = k * dr;

				index++;
			}

	initializeVelocities();

	//Print initial positions and velocities
//	for(i=0;i<N;i++)
//		printf("%lf\t %lf\t %lf\n",r[i][0],r[i][1],r[i][2]);
//	for(i=0;i<N;i++)
//		printf("%lf\t %lf\t %lf\n",v[i][0],v[i][1],v[i][2]);

}
*/

void crystalLattice()
{
	int i,j,k;
	int index = 0;
	double n,dr,dr2;

	n = pow(N/4.0,1.0/3.0);
	dr = L / n;
	dr2 = dr / 2.0;

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			for(k=0;k<n;k++)
			{
				if(index < N)
				{
					r[index][0] = i * dr;
					r[index][1] = j * dr;
					r[index][2] = k * dr;
				}
				index++;

				if(index < N)
				{
					r[index][0] = i * dr + dr2;
					r[index][1] = j * dr + dr2;
					r[index][2] = k * dr;
				}
				index++;

				if(index < N)
				{
					r[index][0] = i * dr;
					r[index][1] = j * dr + dr2;
					r[index][2] = k * dr + dr2;
				}
				index++;

				if(index < N)
				{
					r[index][0] = i * dr + dr2;
					r[index][1] = j * dr;
					r[index][2] = k * dr + dr2;
				}
				index++;
			}

	initializeVelocities();

//        for(i=0;i<N;i++)
//        	printf("%lf\t %lf\t %lf\n",r[i][0],r[i][1],r[i][2]);
}

void wrapToBox()
{
	int i,j;

	for (i=0;i<N;i++)
		for (j=0;j<3;j++)
			r[i][j] += -L * floor(r[i][j]/L);
}

void initializeVelocities()
{
	int i,j;
	double vScale;
	double v2 = 0;
	double instant_temp = 0;
	double vCM[3] = {0,0,0};

	//Assign random initial velocities
	for(i=0;i<N;i++)
		for(j=0;j<3;j++)
			v[i][j] = generateGaussian() - 0.5;

	//Calculating center of mass velocity
        for(i=0;i<N;i++)
                for(j=0;j<3;j++)
                        vCM[j] += v[i][j];

        for(i=0;i<3;i++)
                vCM[i] /= N;

        //Subtract off center of mass velocity
        for(i=0;i<N;i++)
                for(j=0;j<3;j++)
                        v[i][j] -= vCM[j];

	//Scale initial velocity with scaled temperature (i.e. initial temperature from T* against temperature "right now")
	//instant_temp = m * v^2 / ((3N - 3) * kB) -- kB = 1 in reduced units
        for(i=0;i<N;i++)
                for(j=0;j<3;j++)
                        v2 += v[i][j] * v[i][j] / N;

        for(i=0;i<N;i++)
                instant_temp += m * v2;

        instant_temp /= (3.0*N - 3.0);

        vScale = sqrt(T / instant_temp);
	
	for(i=0;i<N;i++)
		for(j=0;j<3;j++)
			v[i][j] *= vScale;
}

double calculateAcceleration()
{
	int i,j,k;
	double F,r2,r6,r12,sor;
	double V = 0;

	//Position of i relative to j
	double rij[3];

	//Initialize acceleration to 0
	for(i=0;i<N;i++)
		for(j=0;j<3;j++)
			a[i][j] = 0;

	//Loop over all distinct pairs i,j
	for(i=0;i<N-1;i++)
		for(j=i+1;j<N;j++)
		{
			r2 = 0.0;
			for(k=0;k<3;k++)
			{
				//Component-by-componenent position of i relative to j
				rij[k] = r[i][k] - r[j][k];

				//Periodic boundary conditions
				rij[k] += -L * trunc(rij[k]/L);
                while(rij[k] >= 0.5*L)
					rij[k] -= L;
				while(rij[k] < -0.5*L)
                    rij[k] += L;

				//Dot product of the position component
				r2 += rij[k] * rij[k];
			}

			if (r2 < 0.25*L*L)
			{
				//Multicomponent variables to use below, using LB mixing rules
				//sigma_ab = 0.5 * (sigma[i] + sigma[j]);
				//epsilon_ab = sqrt(epsilon[i] * epsilon[j]);
				sor = (s*s) / r2;
				r6 = sor * sor * sor;
				r12 = r6 * r6;

				F = 48.0 * e/r2 * (r12 - 0.5*r6);

				//Virial sum for pressure calculation
				V += F * r2;

				for(k=0;k<3;k++)
			        {
			                a[i][k] += rij[k] * F/m;
			                a[j][k] -= rij[k] * F/m;
			        }
			}
		}

	return V;
}

void velocityVerlet()
{
	int i,j;

	//Calculate acceleration from forces at current position
	calculateAcceleration();

	//Update positions and velocity with current velocity and acceleration
	for(i=0;i<N;i++)
		for(j=0;j<3;j++)
		{
			r[i][j] += v[i][j]*dt + 0.5*a[i][j]*pow(dt,2.0);
			v[i][j] += 0.5*a[i][j]*dt;
		}
		
	//Update acceleration from current position
	calculateAcceleration();

	//Update velocity with updated acceleration
	for(i=0;i<N;i++)
		for(j=0;j<3;j++)
			v[i][j] += 0.5*a[i][j]*dt;
}

double potentialEnergy()
{
	int i,j,k;
	double sor,r2,r6,r12;
	double PE = 0.0;
	double rij[3];

	for(i=0;i<N-1;i++)
		for(j=i+1;j<N;j++)
		{
			r2 = 0.0;
	                for(k=0;k<3;k++)
			{
				rij[k] = r[i][k] - r[j][k];
	
				rij[k] += -L * trunc(rij[k]/L);
	                        while(rij[k] >= L/2.0)
					rij[k] -= L;
				while(rij[k] < -L/2.0)
	            	        	rij[k] += L;

	                	r2 += rij[k] * rij[k];
			}

			if(r2 < L*L/4.0)
			{	
				sor = (s*s) / r2;
				r6 = sor * sor * sor;
				r12 = r6 * r6;

				PE += 4.0 * e * (r12 - r6) / N;
			}
		}

	return PE;
}

double kineticEnergy()
{
	int i,j;
	double v2 = 0.0;
	double KE = 0.0;

	for(i=0;i<N;i++)
		for(j=0;j<3;j++)
			v2 += v[i][j] * v[i][j];

	KE += 0.5 * m * v2 / N;

	return KE;
}

//Generation of random number sampling via the Marsaglia polar method (shamelessly stolen from Wikipedia: https://en.wikipedia.org/wiki/Marsaglia_polar_method#Implementation)
double generateGaussian()
{
	static double spare;
	static bool hasSpare = false;
	double u,v,w;

	srand(time(NULL));

	if(hasSpare)
	{
		hasSpare = false;
		return spare;
	}

	else
	{
		do
		{
			u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
			v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
			w = u*u + v*v;
		} while(w >= 1.0 || w == 0.0);

		w = sqrt(-2.0 * log(w)/w);
		spare = v * w;
		hasSpare = true;

		return u*w;
	}
}

double meanSquaredVelocity()
{
	int i;
	double v2;
	double vx2 = 0.0;
	double vy2 = 0.0;
	double vz2 = 0.0;

	for(i=0;i<N;i++)
	{
		vx2 += v[i][0]*v[i][0];
		vy2 += v[i][1]*v[i][1];
		vz2 += v[i][2]*v[i][2];
	}

	v2 = (vx2+vy2+vz2) / N;

	return v2;
}

void thermostat()
{
	int i,j;
    double vScale;
	double v2 = 0.0;
    double instant_temp = 0.0;

	for(i=0;i<N;i++)
		for(j=0;j<3;j++)
			v2 += v[i][j] * v[i][j] / N;

        for(i=0;i<N;i++)
                instant_temp += m * v2;

        instant_temp /= (3.0*N - 3.0);

        vScale = sqrt(T / instant_temp);

        for(i=0;i<N;i++)
                for(j=0;j<3;j++)
                        v[i][j] *= vScale;
}

void radialDist(FILE *fp)
{
        int i,j,n;
        int row;
        double dx,dy,dz,r,dr,rij;
        double g[BIN];
        double x[N],y[N],z[N];

        FILE *fgr, *fxyz;

        dr = L / 2.0 / BIN;

        for(i=0;i<BIN;i++)
                g[i] = 0.0;

        fxyz = fopen("traj.xyz","r");
        for(n=0;n<N_STEPS;n++)
        {
                //Assign the first row in trajectory file as -1
                for(row=-1;row<N;row++)
                {
                        if(row == -1)
                                fscanf(fxyz,"%*d\n\n"); //Ignore the atom type and empty space beneath
                        else
                                fscanf(fxyz,"%*s %lf %lf %lf\n",&x[row],&y[row],&z[row]); //Each array contains the coordinates for N atoms at each x,y,z position. For unknown reasons, this method ignores the final block of coordinates, but this is deemed insignificant for calculating g(r)
                }

                if(n >= THERMO)
                {
                        for(i=0;i<N-1;i++)
                                for(j=i+1;j<N;j++)
                                {
					//Apply boundary conditions after thermostating
                                        dx = x[i] - x[j];
                                        dx += -L * round(dx/L);
					while(dx >= L/2.0)
						dx -= L;
					while(dx < -L/2.0)
						dx += L;

                                        dy = y[i] - y[j];
                                        dy += -L * round(dy/L);
					while(dy >= L/2.0)
                                                dy -= L;
                                        while(dy < -L/2.0)
                                                dy += L;

                                        dz = z[i] - z[j];
                                        dz += -L * round(dz/L);
					while(dz >= L/2.0)
                                                dz -= L;
                                        while(dz < -L/2.0)
                                                dz += L;

                                        rij = sqrt(dx*dx + dy*dy + dz*dz);
					
					if(rij < L/2.0)
	                                        g[(int)(rij/dr)] += 2.0;
                                }
                }
        }
	fclose(fxyz);

        fgr = fopen("radial_dist.dat","w");
        fprintf(fgr,"r\t\t g(r)\n");
        for(i=0;i<BIN;i++)
        {
                r = (i+0.5) * dr;
                g[i] /= THERMO;
                g[i] /= 4.0*M_PI/3.0 * (pow(i+1.0,3.0) - pow(i,3.0)) * pow(dr,3.0) * p;
                fprintf(fgr,"%lf\t %lf\n",r,g[i]/N);
        }
}

void MSD(FILE *fp)
{
        int i,j,n;
        int row;
	double t = 0.0;
	double dx0,dy0,dz0;
        double dx,dy,dz;
	double rij_0,rij_t,avg_r,msd;
	double x0[N],y0[N],z0[N];
        double x[N],y[N],z[N];

        FILE *fxyz, *fmsd;

	avg_r = 0.0;

	fmsd = fopen("msd.dat","w");
	fprintf(fmsd,"Time (ps)\t MSD (A^2)\n");

	//See comments in radial distribution function for how loops function
	fxyz = fopen("traj.xyz","r");
        for(n=0;n<N_STEPS;n++)
        {
		//Stores the initial positions in x,y,z arrays and computes rij after thermostating
		if(n == THERMO)
		{
			for(row=-1;row<N;row++)
			{
				if(row == -1)
					fscanf(fxyz,"%*d\n\n");
				else
					fscanf(fxyz,"%*s %lf %lf %lf\n",&x0[row],&y0[row],&z0[row]);
			}

			//Obtain initial position value, r(0)
			for(i=0;i<N-1;i++)
				for(j=i+1;j<N;j++)
				{
                                        dx0 = x0[i] - x0[j];
                                        dx0 += -L * round(dx0/L);
					while(dx0 >= L/2.0)
                                                dx0 -= L;
                                        while(dx0 < -L/2.0)
                                                dx0 += L;

                                        dy0 = y0[i] - y0[j];
                                        dy0 += -L * round(dy0/L);
					while(dy0 >= L/2.0)
                                                dy0 -= L;
                                        while(dy0 < -L/2.0)
                                                dy0 += L;

                                        dz0 = z0[i] - z0[j];
                                        dz0 += -L * round(dz0/L);
					while(dz0 >= L/2.0)
                                                dz0 -= L;
                                        while(dz0 < -L/2.0)
                                                dz0 += L;

					rij_0 = sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0);
				}
		}

                for(row=-1;row<N;row++)
                {
                        if(row == -1)
                                fscanf(fxyz,"%*d\n\n");
                        else
                                fscanf(fxyz,"%*s %lf %lf %lf\n",&x[row],&y[row],&z[row]); //Unlike the g(r) function, this grabs the final coordinate block. Again, unknown as to why they function differently
                }

		//Obtain r(t) at each step after thermostating
		if(n >= THERMO)
		{
			for(i=0;i<N-1;i++)
				for(j=i+1;j<N;j++)
				{
					dx = x[i] - x[j];
					dx += -L * round(dx/L);
					while(dx >= L/2.0)
                                                dx -= L;
                                        while(dx < -L/2.0)
                                                dx += L;

					dy = y[i] - y[j];
	                                dy += -L * round(dy/L);
					while(dy >= L/2.0)
                                                dy -= L;
                                        while(dy < -L/2.0)
                                                dy += L;
	
	                                dz = z[i] - z[j];
	                                dz += -L * round(dz/L);
					while(dz >= L/2.0)
                                                dz -= L;
                                        while(dz < -L/2.0)
                                                dz += L;

	                                rij_t = sqrt(dx*dx + dy*dy + dz*dz);
				}
			
			//Average distance travelled for all atoms per step
			for(i=0;i<N;i++)
				avg_r += (rij_t - rij_0)*(rij_t - rij_0) / n;

			//Converting simulation step to time and calculating MSD(t)
			t = (n - THERMO) * dt;
			msd = avg_r / 6.0 / N;
			fprintf(fmsd,"%lf\t %E\n",t,msd);
		}
	}
}

/*
void VACF(FILE *fp)
{
        int i,j,n;
        int row;
	double t;
	double dvx0,dvy0,dvz0;
        double dvx,dvy,dvz;
	double vij_0,vij_t,avg_v0,avg_vt,diff_v,sq_v0,sq_vt,vacf;
	double vx0[N],vy0[N],vz0[N];
        double vx[N],vy[N],vz[N];

        FILE *fvel, *fvacf;

	avg_v0 = 0.0;
	avg_vt = 0.0;
	sq_v0 = 0.0;
	sq_vt = 0.0;

	fvacf = fopen("vacf.dat","w");
	fprintf(fvacf,"Time (ps)\t VACF, Normalized\n");

	//See comments in radial distribution function for how loops function
	fvel = fopen("velocities.xyz","r");
        for(n=0;n<N_STEPS;n++)
        {
		if(n == THERMO)
		{
			for(row=-1;row<N;row++)
			{
				if(row == 0)
					fscanf(fvel,"%*d\n\n");
				else
					fscanf(fvel,"%*s %lf %lf %lf\n",&vx0[row],&vy0[row],&vz0[row]);
			}

			//Obtain initial velocity, v(0)
			for(i=0;i<N-1;i++)
				for(j=i+1;j<N;j++)
				{
                                        dvx0 = vx0[i] - vx0[j];
                                        dvx0 += -L * round(dvx0/L);

                                        dvy0 = vy0[i] - vy0[j];
                                        dvy0 += -L * round(dvy0/L);

                                        dvz0 = vz0[i] - vz0[j];
                                        dvz0 += -L * round(dvz0/L);

					vij_0 = sqrt(dvx0*dvx0 + dvy0*dvy0 + dvz0*dvz0);
				}
		}

                for(row=-1;row<N;row++)
                {
                        if(row == -1)
                                fscanf(fvel,"%*d\n\n");
                        else
                                fscanf(fvel,"%*s %lf %lf %lf\n",&vx[row],&vy[row],&vz[row]);
                }

		//Obtain v(t) at each step after thermostating
		if(n >= THERMO)
		{
			for(i=0;i<N-1;i++)
				for(j=i+1;j<N;j++)
				{
					dvx = vx[i] - vx[j];
					dvx += -L * round(dvx/L);
	
					dvy = vy[i] - vy[j];
	                                dvy += -L * round(dvy/L);
	
	                                dvz = vz[i] - vz[j];
	                                dvz += -L * round(dvz/L);
	
	                                vij_t = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
				}

			//Dot product of v(t) and v(0)
			for(i=0;i<N;i++)
			{
				avg_v0 += (vij_0 * vij_0);
				avg_vt += (vij_t * vij_0);
			}

			//Diffusivivty from velocity
			diff_v = avg_vt / 3.0 / N;

			//Converting simulation step to time and normalizing diffusion
			t = (n - THERMO + 1) * dt;
			vacf = diff_v / avg_v0;
			fprintf(fvacf,"%lf\t %lf\n",t,vacf);
	        }
	}
}
*/
