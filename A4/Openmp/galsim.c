#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "graphics.h"
#include <omp.h>

#include <sys/time.h>
#include <unistd.h>

double get_wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec / 1000000;
}

void log_result(const char *filename,
	double runtime,
	int N,
	const char *version,
	const int nsteps,
	const int Nthreads)
{
    FILE *f = fopen(filename, "a");
    if (!f) {
	perror("fopen");
	return;
    }

    fprintf(f, "%.15f %d %s %d %d\n", runtime, N, version, nsteps, Nthreads);
    fclose(f);
}

// graphic +++
const float circleRadius=0.005, circleColor=0;
const int windowWidth=800;
// graphic ---


// Array of Structs approach, easy to understand and implement 
// objectwise manipulations
// although, problems is that between two particles data is structured like
// [x1, y1, m1, dx1, dy1, b1][x2, y2, m2, dx2, dy2, b2]...up to Nth particle

// Struct of Arrays, keeps relevant data contigious
// goal is to make it easier for the compiler
// to use SMID, since the data is structured like
// [x1, x2, x3,...,xN][y1, y2, y3,...,yN][m1, m2, m3,...,mN]...and so on for the other properties

typedef struct {
    int Nparticles;
    char* filename;
    int nsteps;
    double delta_t;
    int en_graphics;
    int Nthreads;
    double dt_G;
    double epsilon;
} Sim_params;

typedef struct {
    double *x;
    double *y;
    double *mass;
    double *dx;
    double *dy;
    double *brightness;
} Gal_state;


Gal_state* init_state(const int N) {

    Gal_state* state = (Gal_state*)malloc(sizeof(Gal_state));
    state->x = (double*)malloc(N * sizeof(double));
    state->y = (double*)malloc(N * sizeof(double));
    state->dx = (double*)malloc(N * sizeof(double));
    state->dy = (double*)malloc(N * sizeof(double));
    state->mass = (double*)malloc(N * sizeof(double));
    state->brightness = (double*)malloc(N * sizeof(double));

    return state;
}

void free_state_memory(Gal_state *state) {
    free(state->x);
    free(state->y);
    free(state->mass);
    free(state->dx);
    free(state->dy);
    free(state->brightness);

    state->x = NULL;
    state->y = NULL;
    state->mass = NULL;
    state->dx = NULL;
    state->dy = NULL;
    state->brightness = NULL;
}

Gal_state* read_state_config(const Sim_params* restrict params) {
    FILE *fp;
    fp = fopen(params->filename, "rb");

    if (fp == NULL) {
	printf("open file error\n");
	return NULL;
    }

    Gal_state* particles = init_state(params->Nparticles);


    for (int i = 0; i < params->Nparticles; i++)
    {   
	if (
		fread(&particles->x[i], sizeof(double), 1, fp) != 1 ||
		fread(&particles->y[i], sizeof(double), 1, fp) != 1 ||
		fread(&particles->mass[i], sizeof(double), 1, fp) != 1 ||
		fread(&particles->dx[i], sizeof(double), 1, fp) != 1 ||
		fread(&particles->dy[i], sizeof(double), 1, fp) != 1 ||
		fread(&particles->brightness[i], sizeof(double), 1, fp) != 1
	   ) 
	{
	    printf("Error reading particle %d\n", i);
	    free_state_memory(particles);
	    free(particles);
	    fclose(fp);
	    return NULL;
	}
    }
    fclose(fp);
    return particles;
}

/* 

   Basic structure for the work a thread does 
   Have an assigned chunk of particles to work with,
   preferrably indices or pointers to an already existing state
   work with and update that chunk as per usual,
   This will create a force array for a chunk of particles
   with symmetric update for the rest of them.
   Padded with zeros in the beginning.

   All of these chunks then contain the force state of the entire system
   perhaps, then it is possible to paralleize and consolidate all of these chunkwise 
   forces into a single array

   Then you can trivially update each and every particle with the information
   from the consolidated chunk

TODO:
1. Fix the calculate_forces function
 * I think its finished
 2. Fix the consolidate_forces function
 * I think it is finished
 3. Fix the update_galstate() function
 * I think it is finished
 4. Figure out what the barrier function should look like
 * Just simple barrier implemented
 5. Fix the main loop so it handles the threads
 * Think its correct, main will participate 
 6. Should there be a pointer swap in this program? 
 * Isnt really needed as particle i update loads the current
 * state and then writes..

*/

int calculate_forces(const int Nparticles, const Gal_state* restrict particles,
	const double epsilon,
	double* restrict fx_net, double* restrict fy_net) {

    const double* restrict x = particles->x;
    const double* restrict y = particles->y;
    const double* restrict m = particles->mass;


    double* restrict fx = fx_net;
    double* restrict fy = fy_net;
    // Force chunk is updated using pointers
    // local force arrays have already been set to zero
    // inside consolidate_forces() function previous time-step

#pragma omp for schedule(dynamic) reduction(+:fx[:Nparticles], fy[:Nparticles])
    for (int i = 0; i < Nparticles; i++)
    {        
	//Copy particle[i] to local variables avoiding reading memory each time
	const double pos_x_i = x[i];
	const double pos_y_i = y[i];
	const double mass_i = m[i];

	// accumulate forces and update once
	double fxi = 0.0;
	double fyi = 0.0;
	// j starts from i+1 - symmetric force update
	for (int j = i+1; j < Nparticles; j++)
	{
	    const double pos_x_j = x[j];
	    const double pos_y_j = y[j];
	    const double mass_j = m[j];

	    const double r_x = pos_x_i - pos_x_j;
	    const double r_y = pos_y_i - pos_y_j;
	    const double r_ij_mag = sqrt(r_x*r_x + r_y*r_y) + epsilon;
	    const double invr = 1.0 / r_ij_mag;
	    const double div = invr * invr * invr;

	    // Update forces symmetrically for both particles: F_ij = -F_ji
	    fxi -= mass_j * div * r_x; //accumulate
	    fyi -= mass_j * div * r_y; //accumulate
	    fx[j] += mass_i * div * r_x;
	    fy[j] += mass_i * div * r_y;
	}

	fx[i] += fxi; // accumulated changes are written to i:th particle
	fy[i] += fyi;

    }
    // end of implicit barrier for the parallel omp for outer loop

    return 0;
}

int update_state(
	double* restrict Fx_array,
	double* restrict Fy_array,
	Gal_state* particles,
	const double dt_G,
	const double delta_t,
	const int Nparticles) {




    double* restrict Fx = Fx_array;
    double* restrict Fy = Fy_array;

    // accumulate and update to new state in one go
    double* restrict x = particles->x;
    double* restrict y = particles->y;
    double* restrict dx = particles->dx;
    double* restrict dy = particles->dy;

#pragma omp for schedule(static)
    for (int i = 0; i<Nparticles; i++) {

	const double vel_x_next = dx[i] + dt_G * Fx[i];
	const double vel_y_next = dy[i] + dt_G * Fy[i];
	Fx[i] = 0.0;
	Fy[i] = 0.0;

	// update
	x[i] += delta_t * vel_x_next;
	y[i] += delta_t * vel_y_next;

	dx[i] = vel_x_next;
	dy[i] = vel_y_next;

    }
    // implicit openmp barrier here
    return 0;
}


void run_simulation(const Sim_params* params,
	Gal_state* particles,
	double* Fx_array,
	double* Fy_array) {


    const int Nparticles = params->Nparticles;
    const double delta_t = params->delta_t;
    const double dt_G = params->dt_G;
    const double epsilon = params->epsilon;
    const int en_graphics = params->en_graphics;
    const int nsteps = params->nsteps;
    const int Nthreads = params->Nthreads;

    // init parallel block
#pragma omp parallel num_threads(Nthreads) shared(params, particles, Fx_array, Fy_array)
    {

	// time stepping loop-e-di-doop
	for (int step = 0; step < nsteps; step++) {

	    //1. Get force chunks
	    //2. Consolidate forces and update at the same time

	    calculate_forces(Nparticles, particles,
		    epsilon, Fx_array, Fy_array);

	    // barrier is impicit inside calculate forces..

	    update_state(Fx_array, Fy_array,
		    particles, dt_G, delta_t, Nparticles);
	    // barrier is implicit inside the update function


	    if (en_graphics) { // en_graphics is either const 0 or 1 for one simulation ==> branch prediction will essentially ignore this after the first timestep
#pragma omp master
		{
		    ClearScreen();
		    for (int i = 0; i<Nparticles; i++)
			DrawCircle(particles->x[i], particles->y[i], 1, 1,
				circleRadius, 1.0 / (Nparticles + 2) * (i+1));
		    Refresh();
		    usleep(10000);
		}
#pragma omp barrier
	    }
	}
    }
}



void write_state(FILE* fp, int N, Gal_state* particles) {
    for (int i = 0; i < N; i++)
    {
	fwrite(&particles->x[i], sizeof(double), 1, fp);
	fwrite(&particles->y[i], sizeof(double), 1, fp);
	fwrite(&particles->mass[i], sizeof(double), 1, fp);
	fwrite(&particles->dx[i], sizeof(double), 1, fp);
	fwrite(&particles->dy[i], sizeof(double), 1, fp);
	fwrite(&particles->brightness[i], sizeof(double), 1, fp);
    }
}

int main(int argc, char *argv[]) {

    if (argc <= 6)
    {
	printf("Not enough input!\n");
	return 0;
    }

    // const char *version_name = argv[7];
    // const double start = get_wall_seconds();

    // Runtime and compile time constants
    const double delta_t = strtod(argv[4], NULL);
    const int N = atoi(argv[1]);
    const Sim_params params = {
	.Nparticles = N,
	.filename = argv[2],
	.nsteps = atoi(argv[3]),
	.delta_t = delta_t,
	.en_graphics = strcmp(argv[5], "1") == 0,
	.Nthreads = atoi(argv[6]),
	.dt_G = (delta_t * 100.0) / (double)N,
	.epsilon = 1e-3
    };

    // Need to initialize the arrays to be used. 
    const int Nparticles = params.Nparticles;
    const int Nthreads = params.Nthreads;

    // particle state
    Gal_state *particles = read_state_config(&params);

    double* Fx_array = calloc(Nparticles, sizeof(double));
    double* Fy_array = calloc(Nparticles, sizeof(double));
    if (params.en_graphics) {
	InitializeGraphics(argv[0], windowWidth, windowWidth / 2);
	SetCAxes(0, 1);
    }

    run_simulation(&params, particles, Fx_array, Fy_array);

    if (params.en_graphics) {
	FlushDisplay();
	CloseDisplay();
    }

    // no need to write state when simulating
    FILE *fp = fopen("result.gal", "wb");
    write_state(fp, Nparticles, particles);
    fclose(fp);

    free(Fx_array); free(Fy_array);
    free_state_memory(particles);
    free(particles);

    // const double runtime = get_wall_seconds() - start;
    // log_result("timings.txt", runtime, Nparticles, version_name, params.nsteps, Nthreads);
    return 0;

}
