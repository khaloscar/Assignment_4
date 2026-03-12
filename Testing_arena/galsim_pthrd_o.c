#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "graphics.h"
#include <pthread.h>

#include <sys/time.h>

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

pthread_barrier_t barrier; 
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

typedef struct {
    int thrdid;
    const Sim_params* params;
    Gal_state* particles;
    double** Fx_mtrx;
    double** Fy_mtrx;
} Thread_args;

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

int calculate_forces(const int thrdid, const int Nthreads, const int Nparticles, const Gal_state* restrict particles,
                const double epsilon,
                double* restrict f_x_ij, double* restrict f_y_ij) {

    const double* restrict x = particles->x;
    const double* restrict y = particles->y;
    const double* restrict m = particles->mass;


    double* restrict fx = f_x_ij;
    double* restrict fy = f_y_ij;
    // Force chunk is updated using pointers
    // local force arrays have already been set to zero
    // inside consolidate_forces() function previous time-step
    const int chunk = 8;
    const int chunk_stride = Nthreads * chunk;
    for (int iblock = thrdid * chunk; iblock < Nparticles; iblock += chunk_stride) {
	const int iend = (iblock + chunk < Nparticles) ? (iblock + chunk) : Nparticles;
    
	for (int i = iblock; i < iend; i++)
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
    }
    return 0;
}

int consolidate_and_update_state(int Nthreads,
			double** restrict Fx_mtrx,
			double** restrict Fy_mtrx,
			const int start,
			const int stop,
			Gal_state* particles,
			const double dt_G,
			const double delta_t) {

    
    
    double vel_x_next; double vel_y_next;
    // accumulate and update to new state in one go
    double* restrict x = particles->x;
    double* restrict y = particles->y;
    double* restrict dx = particles->dx;
    double* restrict dy = particles->dy;

    for (int i = start; i<stop; i++) {

	double ifx = 0.0; double ify = 0.0; // accumulate force (or really acceleration) into one variable
	for (int j = 0; j<Nthreads; j++) {
	    ifx += Fx_mtrx[j][i];
	    ify += Fy_mtrx[j][i];

	    Fx_mtrx[j][i] = 0.0; // set to zero, avoiding call to memset later
	    Fy_mtrx[j][i] = 0.0;
	}

        vel_x_next = dx[i] + dt_G * ifx;
        vel_y_next = dy[i] + dt_G * ify;

        // update
        x[i] += delta_t * vel_x_next;
        y[i] += delta_t * vel_y_next;

        dx[i] = vel_x_next;
        dy[i] = vel_y_next;

    }
    return 0;
}

// old and currently unused code
// int update_galstate(Gal_state* restrict particles,
// 		     double* restrict F_x,
// 		     double* restrict F_y,
// 		     const double dt_G,
// 		     const double delta_t,
// 		     const int start,
// 		     const int stop) {
//
//     double* restrict x = particles->x;
//     double* restrict y = particles->y;
//     double* restrict dx = particles->dx;
//     double* restrict dy = particles->dy;
//     // trivial to parallelize, contigious scheduling
//     for (int i = start; i < stop; i++) {
//
//         const double vel_x_next = dx[i] + dt_G * F_x[i];
//         const double vel_y_next = dy[i] + dt_G * F_y[i];
//
//         // update
//         x[i] += delta_t * vel_x_next;
//         y[i] += delta_t * vel_y_next;
//
//         dx[i] = vel_x_next;
//         dy[i] = vel_y_next;
//
// 	// zero forces for next timestep, will avoid a separate memset or calloc..
// 	F_x[i] = 0.0;
// 	F_y[i] = 0.0;
//     }
//
//     return 0;
// }


void* thread_worker(void *args) {

    Thread_args* data = (Thread_args*)args;
    const Sim_params* params = data->params;

    const int thrdID = data->thrdid;
    Gal_state* particles = data->particles;

    const int Nparticles = params->Nparticles;
    const double delta_t = params->delta_t;
    const int Nthreads = params->Nthreads;
    const double dt_G = params->dt_G;
    const double epsilon = params->epsilon;
    const int en_graphics = params->en_graphics;

    // contigious scheduling, used in the consolidation of forces and update_state function
    const int base = Nparticles / Nthreads;
    const int rem = Nparticles % Nthreads;
    const int start = thrdID * base + (thrdID < rem ? thrdID : rem);
    const int stop = start + base + (thrdID < rem ? 1 : 0);

    // force matrices used for book-keeping all of the forces
    // calculated by each thread, Nthreads x Nparticles
    double** Fx_mtrx = data->Fx_mtrx;
    double** Fy_mtrx = data->Fy_mtrx;

    // the local force arrays used by each thread
    double* fx_chunk = Fx_mtrx[thrdID];
    double* fy_chunk = Fy_mtrx[thrdID];

    // time stepping loop-e-di-doop
    for (int step = 0; step < params->nsteps; step++) {

	//1. get force chunks
	//2. consolidate forces and update at the same time

	calculate_forces(thrdID, Nthreads, Nparticles, particles,
		epsilon, fx_chunk, fy_chunk);
	// barrier
	pthread_barrier_wait(&barrier);
	consolidate_and_update_state(Nthreads, Fx_mtrx, Fy_mtrx, start, stop, particles, dt_G, delta_t);
	//barrier
	pthread_barrier_wait(&barrier);

	if (en_graphics) { // en_graphics is either const 0 or 1 for one simulation ==> branch prediction will essentially ignore this after the first timestep
	    if (thrdID == 0) {
		ClearScreen();
		for (int i = 0; i < Nparticles; i++)
		    DrawCircle(particles->x[i], particles->y[i], 1, 1,
			       circleRadius, 1.0 / (Nparticles + 2) * (i + 1));
		Refresh();
		usleep(10000);
	    }
	    pthread_barrier_wait(&barrier);
	}
    }

    return NULL;
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
    
    if (argc <= 7)
    {
        printf("Not enough input!\n");
        return 0;
    }

    const char *version_name = argv[7];
    const double start = get_wall_seconds();

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
    // The big mtrx arrays for which the forces uses locally Nthreads x Nparticles size
    // The shared Force array which will update which is 1xNparticles
    // The shared state array
    const int Nparticles = params.Nparticles;
    const int Nthreads = params.Nthreads;

    // particle state
    Gal_state *particles = read_state_config(&params);
    
    // create the AoS for the thread data on the stack since its rather small
    // no free call needed
    pthread_t* threads = (pthread_t*)malloc((Nthreads-1)*sizeof(pthread_t));
    Thread_args thrd_data_arr[Nthreads];
    pthread_barrier_init(&barrier, NULL, Nthreads);
    // Each thread gets a row
    double** Fx_mtrx = (double**)malloc(Nthreads*sizeof(double*));
    double** Fy_mtrx = (double**)malloc(Nthreads*sizeof(double*));
    for (int i = 0; i < Nthreads; i++) {
	Fx_mtrx[i] = (double*)calloc(Nparticles, sizeof(double));
	Fy_mtrx[i] = (double*)calloc(Nparticles, sizeof(double));

	thrd_data_arr[i].thrdid = i;
	thrd_data_arr[i].params = &params;
	thrd_data_arr[i].particles = particles;
	thrd_data_arr[i].Fx_mtrx = Fx_mtrx;
	thrd_data_arr[i].Fy_mtrx = Fy_mtrx;
    }

    if (params.en_graphics) {
	InitializeGraphics(argv[0], windowWidth, windowWidth / 2);
	SetCAxes(0, 1);
    }
    // spawn threads, main will participate
    for (int i = 1; i < Nthreads; i++) {
	pthread_create(&threads[i-1], NULL, thread_worker, &thrd_data_arr[i]);
    }

    thread_worker(&thrd_data_arr[0]);

    for (int i = 0; i < Nthreads-1; i++) {
	pthread_join(threads[i], NULL);
    }

    if (params.en_graphics) {
	FlushDisplay();
	CloseDisplay();
    }

    // no need to write state when simulating
    // FILE *fp = fopen("result.gal", "wb");
    // write_state(fp, Nparticles, particles);
    // fclose(fp);
    
    for (int i = 0; i < Nthreads; i++) {
	free(Fx_mtrx[i]);
	free(Fy_mtrx[i]);
    }

    free(Fx_mtrx); free(Fy_mtrx);
    free_state_memory(particles);
    free(particles); free(threads);
    pthread_barrier_destroy(&barrier);

    const double runtime = get_wall_seconds() - start;
    log_result("timings.txt", runtime, Nparticles, version_name, params.nsteps, Nthreads);
    return 0;

}

// This code changes the structure of data
// also it is able to vectorize part of the computations
// added const everywhere
// this code does not write final state to file
// this code times the main loop and writes the time to file
