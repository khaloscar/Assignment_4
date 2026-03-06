#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "graphics.h"
#define _XOPEN_SOURCE 600
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


// graphic +++
const float circleRadius=0.005, circleColor=0;
const int windowWidth=800;

void keep_within_box(float* xA, float* yA) {
  if(*xA > 1)
    *xA = 0;
  if(*yA > 1)
    *yA = 0;
}

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

Gal_state* read_state_config(const char* fileName, const int N) {
    FILE *fp;
    fp = fopen(fileName, "rb");

    if (fp == NULL) {
        printf("open file error\n");
        return NULL;
    }

    Gal_state* particles = init_state(N);


    for (int i = 0; i < N; i++)
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

void update_state(const int start, const int end, const int N, Gal_state* restrict particles,
                const double delta_t, const double G, const double epsilon,
                double* restrict f_x_ij, double* restrict f_y_ij, pthread_mutex_t* particle_locks) {
    double* f_x_ij_local = calloc(N, sizeof(double));
    double* f_y_ij_local = calloc(N, sizeof(double));

    for (int i = start; i < end; i++)
    {   
        //Copy particle[i] to local variables avoiding reading memory each time
        const double pos_x_i = particles->x[i];
        const double pos_y_i = particles->y[i];
        const double mass_i = particles->mass[i];

        double fxi = 0.0;
        double fyi = 0.0;
        // j starts from i+1 to calculate each particle pair only once
        for (int j = i+1; j < N; j++)
        {
            const double mass_j = particles->mass[j];
            const double pos_x_j = particles->x[j];
            const double pos_y_j = particles->y[j];
    
            const double r_x = pos_x_i - pos_x_j;
            const double r_y = pos_y_i - pos_y_j;
            const double r_ij_mag = sqrt(r_x*r_x + r_y*r_y) + epsilon;
            const double div = 1/((r_ij_mag)*(r_ij_mag)*(r_ij_mag));
            // Update forces symmetrically for both particles: F_ij = -F_ji
            fxi += mass_j * div * r_x; //accumulate changes to i here instead of changing memory each iteration, this made it possible to vectorize
            fyi += mass_j * div * r_y;

            f_x_ij_local[j] += - mass_i * div * r_x;
            f_y_ij_local[j] += - mass_i * div * r_y;
        }

        f_x_ij_local[i] += fxi; // accumulated changes are written to i:th particle
        f_y_ij_local[i] += fyi;
    }

    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&particle_locks[i]);
        f_x_ij[i] += f_x_ij_local[i];
        f_y_ij[i] += f_y_ij_local[i];
        pthread_mutex_unlock(&particle_locks[i]);
    }

    free(f_x_ij_local);
    free(f_y_ij_local);
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

// pthread +++
typedef struct {
    int start;
    int end;
    int nsteps;
    int N;
    Gal_state* restrict particles;
    double delta_t;
    double G;
    double epsilon;
    double* restrict f_x_ij;
    double* restrict f_y_ij;

    pthread_barrier_t* barrier;
    pthread_mutex_t* particle_locks;
    int thread_id;
} ThreadData;
// pthread ---

void* simulation(void* arg)
{
    ThreadData* data = (ThreadData*) arg;
    int start = data->start;
    int end = data->end;
    int nsteps = data->nsteps;
    int N = data->N;
    Gal_state* particles = data->particles;
    double delta_t = data->delta_t;
    double G = data->G;
    double epsilon = data->epsilon;
    double* f_x_ij = data->f_x_ij;
    double* f_y_ij = data->f_y_ij;
    pthread_barrier_t* barrier = data->barrier;
    pthread_mutex_t* particle_locks = data->particle_locks;
    int thread_id = data->thread_id;


    for (int step = 0; step < nsteps; step++)
    {
        if (thread_id == 0)
        {
            memset(f_x_ij, 0, N * sizeof(*f_x_ij));
            memset(f_y_ij, 0, N * sizeof(*f_y_ij));
        }

        pthread_barrier_wait(barrier);

        update_state(start, end, N, particles,
                    delta_t, G, epsilon,
                    f_x_ij, f_y_ij, particle_locks);
        // printf("thead %d done.\n", thread_id);
        pthread_barrier_wait(barrier);
        
        for (int i = start; i < end; i++)
        {
            
            const double pos_x_i = particles->x[i];
            const double pos_y_i = particles->y[i];
            const double mass_i = particles->mass[i];
            const double vel_x_i = particles->dx[i];
            const double vel_y_i = particles->dy[i];
            const double F_x = -G * mass_i * f_x_ij[i];
            const double F_y = -G * mass_i * f_y_ij[i];
            const double mass_i_div = 1.0f / mass_i;
            const double vel_x_next = vel_x_i + delta_t * F_x * mass_i_div;
            const double vel_y_next = vel_y_i + delta_t * F_y * mass_i_div; // last four definitons isnt used, save 4x mem?
            const double pos_x_next = pos_x_i + delta_t * vel_x_next;
            const double pos_y_next = pos_y_i + delta_t * vel_y_next;

            // update
            particles->x[i] = pos_x_next;
            particles->y[i] = pos_y_next;
            particles->dx[i] = vel_x_next;
            particles->dy[i] = vel_y_next;
        }
        pthread_barrier_wait(barrier);
    }

    return NULL;
}

void init_thread_data(ThreadData* threadData, int n_thread, int nsteps, int N, Gal_state* particles,
                      double delta_t, double G, double epsilon,
                      double* f_x_ij, double* f_y_ij,
                      pthread_barrier_t* barrier, pthread_mutex_t* particle_locks) {

    for (int i = 0; i < n_thread; i++) {
        threadData[i].nsteps = nsteps;
        threadData[i].N = N;
        threadData[i].particles = particles;
        threadData[i].delta_t = delta_t;
        threadData[i].G = G;
        threadData[i].epsilon = epsilon;
        threadData[i].f_x_ij = f_x_ij;
        threadData[i].f_y_ij = f_y_ij;
        threadData[i].barrier = barrier;
        threadData[i].particle_locks = particle_locks;
        threadData[i].thread_id = i;
        threadData[i].start = 0;
        threadData[i].end = N;
    }
}

void compute_thread_range(int N, int n_thread, int i, int *start_idx, int *end_idx)
{   
    float a = N * (N-1) / (2.0f * n_thread);
    int start = (*start_idx < N) ? *start_idx : N;
    int end = (*start_idx < N) ? *start_idx + 1 : N;
    int accumulate = (N - 1) - end;

    while (end < N) {
        if (accumulate >= a && i != n_thread - 1) break;
        end++;
        accumulate += N - 1 - end;
    }

    *start_idx = start;
    *end_idx = end;
    // printf("avg calulation: %.2f, #thread: %d, cnt:%d, col:%d~%d\n", a, i, accumulate, start, end);
}

void launch_thread(pthread_t* threads, ThreadData* threadData, int N, int n_thread) {
    int start_idx = 0;
    int end_idx = 0;
    for(int i = 0; i < n_thread; i++){
        compute_thread_range(N, n_thread, i, &start_idx, &end_idx);
        threadData[i].start = start_idx;
        threadData[i].end = end_idx;
        start_idx = end_idx;

        pthread_create(&threads[i], NULL, simulation, &threadData[i]);

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
    const int n_thread = atoi(argv[6]);
    const int N  = atoi(argv[1]);
    const char* filename = argv[2];
    const int nsteps = atoi(argv[3]);
    const double delta_t = strtod(argv[4], NULL);
    const bool en_graphics = (strcmp(argv[5], "1") == 0);
    const double G = 100.0f/N;
    const double epsilon = 1e-3; // in this case epsilon is really compile time constant

    pthread_t threads[n_thread];
    ThreadData* threadData = (ThreadData*)malloc(n_thread * sizeof(ThreadData));
    pthread_barrier_t barrier;
    pthread_mutex_t particle_locks[N];
    for (int i = 0; i < N; i++) {
        pthread_mutex_init(&particle_locks[i], NULL);
    }

    pthread_barrier_init(&barrier, NULL, n_thread);

    double* f_x_ij = (double*)malloc(N*sizeof(double));
    double* f_y_ij = (double*)malloc(N*sizeof(double)); // These are used to efficiently pre-compute forces, want to avoid calloc -> memset inside loop is used to reset the arrays?? Idk if it does anything useful in this code though

    Gal_state *particles = read_state_config(filename, N);

    init_thread_data(threadData, n_thread, nsteps, N, particles, 
                    delta_t, G, epsilon, f_x_ij, f_y_ij, &barrier, particle_locks);

    if (en_graphics)
    {
        float L=1, W=1;
        InitializeGraphics(argv[0], windowWidth, windowWidth/2);
        SetCAxes(0,1);

        for (int step = 0; step < nsteps; step++)
        {
            ClearScreen();
            for (int i = 0; i < N; i++)
            {
                DrawCircle(particles->x[i], particles->y[i], L, W, circleRadius, 1.0/(N+2)*(i+1));
            }


            launch_thread(threads, threadData, N, n_thread);


            for (int i = 0; i < n_thread; i++)
            {   
                pthread_join(threads[i], NULL);
            }


            Refresh();
            usleep(10000);
        }
        FlushDisplay();
        CloseDisplay();
    }
    else
    {
        launch_thread(threads, threadData, N, n_thread);

        for (int i = 0; i < n_thread; i++)
        {
            pthread_join(threads[i], NULL);
        }
    }
    
    // no need to write state when simulating
    // FILE *fp = fopen("result.gal", "wb");
    // write_state(fp, Nparticles, particles);
    // fclose(fp);
    

    pthread_barrier_destroy(&barrier);
    for (int i = 0; i < N; i++)
    {
        pthread_mutex_destroy(&particle_locks[i]);
    }

    free_state_memory(particles);
    free(threadData);
    free(particles);
    free(f_x_ij);
    free(f_y_ij);
    particles = NULL;
    f_y_ij = NULL; f_x_ij = NULL;
    
    const double runtime = get_wall_seconds() - start;
    log_result("timings.txt", runtime, N, version_name, nsteps, n_thread);

    return 0;

}

// This code changes the structure of data
// also it is able to vectorize part of the computations
// added const everywhere
// this code does not write final state to file
// this code times the main loop and writes the time to file
