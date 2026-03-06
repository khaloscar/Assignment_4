#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "graphics.h"

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

void update_state(const int N, Gal_state* restrict particles,
                const double delta_t, const double G, const double epsilon,
                double* restrict f_x_ij, double* restrict f_y_ij) {

    memset(f_x_ij, 0, N*sizeof(double));
    memset(f_y_ij, 0, N*sizeof(double));
    for (int i = 0; i < N; i++)
    {        
        //Copy particle[i] to local variables avoiding reading memory each time
        const double pos_x_i = particles->x[i];
        const double pos_y_i = particles->y[i];
        const double mass_i = particles->mass[i];
        const double vel_x_i = particles->dx[i];
        const double vel_y_i = particles->dy[i];

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
            f_x_ij[j] += - mass_i * div * r_x;
            f_y_ij[j] += - mass_i * div * r_y;
        }

        f_x_ij[i] += fxi; // accumulated changes are written to i:th particle
        f_y_ij[i] += fyi;

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
    
    if (argc <= 5)
    {
        printf("Not enough input!\n");
        return 0;
    }

    // Runtime and compile time constants
    const int N  = atoi(argv[1]);
    const char* filename = argv[2];
    const int nsteps = atoi(argv[3]);
    const double delta_t = strtod(argv[4], NULL);
    const bool en_graphics = (strcmp(argv[5], "1") == 0);
    const double G = 100.0f/N; const double epsilon = 1e-3; // in this case epsilon is really compile time constant

    double* f_x_ij = (double*)malloc(N*sizeof(double));
    double* f_y_ij = (double*)malloc(N*sizeof(double)); // These are used to efficiently pre-compute forces, want to avoid calloc -> memset inside loop is used to reset the arrays?? Idk if it does anything useful in this code though

    Gal_state *particles = read_state_config(filename, N);

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

            update_state(N, particles,
                        delta_t, G, epsilon,
                        f_x_ij, f_y_ij);

            Refresh();
            usleep(10000);
        }
        FlushDisplay();
        CloseDisplay();
    }
    else
    {
        for (int step = 0; step < nsteps; step++)
        {
            update_state(N, particles,
                        delta_t, G, epsilon,
                        f_x_ij, f_y_ij);
        }
    }
    
    FILE *fp = fopen("result.gal", "wb");
    write_state(fp, N, particles);
    fclose(fp);
    
    free_state_memory(particles);
    free(particles);
    free(f_x_ij);
    free(f_y_ij);
    particles = NULL;
    f_y_ij = NULL; f_x_ij = NULL;
    


    return 0;

}

// This code changes the structure of data
// also it is able to vectorize part of the computations
// added const everywhere
// this code does not write final state to file
// this code times the main loop and writes the time to file