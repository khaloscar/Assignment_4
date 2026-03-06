#!/bin/bash

# reference folder
input_folder="./input_data"
out_folder="./Testing_arena"
out_folder="./graphics/"

out_file="./result.gal"

# Do graphics on, and then
# Filenames, circles_N_2.gal, circles_N_4.gal, sun_and_planet_N_2.gal, sun_and_planets_N_3.gal, sun_and_planet_N_4.gal,
# arguments to file goes like this ./galsim_v4 N filename nsteps delta_t graphics

steps=1000
Nthreads=8

N=2
input_file="${input_folder}/circles_N_2.gal"
"${out_folder}/galsim" "$N" "$input_file" "$steps" 0.00001 1 "$Nthreads"

N=4
input_file="${input_folder}/circles_N_4.gal"
"${out_folder}/galsim" "$N" "$input_file" "$steps" 0.00001 1 "$Nthreads"

N=2
input_file="${input_folder}/sun_and_planet_N_2.gal"
"${out_folder}/galsim" "$N" "$input_file" "$steps" 0.00001 1 "$Nthreads"

N=3
input_file="${input_folder}/sun_and_planets_N_3.gal"
"${out_folder}/galsim" "$N" "$input_file" "$steps" 0.00001 1 "$Nthreads"
N=4
input_file="${input_folder}/sun_and_planets_N_4.gal"
"${out_folder}/galsim" "$N" "$input_file" "$steps" 0.00001 1 "$Nthreads"
