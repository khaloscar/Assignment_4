#!/bin/bash
input_folder="./input_data"
ref_folder="./ref_output_data"
out_folder="./graphics"

nthreads=${1:-4}

Ns=(10 100 500 1000 2000)
for N in "${Ns[@]}"; do
    ref_file="${ref_folder}/ellipse_N_$(printf "%05d" $N)_after200steps.gal"
    out_file="./result.gal"
    input_file="${input_folder}/ellipse_N_$(printf "%05d" $N).gal"
    "${out_folder}/galsim" "$N" "$input_file" 200 0.00001 0 "$nthreads"
    "./compare_gal_files/a.out" "$N" "$ref_file" "$out_file"
done

ref_file="${ref_folder}/ellipse_N_03000_after100steps.gal"
out_file="./result.gal"
input_file="${input_folder}/ellipse_N_03000.gal"
"${out_folder}/galsim" 3000 "$input_file" 100 0.00001 0 "$nthreads"
"./compare_gal_files/a.out" 3000 "$ref_file" "$out_file"
