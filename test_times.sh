#!/bin/bash
input_folder="./input_data"
out_folder="./Testing_arena"
Ns=(5000 1000)
time_steps=(100)
threads=(1 2 4 8 12 16)
versions=("pthrd_o" "pthrd_ov2" "pthrd_yu" "pthrd_ov3")
versions=("pthrd_o" "pthrd_yu" "omp")
delta_t=0.00001

make -C "$out_folder" clean
make -C "$out_folder"

# sequential baseline - single run, threads arg ignored
    for N in "${Ns[@]}"; do
    input_file="${input_folder}/ellipse_N_$(printf "%05d" $N).gal"
    for n_timesteps in "${time_steps[@]}"; do
        echo "Timing seq | N=${N} | steps=${n_timesteps}"
        for ((_i=0; _i<20; _i++)); do
            "${out_folder}/galsim_seq" "$N" "$input_file" "$n_timesteps" "$delta_t" 0 1 "seq"
        done
    done
done

# parallel versions - sweep thread counts
for version in "${versions[@]}"; do
    for N in "${Ns[@]}"; do
        input_file="${input_folder}/ellipse_N_$(printf "%05d" $N).gal"
        for n_timesteps in "${time_steps[@]}"; do
            for nthreads in "${threads[@]}"; do
                echo "Timing ${version} | N=${N} | steps=${n_timesteps} | threads=${nthreads}"
                for ((_i=0; _i<20; _i++)); do
                    "${out_folder}/galsim_${version}" "$N" "$input_file" "$n_timesteps" "$delta_t" 0 "$nthreads" "$version"
                done
            done
        done
    done
done

make -C "$out_folder" clean

echo "Running simulation analysis..."
python3 analyze_timings.py
