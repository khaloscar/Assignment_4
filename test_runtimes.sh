#!/bin/bash
input_folder="./input_data"
out_folder="./Testing_arena"
Ns=(10 100 500 1000 2000 5000 7000 10000)
time_steps=(100)
threads=(4 8 12 16)
versions=("pthrd_o" "pthrd_yu" "omp")
delta_t=0.00001
repeats=10

make -C "$out_folder" clean
make -C "$out_folder"

# sequential baseline
for N in "${Ns[@]}"; do
    input_file="${input_folder}/ellipse_N_$(printf "%05d" $N).gal"
    echo "Timing seq | N=${N} | steps=100"
    for ((_i=0; _i<$repeats; _i++)); do
        "${out_folder}/galsim_seq" "$N" "$input_file" 100 "$delta_t" 0 1 "seq"
    done
done

# parallel versions
for version in "${versions[@]}"; do
    for N in "${Ns[@]}"; do
        input_file="${input_folder}/ellipse_N_$(printf "%05d" $N).gal"
        for nthreads in "${threads[@]}"; do
            echo "Timing ${version} | N=${N} | steps=100 | threads=${nthreads}"
            for ((_i=0; _i<$repeats; _i++)); do
                "${out_folder}/galsim_${version}" "$N" "$input_file" 100 "$delta_t" 0 "$nthreads" "$version"
            done
        done
    done
done

make -C "$out_folder" clean
echo "Running analysis..."
mv timings.txt timings_runtimes.txt
python3 analyze_runtimes.py
