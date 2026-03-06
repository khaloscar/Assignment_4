#!/bin/bash
SRC="galsim.c"
OUTDIR="vec_reports"
INCLUDES="-I/opt/X11/include"

mkdir -p "$OUTDIR"

echo "=== Vectorization report ==="

echo "1. With -ffast-math: vectorized loops"
gcc -Wall -O3 -march=native -ffast-math -fopt-info-vec $INCLUDES -c "$SRC" 2> "$OUTDIR/vec_fastmath.txt"
echo "   -> $OUTDIR/vec_fastmath.txt"

echo "2. With -ffast-math: missed vectorizations"
gcc -Wall -O3 -march=native -ffast-math -fopt-info-vec-missed $INCLUDES -c "$SRC" 2> "$OUTDIR/vec_fastmath_missed.txt"
echo "   -> $OUTDIR/vec_fastmath_missed.txt"

echo "3. Without -ffast-math: vectorized loops"
gcc -Wall -O3 -march=native -fopt-info-vec $INCLUDES -c "$SRC" 2> "$OUTDIR/vec_no_fastmath.txt"
echo "   -> $OUTDIR/vec_no_fastmath.txt"

echo "4. Without -ffast-math: missed vectorizations"
gcc -Wall -O3 -march=native -fopt-info-vec-missed $INCLUDES -c "$SRC" 2> "$OUTDIR/vec_no_fastmath_missed.txt"
echo "   -> $OUTDIR/vec_no_fastmath_missed.txt"

# cleanup
rm -f *.o

echo ""
echo "Done. Reports saved to $OUTDIR/"
