CC=gcc
CFLAGS=-Wall -O3 -march=native -ffast-math -funroll-loops -fopenmp
INCLUDES=-I/opt/X11/include
LDFLAGS=-L/opt/X11/lib -lX11 -lm -lpthread -fopenmp
SRC=.

all: galsim_pthrd_o galsim_pthrd_yu galsim_seq galsim_pthrd_ov2 galsim_pthrd_ov3 galsim_omp

graphics.o: $(SRC)/graphics.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/graphics.c -o $@

galsim_pthrd_o.o: $(SRC)/galsim_pthrd_o.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_pthrd_o.c -o $@

galsim_pthrd_ov2.o: $(SRC)/galsim_pthrd_ov2.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_pthrd_ov2.c -o $@

galsim_pthrd_ov3.o: $(SRC)/galsim_pthrd_ov3.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_pthrd_ov3.c -o $@

galsim_pthrd_yu.o: $(SRC)/galsim_pthrd_yu.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_pthrd_yu.c -o $@

galsim_omp.o: $(SRC)/galsim_omp.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_omp.c -o $@

galsim_seq.o: $(SRC)/galsim_seq.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_seq.c -o $@

galsim_pthrd_o: galsim_pthrd_o.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

galsim_pthrd_ov2: galsim_pthrd_ov2.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

galsim_pthrd_ov3: galsim_pthrd_ov3.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

galsim_pthrd_yu: galsim_pthrd_yu.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

galsim_omp: galsim_omp.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

galsim_seq: galsim_seq.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o galsim_pthrd_o galsim_pthrd_yu galsim_seq galsim_pthrd_ov2 galsim_pthrd_ov3 galsim_omp
