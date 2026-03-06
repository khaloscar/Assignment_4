CC=gcc
CFLAGS=-Wall -O3 -march=native -ffast-math
INCLUDES=-I/opt/X11/include
LDFLAGS=-L/opt/X11/lib -lX11 -lm -lpthread
SRC=.

all: galsim_pthrd_o galsim_pthrd_yu galsim_seq

graphics.o: $(SRC)/graphics.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/graphics.c -o $@

galsim_pthrd_o.o: $(SRC)/galsim_pthrd_o.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_pthrd_o.c -o $@

galsim_pthrd_yu.o: $(SRC)/galsim_pthrd_yu.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_pthrd_yu.c -o $@

galsim_seq.o: $(SRC)/galsim_seq.c $(SRC)/graphics.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SRC)/galsim_seq.c -o $@

galsim_pthrd_o: galsim_pthrd_o.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

galsim_pthrd_yu: galsim_pthrd_yu.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

galsim_seq: galsim_seq.o graphics.o
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o galsim_pthrd_o galsim_pthrd_yu galsim_seq
