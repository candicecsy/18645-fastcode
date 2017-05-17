PROJ = interpolation
CC = g++

CFLAGS = -fopenmp -c -g -Wall -std=c++11
OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OMPFLAGS = -fopenmp
SIMDFLAGS = -msse2 -msse3 -O3 -lm -march=core-avx2

all: $(PROJ)

$(PROJ): $(OBJS)
	$(CC) $(OMPFLAGS) $^ -o $@ 

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@ $(SIMDFLAGS)

%.o : %.cpp %.h
	$(CC) $(CFLAGS) $< -o $@ $(SIMDFLAGS)	

clean:
	rm -f $(PROJ) $(OBJS) *.xml