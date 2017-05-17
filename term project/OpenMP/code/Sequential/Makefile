PROJ = sequential
CC = g++

CFLAGS = -c -g -Wall -std=c++11
OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: $(PROJ)

$(PROJ): $(OBJS)
	$(CC) $(OMPFLAGS) $^ -o $@ 

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@

%.o : %.cpp %.h
	$(CC) $(CFLAGS) $< -o $@	

clean:
	rm -f $(PROJ) $(OBJS) *.xml
