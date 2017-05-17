#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

#include "inserts.h"
static void usage(char *argv0){
	char *help =
		"Usage: %s [switches] -i filename -n operation\n"
		"-i filename : file containing data to be operated\n"
		"-o : output file place"
		"-n : the operation type: \n"
		"-t : output timing results(default no)\n";
		fprintf(stderr,help,argv0);
		exit(-1);
}

int main(int argc,char **argv){

	int is_output_timing;
	char *filename;
	double timing,io_timing, calculate_timing;
	char *output_path;
	int opt;
	float multiples;

	int numWeight, numHight , numChannel;
	int ***objects;      //[numObjs][numCoords][numChannel];
	int ***output_objects,***output_objects2;  // [newnumOjbs][numCoords][]
	//some defalut values
	is_output_timing = 0;
	filename = NULL;
	output_path=NULL;

	while((opt = getopt(argc,argv,"i:o:n:t"))!= EOF) {
		switch(opt) {
			case 'i': filename = optarg;
					break;
			case 'o': output_path = optarg;
					break;
			case 'n': multiples = atof(optarg);
					break;
			case 't': is_output_timing=1;
					break;
			case '?': usage(argv[0]);                                  
                      break;                                                      
            default: usage(argv[0]);                                   
                      break; 
		}
	}
	if(filename== 0)	usage(argv[0]);
	if(is_output_timing) io_timing = wtime();
	//read data points from file ----------------------------------------------
	objects = file_read(filename,&numWeight,&numHight,&numChannel);
		
	if(objects == NULL) exit(1);

	if(is_output_timing) {
		timing = wtime();
		io_timing = timing - io_timing;
		calculate_timing = timing; 
	}
	// start the timer for the core computation
	output_objects = cuda_computation(objects,numWeight,numHight,numChannel,multiples);
	output_objects2 = cuda_computation2(objects,numWeight,numHight,numChannel); 
	// move

	free(objects[0][0]);
	free(objects[0]);
	free(objects);

	if(is_output_timing){

		timing  = wtime();
		calculate_timing = timing - calculate_timing;
	}
    int newnumWeight = numWeight * multiples;
	int newnumHight = numHight * multiples;

	file_write(output_path,newnumWeight,newnumHight,numChannel,output_objects);

	//file_write("moveoutput",numWeight,numHight,numChannel,output_objects2);

	free(output_objects[0][0]);
	free(output_objects[0]);
	free(output_objects);

	if(is_output_timing){
		io_timing += wtime()-timing;
		printf("\nPerforming **** linear insertion (CUDA version) ****\n");

		printf("Input file:  %s\n", filename);
		printf("numWeight       = %d\n", newnumWeight);
        printf("numHights     = %d\n", newnumHight);
        printf("numChannel   = %d\n", numChannel);
		printf("I/O time           = %10.4f sec\n", io_timing);
        printf("Computation timing = %10.4f sec\n", calculate_timing);


	}
	return 0;
}
