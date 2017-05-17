/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         file_io.c                                                 */
/*   Description:  This program reads point data from a file                 */
/*                 and write cluster output to files                         */
/*   Input file format:                                                      */
/*                 ascii  file: each line contains 1 data object             */
/*                 binary file: first 4-byte integer is the number of data   */
/*                 objects and 2nd integer is the no. of features (or        */
/*                 coordinates) of each object                               */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */

#include "inserts.h"

#define MAX_CHAR_PER_LINE 128


/*---< file_read() >---------------------------------------------------------*/
int*** file_read( char *filename,      /* input file name */
                  int  *numWeight,       /* no. data objects (local) */
                  int  *numHight,
				  int  *numChannel)     /* no. coordinates */
{
		int ***objects;
		int     i, j,k;

 /* input file is in ASCII format -------------------------------*/
        FILE *infile;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }
		fscanf(infile,"%d %d %d",numWeight,numHight,numChannel);

        
		/* allocate space for objects[][] and read all objects */
        objects    = (int ***) malloc((*numWeight) * sizeof(int**));
		
        for (i=0; i<(*numWeight); i++){
			if(i==0)
				objects[i] =  (int**) malloc((*numWeight)*(*numHight) * sizeof(int*));
			else
				objects[i] = objects[i-1] + (*numHight);
 
			for(j=0;j<(*numHight);j++){
				if(i==0 && j==0)
					objects[i][j]= (int*) malloc ((*numWeight)*(*numHight) *(*numChannel)* sizeof(int));
				else if(j==0 && i!=0)
					objects[i][j] = objects[i-1][(*numHight)-1] + (*numChannel);
				else
					objects[i][j]= objects[i][j-1] + (*numChannel);
			}
		}
		

        /* read all objects */
		for(k=0;k<(*numChannel);k++)
			for (i=0;i< (*numWeight);i++)
				for(j=0;j< (*numHight);j++)
					fscanf(infile,"%d",&objects[i][j][k]);

        	
		//
        fclose(infile);
		return objects;
}

/*---< file_write() >---------------------------------------------------------*/
int file_write(char      *filename,     /* input file name */
               int        numWeight,  /* no. clusters */
               int        numHight,      /* no. data objects */
               int        numChannel,    /* no. coordinates (local) */
               int     ***objects)     /* [numClusters][numCoords] centers */
{
    FILE *fptr;
    int   i, j, k;
    char  outFileName[1024];

    /* output: the coordinates of the cluster centres ----------------------*/
    sprintf(outFileName, "%s.txt", filename);
    printf("Writing matrix to the file\n", outFileName);

    fptr = fopen(outFileName, "w+");

	fprintf(fptr,"%d %d %d",numWeight,numHight,numChannel);
	fprintf(fptr,"\n");
	
    for(k=0;k<numChannel;k++){
		for (i=0;i<numWeight;i++)
			for(j=0;j<numHight;j++)
				fprintf(fptr, "%d ", objects[i][j][k]);
        fprintf(fptr, "\n");
	}
	
    fclose(fptr);
    return 1;
}
