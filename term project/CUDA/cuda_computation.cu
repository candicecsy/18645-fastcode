#include <stdio.h>
#include <stdlib.h>
#include <inserts.h>

#define numThreads  32 
__global__
void new_matrix_computation(int * device_object,int *result, int numWeight,int numHight, int numChannel, float ratio){

	int bx = blockIdx.x; int by= blockIdx.y;
	int tx = threadIdx.x; int ty= threadIdx.y;
	
    extern __shared__ int s[];

	//compute the new row element 
	int row = by * blockDim.y + ty;
	//compute the new col element
	int col = bx * blockDim.x + tx;
	/*
	int size =  (int)ceil((double) numThreads * ratio);
	int oldnumWeight = numWeight* ratio;
	int oldnumHight = numHight * ratio;
	
	if(row<numWeight && col < numHight){
		for(int i=ty; i< size; i+= blockDim.x){
			// calculate the original
			int row0 = (by*numThreads + i) * ratio;
			for(int j=tx; j<size; j+=blockDim.y){
				int col0 = (bx* numThreads +j) * ratio;
				for(int m=0; m<numChannel;m++)
						s[i *size*numChannel + j*numChannel + m]=
							device_object[row0 *numHight *numChannel + col0* numChannel + m ];
			}
		}
	}
	__syncthreads();
	
	if(row<numWeight && col < numHight){
		// new rol and col;
		
		float rr = ty * ratio; 
		float yy = tx * ratio;	
		int l = (int) floor(yy);
		int r = (int) ceil(yy);
		int t = (int) floor(rr);
		int d = (int) ceil(rr);

		for(int i=0; i<numChannel; i++){
			if( r<size && d<size){
				float p= (1.0-(yy - l)) * (1.0-(rr - t)) * s[t * size*numChannel + l *numChannel +i] +
						(yy-l) *(1.0-(rr - t)) *s[d * size*numChannel + l *numChannel +i] +
						 (1.0-(yy-l))*(rr-t) * s[t * size*numChannel + r *numChannel +i] +
						(yy-l) *(rr-t) * s[d * size*numChannel + r *numChannel +i];
			
				result[row*numHight*numChannel + col* numChannel + i] = p;
			}
			else
				result[row*numHight*numChannel + col* numChannel + i] = 255;

		}
	}
	*/
		
	if(row<numWeight && col < numHight){	
		float rr = row * ratio; 
		float yy = col * ratio;	
		int l = (int) floor(yy);
		int r = (int) ceil(yy);
		int t = (int) floor(rr);
		int d = (int) ceil(rr);
		
		for(int i=0; i<numChannel; i++){
			if( d < oldnumWeight && r < oldnumHight ){
				float p= (1.0-(yy - l)) * (1.0-(rr - t))* device_object[t * oldnumHight * numChannel + l *numChannel +i] +
						(yy-l) *(1.0-(rr - t)) *device_object[d * oldnumHight * numChannel + l *numChannel +i] +
						 (1.0-(yy-l))*(rr-t) * device_object[t * oldnumHight  * numChannel + r *numChannel +i] +
						(yy-l) *(rr-t) * device_object[d * oldnumHight * numChannel + r *numChannel +i];
			
				result[row*numHight*numChannel + col* numChannel + i] = p;
			}
			else
				result[row*numHight*numChannel + col* numChannel + i] = 255;
		
		}
		
	}
	
	/*
	if(row<numWeight && col < numHight)	{
		int x = (col+0.5) * ratio -0.5;
		int y = (row+0.5) * ratio -0.5;
		float tx = col * ratio - x;
		float ty = row * ratio - y;
		for(int i=0; i<numChannel; i++){
			if(x< numWeight && y < numHight){
				int p = (1.0-tx) * (1.0-ty) *device_object[y * numHight * numChannel + x * numChannel +i] +
						 tx * (1.0-ty) * device_object[ y *  numHight  * numChannel + (x+1) * numChannel +i] +
						 (1.0-tx)* ty *  device_object[ (y+1) *  numHight  * numChannel + x * numChannel +i] +
						tx * ty *  device_object[ (y+1) *  numHight  * numChannel + (x+1) * numChannel +i];
			
				result[row*numHight*numChannel + col* numChannel + i] = p;

			}

		}
	}
	*/
	
}


int*** cuda_computation( int *** objects,
						 int numWeight,
						 int numHight,
						 int numChannel,
						 float multiples)
{
	int ***new_output;	//output : [newnumWeight][newnumHight][numChannel]
	int *device_objects;
	int *device_output;
	int newnumWeight = numWeight * multiples;
	int newnumHight = numHight * multiples;

	dim3 dimBlock(numThreads,numThreads);
	dim3 dimGrid((int)ceil((double) newnumHight/(numThreads*1.0)),
			     (int)ceil((double) newnumWeight/(numThreads*1.0)));
	
	//first calculate the new matrix size;
	float ratio = 1.0/ (multiples*1.0);
	printf("ratio is%f, newnumWeight is %d, newnumHight is %d \n",ratio,newnumWeight,newnumHight);
	new_output = (int ***) malloc(newnumWeight*sizeof(int**));
	for (int i=0; i<newnumWeight; i++){
			if(i==0)
				new_output[i] =  (int**) malloc(newnumWeight * newnumHight * sizeof(int*));
			else
				new_output[i] = new_output[i-1] + newnumHight;
 
			for(int j=0;j<newnumHight;j++){
				if(i==0 && j==0)
					new_output[i][j]= (int*) malloc (newnumWeight * newnumHight  *numChannel * sizeof(int));
				else if ( j==0 && i!=0 )
					new_output[i][j] = new_output[i-1][newnumHight-1] + numChannel;
				else	
					new_output[i][j] = new_output[i][j-1] + numChannel;
			}
	}

	const unsigned int SharedMem =  (int)ceil((double) numThreads * ratio )*
									(int)ceil((double) numThreads * ratio )* numChannel* sizeof(int); 
	//allocate memory and transfer data
	checkCuda(cudaMalloc(&device_objects,numWeight*numHight*numChannel*sizeof(int)));

	checkCuda(cudaMemcpy(device_objects,objects[0][0],
			  numWeight*numHight*numChannel*sizeof(int),cudaMemcpyHostToDevice));
	checkCuda(cudaMalloc(&device_output, newnumWeight*newnumHight*numChannel*sizeof(int)));
	//the core function
	new_matrix_computation<<<dimGrid, dimBlock,SharedMem>>>(device_objects,device_output,newnumWeight,newnumHight,numChannel,ratio);

	checkCuda(cudaMemcpy(new_output[0][0],device_output,newnumWeight*newnumHight*numChannel*sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(device_objects);
	cudaFree(device_output);
	return new_output;
}

__global__
void new_matrix_computation2(int * device_object,int *result, int numWeight,int numHight, int numChannel ,int x ,int y){

	int bx = blockIdx.x; int by= blockIdx.y;
	int tx = threadIdx.x; int ty= threadIdx.y;
	//compute the new row element 
	int row = by*numThreads + ty;
	//compute the new col element
	int col = bx*numThreads + tx;
	if(row<numWeight && col < numHight){
		int orignaly = row - y;
		int orignalx = col - x;
		for(int i=0; i<numChannel; i++){
			if(orignaly <0 || orignalx <0)
				result[row*numHight*numChannel + col* numChannel + i] = 255;
			else
				result[row*numHight*numChannel + col* numChannel + i] = device_object[orignaly*numHight*numChannel + orignalx* numChannel + i];
		}
	}

}


int*** cuda_computation2 ( int *** objects,
						 int numWeight,
						 int numHight,
						 int numChannel){

	int ***new_output;	
	int *device_objects;
	int *device_output;
	int x= 200 , y=100;
	dim3 dimBlock(numThreads,numThreads);
	dim3 dimGrid((int)ceil((double) numHight/(numThreads*1.0)),
			     (int)ceil((double) numWeight/(numThreads*1.0)));

	new_output = (int ***) malloc(numWeight*sizeof(int**));

	for (int i=0; i<numWeight; i++){
			if(i==0)
				new_output[i] =  (int**) malloc(numWeight * numHight * sizeof(int*));
			else
				new_output[i] = new_output[i-1] + numHight;
 
			for(int j=0;j<numHight;j++){
				if(i==0 && j==0)
					new_output[i][j]= (int*) malloc (numWeight * numHight  *numChannel * sizeof(int));
				else if ( j==0 && i!=0 )
					new_output[i][j] = new_output[i-1][numHight-1] + numChannel;
				else	
					new_output[i][j] = new_output[i][j-1] + numChannel;
			}
	}
	checkCuda(cudaMalloc(&device_objects,numWeight*numHight*numChannel*sizeof(int)));

	checkCuda(cudaMemcpy(device_objects,objects[0][0],
			  numWeight*numHight*numChannel*sizeof(int),cudaMemcpyHostToDevice));

	checkCuda(cudaMalloc(&device_output, numWeight*numHight*numChannel*sizeof(int)));

	new_matrix_computation2<<<dimGrid, dimBlock>>>(device_objects,device_output,numWeight,numHight,numChannel,x,y);

	checkCuda(cudaMemcpy(new_output[0][0],device_output,numWeight*numHight*numChannel*sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(device_objects);
	cudaFree(device_output);
	return new_output;



}
