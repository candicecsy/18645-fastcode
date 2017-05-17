/*

    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <omp.h>
#include "matrix_mul.h"
#include <stdio.h>
#include <immintrin.h>

namespace omp
{
  void
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
    //define a matrix to transfer the sqmatrix_2 matrix
    float *sq_matrix_3 = (float *) malloc(sq_dimension * sq_dimension * sizeof(float));
    
    #pragma omp parallel for
    //transfer the sqmatrix_2 matrix
    for (unsigned int i = 0; i < sq_dimension; i++) {
        for (unsigned int j = 0; j < sq_dimension; j++) {
            sq_matrix_3[i*sq_dimension + j] = sq_matrix_2[j*sq_dimension + i];
        }
    }
    #pragma omp parallel for
    for (unsigned int i = 0; i < sq_dimension; i++) 
      {
          __m256 X, Y;   //vectors of length 8
          __m256 X1, Y1;
          
          float temp[8]; //store the result
          float temp1[8];
          
          for(unsigned int j = 0; j < sq_dimension; j++)
          {
              float sum = 0.0;
              unsigned int k = 0;
              __m256 acc = _mm256_setzero_ps(); //set to (0,0,0,0,0,0,0,0)
              __m256 acc1 = _mm256_setzero_ps();
              
              sq_matrix_result[i*sq_dimension + j] = 0;
              
              
              //compute the result by using AVX and loop unrolling
              if(sq_dimension >= 16)
              {
                  for (; k <= sq_dimension - 16; k += 16){
                  
                      //load chunk of 8 floats
                      X = _mm256_loadu_ps(&sq_matrix_1[i*sq_dimension + k]);
                      Y = _mm256_loadu_ps(&sq_matrix_3[j*sq_dimension + k]);
                      acc = _mm256_add_ps(acc, _mm256_mul_ps(X,Y));
                  
                      X1 = _mm256_loadu_ps(&sq_matrix_1[i*sq_dimension + k + 8]);
                      Y1 = _mm256_loadu_ps(&sq_matrix_3[j*sq_dimension + k + 8]);
                      acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(X1,Y1));

                  }
              //store acc into an array of floats
              _mm256_storeu_ps(&temp[0], acc);
              _mm256_storeu_ps(&temp1[0], acc1);
              sum = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7]+
                    temp1[0]+temp1[1]+temp1[2]+temp1[3]+temp1[4]+temp1[5]+temp1[6]+temp1[7];
              }
              
              //add the remaining values
              for (; k < sq_dimension; k++) {
                  sum += sq_matrix_1[i*sq_dimension + k] * sq_matrix_3[j*sq_dimension + k];
              }
              
              sq_matrix_result[i*sq_dimension + j] = sum;
              
          }
      }// End of parallel region
      free(sq_matrix_3);
  }
  
} //namespace omp


