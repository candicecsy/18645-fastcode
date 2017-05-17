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
    for (unsigned int i = 0; i < sq_dimension; i++) 
      {
          __m256 X, Y;
          __m128 F, N;
          
          float temp[8];
          
          for(unsigned int j = 0; j < sq_dimension; j++)
          {
              float sum = 0.0;
              unsigned int k = 0;
              __m256 acc = _mm256_setzero_ps(); //set to (0,0,0,0,0,0,0,0)
              __m128 acc1 = _mm_setzero_ps(); //set to (0,0,0,0)
              
              sq_matrix_result[i*sq_dimension + j] = 0;
              
            
              for (; k < sq_dimension - 8; k += 8){
                  
                  X = _mm256_loadu_ps(&sq_matrix_1[i*sq_dimension + k]);
                  Y = _mm256_loadu_ps(&sq_matrix_3[j*sq_dimension + k]);
                  acc = _mm256_add_ps(acc, _mm256_mul_ps(X,Y));
              }
              _mm256_storeu_ps(&temp[0], acc);
              sum = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
              
              
              for (; k < sq_dimension-4; k += 4) {
                  F = _mm_loadu_ps(&sq_matrix_1[i*sq_dimension + k]);
                  N = _mm_loadu_ps(&sq_matrix_3[j*sq_dimension + k]);
                  acc1 = _mm_add_ps(acc1, _mm_mul_ps(F, N));
                  
              }
              _mm_storeu_ps(&temp[0], acc1);
              sum += temp[0]+temp[1]+temp[2]+temp[3];
              
                      
              for (; k < sq_dimension; k++) {
                  sum += sq_matrix_1[i*sq_dimension + k] * sq_matrix_3[j*sq_dimension + k];
              }
              
              sq_matrix_result[i*sq_dimension + j] = sum;
              
          }
      }// End of parallel region
      free(sq_matrix_3);
  }
  
} //namespace omp

/*
  dat1:
 Test Case 1	0.273926 milliseconds
 
 Test Case 2	0.0639648 milliseconds
 
 Test Case 3	0.0378418 milliseconds
 
 Test Case 4	0.0371094 milliseconds

 dat2:
 Test Case 1	0.105957 milliseconds
 
 Test Case 2	0.00805664 milliseconds
 
 Test Case 3	0.019043 milliseconds
 
 Test Case 4	0.0180664 milliseconds
 
 Test Case 5	396.782 milliseconds

 
 */
