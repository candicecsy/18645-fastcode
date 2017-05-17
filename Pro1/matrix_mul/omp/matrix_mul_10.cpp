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

namespace omp
{
  void
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
#pragma omp parallel for
    for (unsigned int i = 0; i < sq_dimension; i++) 
      {
	for(unsigned int j = 0; j < sq_dimension; j++) 
	  {
        double sum = 0.0;
          
	    sq_matrix_result[i*sq_dimension + j] = 0;
          for (unsigned int k = 0; k < sq_dimension; k++){
	      sum += sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
          }
        
        sq_matrix_result[i*sq_dimension + j] = sum;
	  }
          
      }// End of parallel region
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
