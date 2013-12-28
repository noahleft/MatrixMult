#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "matrix_mult_Kernel.cl"
#define TRANSPOSE_FUNC "transpose"
#define MULT_FUNC "matrix_mult"

#define MATRIX_DIM 128

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {

   /* Host/device data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel transpose_kernel, mult_kernel;
   size_t global_size;
   cl_ulong mem_size;
   cl_int i, j, k, err, check;

   /* Data and buffers */
   cl_uint matrix_dim;
   float a_mat[MATRIX_DIM][MATRIX_DIM], b_mat[MATRIX_DIM][MATRIX_DIM], 
         c_mat[MATRIX_DIM][MATRIX_DIM], check_mat[MATRIX_DIM][MATRIX_DIM];
   cl_mem a_buffer, b_buffer, c_buffer;
   
   /* Initialize A, B, and check matrices */
   srand((unsigned int)time(0));
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         a_mat[i][j] = (float)rand()/RAND_MAX;
      }
   }
   srand((unsigned int)time(0));
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         b_mat[i][j] = (float)rand()/RAND_MAX;
         check_mat[i][j] = 0.0f;
      }
   }
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         for(k=0; k<MATRIX_DIM; k++) {
            check_mat[i][j] += a_mat[i][k] * b_mat[k][j];
         }
      }
   }


















   /* Check result */
   check = 1;
   for(i=0; i<MATRIX_DIM; i++) {
      for(j=0; j<MATRIX_DIM; j++) {
         if(fabs(c_mat[i][j] - check_mat[i][j]) > 0.01f) {
            check = 0;
            break;
         }
      }
   }
   if(check)
      printf("Multiplication check succeeded.\n");
   else
      printf("Multiplication check failed.\n");

   /* Deallocate resources */
   clReleaseMemObject(a_buffer);
   clReleaseMemObject(b_buffer);
   clReleaseMemObject(c_buffer);
   clReleaseKernel(mult_kernel);
   clReleaseKernel(transpose_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
