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
#include <iostream>
#include <fstream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#define BINARY_FILE "OpenCL/matrix_mult_Kernel.cl.gpu_64.bc"
#else
#include <CL/cl.h>
#define BINARY_FILE "matrix_mult_Kernel.bin"
#endif

void buildWithBinary( cl_program &mProgram, cl_context &mContext, const cl_device_id* const mDevice );

int main() {
    
   /* Host/device data structures */
   cl_device_id *device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel transpose_kernel, mult_kernel;
   size_t global_size[2];
   cl_ulong mem_size;
   cl_int i, j, k, err, check;

   /* Data and buffers */
   cl_uint matrix_dim;
   float a_mat[MATRIX_DIM][MATRIX_DIM], b_mat[MATRIX_DIM][MATRIX_DIM], 
         c_mat[MATRIX_DIM][MATRIX_DIM], check_mat[MATRIX_DIM][MATRIX_DIM];
   cl_mem a_buffer, b_buffer, c_buffer;
   cl_mem t_buffer;
   
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

    //a_mat,b_mat store the data. i.e A*B=C
    //check_mat is the result calculate by CPU code
    //c_mat is the result calculate by OpenCL code

    //Get # of platforms at first
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if (err != CL_SUCCESS) {
        std::cerr << "Unable to get platforms\n";
        return 0;
    }

    //Get id of platforms
    std::vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if (err != CL_SUCCESS) {
        std::cerr << "Unable to get paltform ID\n";
        return 0;
    }
    
    //Get Device list
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num);
    device = new cl_device_id[num];
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num, device, NULL);
    
    //Create OpenCL context
    context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
    

    //Get devices info
    size_t cb;
    clGetDeviceInfo(*device, CL_DEVICE_NAME, 0, NULL, &cb);
    std::string devname;
    devname.resize(cb);
    clGetDeviceInfo(*device, CL_DEVICE_NAME, cb, &devname[0], 0);
    std::cout << "Device: " << devname.c_str() << "\n";
    
    //Create command queue
    queue = clCreateCommandQueue(context, device[0], 0, 0);
    if (queue == 0) {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return 0;
    }
    
    //Alloc memory
    a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * MATRIX_DIM * MATRIX_DIM, &a_mat[0], NULL);
    b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * MATRIX_DIM * MATRIX_DIM, &b_mat[0], NULL);
    t_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * MATRIX_DIM * MATRIX_DIM, NULL, NULL);
    c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * MATRIX_DIM * MATRIX_DIM, NULL, NULL);
    if (a_buffer==0 || b_buffer==0 || t_buffer==0 || c_buffer==0) {
        std::cerr << "Can't create OpenCL buffer\n";
        clReleaseMemObject(a_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(c_buffer);
        clReleaseMemObject(t_buffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }
    
    //Load program
    //build program with binary
    //please use program "m2c" amd do not rename matrix_mult_Kernel.cl
    buildWithBinary(program, context, &device[0]);
    if (program == 0) {
        std::cerr << "Can't load or build program\n";
        clReleaseMemObject(a_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(c_buffer);
        clReleaseMemObject(t_buffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }
    
    //Inital kernel
    mult_kernel = clCreateKernel(program, "matrix_mult", 0);
    transpose_kernel = clCreateKernel(program, "transpose", 0);
    if (mult_kernel == 0 || transpose_kernel == 0) {
        std::cerr << "Can't load kernel\n";
        clReleaseProgram(program);
        clReleaseMemObject(a_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(c_buffer);
        clReleaseMemObject(t_buffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }
    
    //Run transpose kernel
    clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), &b_buffer);
    clSetKernelArg(transpose_kernel, 1, sizeof(cl_mem), &t_buffer);
    global_size[0]=4;
    global_size[1]=4;
    err = clEnqueueNDRangeKernel(queue, transpose_kernel, 2, 0, global_size, 0, 0, 0, 0);
    
    //Run matrix multiply kernel
    clSetKernelArg(mult_kernel, 0, sizeof(cl_mem), &a_buffer);
    clSetKernelArg(mult_kernel, 1, sizeof(cl_mem), &t_buffer);
    clSetKernelArg(mult_kernel, 2, sizeof(cl_mem), &c_buffer);
    global_size[0]=8;
    global_size[1]=4;
    err = clEnqueueNDRangeKernel(queue, mult_kernel, 2, 0, global_size, 0, 0, 0, 0);
    if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, sizeof(float) * MATRIX_DIM * MATRIX_DIM, &c_mat[0], 0, 0, 0);
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
   clReleaseMemObject(t_buffer);
   clReleaseKernel(mult_kernel);
   clReleaseKernel(transpose_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}


void buildWithBinary( cl_program &mProgram, cl_context &mContext, const cl_device_id* const mDevice ){
    
    using std::cout;
    using std::endl;
    
    int err_code;
    size_t file_size;
    unsigned char* bin_content;
    std::fstream file;
    
    file.open(BINARY_FILE, std::ios::in );
    file.seekg(0, file.end);
    file_size = file.tellg();
    file.seekg(0, file.beg);
    
    bin_content = new unsigned char[ file_size ];
    file.read( (char*) bin_content, file_size );
    file.close();
    
    mProgram = clCreateProgramWithBinary( mContext, 1, mDevice, &file_size, (const unsigned char**) &bin_content, NULL, &err_code );
    if( err_code != CL_SUCCESS ){
        std::cout<<"Error building program"<<std::endl;
    }
    if( err_code == CL_INVALID_CONTEXT ){
        cout<<file_size<<endl;
    }
    
    delete [] bin_content;
    clBuildProgram( mProgram, 1, &mDevice[0], NULL, NULL, NULL );
}