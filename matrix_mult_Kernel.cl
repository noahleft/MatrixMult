//__kernel void matrix_mult(__global float4 *a_mat, __global float4 *b_mat, __global float *c_mat) {
__kernel void matrix_mult(__global float *a_mat, __global float *b_mat, __global float *c_mat) {

//    int idx = get_global_id(0);
//    int bias_i=32*(idx/4);
//    int bias_j=32*(idx%4);
    int bias_i=get_global_id(0)<<4;
    int bias_j=get_global_id(1)<<5;
    
    __global float* target_address=c_mat+(bias_i<<7)+bias_j;
    
    for(int x=0,index_a=bias_i<<7 ; x<16 ; x++,index_a+=128) {
        
        for(int y=0,index_b=bias_j<<7 ; y<32 ; y++) {
            
            float tmp=0;
            for(int k=0;k<128;k++,index_b++) {
                tmp+=a_mat[index_a+k] * b_mat[index_b];
            }
            
            *(target_address+y)=tmp;
        }
        
        target_address+=128;
    }
}



__kernel void transpose(__global float *mat,__global float *t_mat) {
    
//    int idx = get_global_id(0);
//    int x = idx/4;
//    int y = idx%4;
    int x = get_global_id(0);
    int y = get_global_id(1);
    //matrix idx start = 32*x + 128*32*y
    
    int bias = (x<<5) + (y<<12);
    int source = (y<<5) + (x<<12);
    for(int i=0;i<32;i++) {
        for(int j=0;j<32;j++) {
            t_mat[bias+(i<<7)+j]=mat[source+(j<<7)+i];
        }
    }
}