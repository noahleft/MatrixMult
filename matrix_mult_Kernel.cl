//__kernel void matrix_mult(__global float4 *a_mat, __global float4 *b_mat, __global float *c_mat) {
__kernel void matrix_mult(__global float *a_mat, __global float *b_mat, __global float *c_mat) {

//    int idx = get_global_id(0);
//    int bias_i=32*(idx/4);
//    int bias_j=32*(idx%4);
    int bias_i=get_global_id(0)<<4;
    int bias_j=get_global_id(1)<<5;
    
    __global float* target_address=c_mat;
    
    for(int x=0,i=bias_i;x<16;x++,i++) {
        for(int y=0,j=bias_j;y<32;y++,j++) {
            target_address=c_mat+(i<<7)+j;
            
            float tmp=0;
            for(int k=0,index_a=i<<7,index_b=j<<7;k<128;k++,index_a++,index_b++) {
                tmp+=a_mat[index_a]*b_mat[index_b];
            }
            *(target_address)=tmp;
        }
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