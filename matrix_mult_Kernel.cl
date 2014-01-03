//__kernel void matrix_mult(__global float4 *a_mat, __global float4 *b_mat, __global float *c_mat) {
__kernel void matrix_mult(__global float *a_mat, __global float *b_mat, __global float *c_mat) {

    for(int i=0;i<128;i++) {
        for(int j=0;j<128;j++) {
            int index=i*128+j;
            
            float tmp=0;
            for(int k=0;k<128;k++) {
                int index_a=i*128+k;
                int index_b=j*128+k;
                tmp+=a_mat[index_a]*b_mat[index_b];
            }
            c_mat[index]=tmp;
        }
    }

}



__kernel void transpose(__global float *mat,__global float *t_mat) {
    
    int idx = get_global_id(0);
    int x = idx/4;
    int y = idx%4;
    //matrix idx start = 32*x + 128*32*y
    
    int bias = 32*x + 128*32*y;
    int source = 32*y + 128*32*x;
    for(int i=0;i<32;i++) {
        for(int j=0;j<32;j++) {
            t_mat[bias+i*128+j]=mat[source+j*128+i];
        }
    }
}