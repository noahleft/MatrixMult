__kernel void matrix_mult(__global float4 *a_mat, __global float4 *b_mat, __global float4 *c_mat) {

    for(int i=0;i<128;i++) {
        for(int j=0;j<128;j++) {
            int index=i*128+j;
            
            float4 tmp=0;
            for(int k=0;k<128;k++) {
                int index_a=i*128+k;
                int index_b=k*128+j;
                tmp+=a_mat[index_a]*b_mat[index_b];
            }
            c_mat[index]=tmp;
        }
    }

}

