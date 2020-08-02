
__kernel void  executeFirstLayer(__global float *bias, __global float *Layer1_Neurons_GPU, __global float *Layer1_Weights_GPU, __global float *Layer2_Neurons_GPU,__global int *r_offset, __global int *c_offset,__global int *threadx,__global int *thready)
{   
    float product = 0.0;
    int col_width = 227;
    int stride_width = 4;
    int stride = 0,colstride = 0;
    int output = get_group_id(0);
    int tdx = (ulong) threadx[0]; //x dimension
    ulong tdy = (ulong )thready[0]; // y dimension
    uint tid = get_local_id(0);
    int row = (tid / tdx) + (r_offset[0]);
    int col = (tid % tdx)+ (c_offset[0]);
    colstride = 3*row*stride_width*col_width;
    stride = 0;
    product = 0;
    stride = col * 4 * 3;

    /* RGB weights and input 11*11*3 */
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            product +=        ((Layer1_Neurons_GPU[i*227*3 + j*3 + stride + colstride] * Layer1_Weights_GPU[i*11 + j + (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[i*227*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11 + j+ (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[i*227*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11*2 + j+ (output * 11*11*3)]));
        }
    }
    product += bias[output];
    if(product < 0) /* RELU Layer */
        product = 0; // max(0,x)
    Layer2_Neurons_GPU[output*55*55 + row*55 + col] = product;
    
    product = 0.0;
   

}

__kernel void executepoolingCl( __global float *Layer2_Neurons_GPU, __global float *Layer2_pool_GPU,__global int *out, __global int *out_fr,__global int *out_fc,__global int *kernel1, __global int *stride_width, __global int *in_fr,__global int *in_fc,__global int *threadx,__global int *thready )
{
    float max = 0.0;
    int stride = 0,colstride = 0;
    int output = get_group_id(0);
    int tdx = (ulong) threadx[0]; //x dimension
    ulong tdy = (ulong )thready[0]; // y dimension
    int row = (get_local_id(0) / tdx) ;
    int col = (get_local_id(0) % tdx) ;
    colstride = row * (stride_width[0])*(in_fc[0]);
    stride = col * (stride_width[0]);
    for(int i = 0; i < kernel1[0]; i++)
    {
        for(int j = 0; j < kernel1[0]; j++)
        {
            if(max < ((Layer2_Neurons_GPU[(output*in_fr[0]*in_fc[0]) + i*in_fc[0] + j + stride + colstride])))
                max =   ((Layer2_Neurons_GPU[(output*in_fr[0]*in_fc[0]) + i*in_fc[0] + j + stride + colstride])) ;

        }
    }
    Layer2_pool_GPU[output*out_fr[0]*out_fc[0] + row*out_fc[0] + col] = max;
    max = 0.0;
    stride+= stride_width[0];
    
}


__kernel void  execute3DconvolutionCl(__global float *bias, __global float *Layer2_Neurons_GPU, __global float *Layer2_Weights_GPU,__global float *Layer3_Neurons_GPU, __global int *out, __global int *fr,__global int *fc,__global int *stride_width,  __global int *kernel1,__global int *pad,__global int *in_output,__global int *group, __global int *threadx,__global int *thready )
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    int stride = 0,colstride = 0;
    int output = get_group_id(0); // 128
    colstride = 0;
    int tdx = (ulong) threadx[0]; //x dimension
    ulong tdy = (ulong )thready[0]; // y dimension
    int row = (get_local_id(0) / tdx) ;
    stride = 0;
    if(row > pad[0])
       colstride = (row - pad[0]) * (fr[0]);
    int col = (get_local_id(0) % tdx) ;
    if(col >= pad[0])
        stride = col * (stride_width[0]);
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel1[0];loopr = kernel1[0];
    /* take care of padding in left hand side of image*/
    if( row < pad[0])
    {
        x_pad = pad[0] - row;
        loopr = kernel1[0] - x_pad;
    }
    /* take care of padding in upper side of image*/
    if( col < pad[0] )
    {
        y_pad = pad[0] - col;
        loopc = kernel1[0] - y_pad;
    }
    /* take care of padding in right side of image*/
    if(col >= fc[0] - pad[0])
        loopc =  fc[0] + pad[0] - col;
    /* take care of padding in bottom of image */
    if(row >= fr[0] - pad[0])
        loopr =  fr[0] + pad[0] - row;
    for(int feature =0; feature < in_output[0] ; feature++) // calculate the feature maps
    {
        for(int i =0; i < loopr ; i++) // kernel convolution
        {
            for(int j =0; j < loopc ; j++) // kernel convolution
            {
                product += ( Layer2_Neurons_GPU[feature*(fr[0])*(fc[0])+ i*(fc[0]) + j + stride + colstride] * Layer2_Weights_GPU[output*kernel1[0]*kernel1[0]*(in_output[0]) + feature*kernel1[0]*kernel1[0] + i*kernel1[0] + j + kernel1[0]*x_pad + y_pad]);
            }
        }
    }
    product += bias[output];
    if(product < 0) /* ReLU Layer */
        product = 0;
    Layer3_Neurons_GPU[output*(fr[0])*(fc[0]) + row*(fc[0]) + col] = product;
    product = 0.0;
    if(col >= pad[0])
        stride+=stride_width[0];
    
}

__kernel void execute3Dconvolutiongroup2Cl(__global float *bias, __global float *Layer2_Neurons_GPU, __global float *Layer2_Weights_GPU, __global float *Layer3_Neurons_GPU, __global int *out, __global int *fr,__global int *fc,__global int *stride_width, __global int *kernel1, __global int *pad,__global int *in_output,__global int *group, __global int *threadx,__global int *thready )
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    int stride = 0,colstride = 0;
    /* Execute second set of inputs */
    int output = get_group_id(0) + out[0];
    colstride = 0;
    int tdx = (ulong) threadx[0]; //x dimension
    ulong tdy = (ulong )thready[0]; // y dimension
    int row = (get_local_id(0) / tdx) ;
    stride = 0;
    if(row > pad[0])
        colstride = (row - pad[0]) * (fr[0]);
    int col = (get_local_id(0) % tdx) ;
    if(col >= pad[0])
        stride = col*stride_width[0];
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel1[0];loopr = kernel1[0];
    /* take care of padding in left hand side of image*/
    if( row < pad[0])
    {
        x_pad = pad[0] - row;
        loopr = kernel1[0] - x_pad;
    }
    /* take care of padding in upper side of image*/
    if( col < pad[0] )
    {
        y_pad = pad[0] - col;
        loopc = kernel1[0] - y_pad;
    }
    /* take care of padding in right side of image*/
    if(col >= fc[0] - pad[0])
        loopc =  fc[0] + pad[0] - col;
    /* take care of padding in bottom of image */
    if(row >= fr[0] - pad[0])
        loopr =  fr[0] + pad[0] - row;
    for(int feature = in_output[0] ; feature < (in_output[0] << 1) ; feature++) // calculate the feature maps
    {
        for(int i =0; i < loopr ; i++) // kernel1 convolution
        {
            for(int j =0; j < loopc ; j++) // kernel convolution
            {
                product += (( Layer2_Neurons_GPU[feature*(fr[0])*(fc[0]) + i*(fc[0]) + j + stride + colstride] * Layer2_Weights_GPU[output*kernel1[0]*kernel1[0]*(in_output[0]) + (feature-(in_output[0]))*kernel1[0]*kernel1[0] + i*kernel1[0] + j + kernel1[0]*x_pad + y_pad]));
            }
        }
    }
    product += bias[output];
    if(product < 0) /* ReLU Layer */
        product = 0;
    Layer3_Neurons_GPU[output*(fr[0])*(fc[0]) + row*(fc[0]) + col] = product;
    product = 0.0;
    
}
__kernel void  executelrnNormCl_split( __global float *Layer_InNeurons_GPU, __global float *alpha,__global float *beta, __global int *local_size, __global int *out, __global int *fr,__global  int *fc, __global float *Layer_OutNeurons_GPU, __global  int *r_offset, __global int *c_offset, __global int *threadx,__global int *thready )
{
        int nStart = 0, nEnd = 0;
        float value = 0.0;float sum = 0.0;
        int output = get_group_id(0);
        int tdx = (ulong) threadx[0]; //x dimension
        ulong tdy = (ulong )thready[0]; // y dimension
        int row = (get_local_id(0) / tdx) + (r_offset[0]);
        int col = (get_local_id(0) % tdx)+ (c_offset[0]);

        nStart=(output-2) > 1 ? (output-2) : 1 ;
        nEnd=(output+2) <  out[0] ? (output+2) : out[0] ;
        for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution
        {
            sum += pow(( Layer_InNeurons_GPU[i*fr[0]*fc[0] + row*fc[0] + col]),2);
        }
        value = (Layer_InNeurons_GPU[output*fr[0]*fc[0] + row*fc[0] + col]) / (pow( 1 + ((alpha[0]/local_size[0]) *sum),beta[0]));
        sum = 0;
        Layer_OutNeurons_GPU[output*fr[0]*fc[0] + row*fc[0] + col] = value;
    
}

__kernel void  executelrnNormCl( __global float *Layer_InNeurons_GPU, __global float *alpha, __global  float *beta, __global int *local_size, __global int *out, __global int *fr, __global int *fc, __global float *Layer_OutNeurons_GPU,__global int *func_call, __global int *threadx,__global int *thready)
{
        int nStart = 0, nEnd = 0;
        float value = 0.0;float sum = 0.0;
        int output = get_group_id(0);
        int tdx = (ulong) threadx[0]; //x dimension
        ulong tdy = (ulong )thready[0]; // y dimension
         int row = (get_local_id(0) / tdx)+ func_call[0] * 32;
        int col = (get_local_id(0) % tdx) + func_call[0] * 32;
        nStart=(output-2) > 1 ? (output-2) : 1 ;
        nEnd=(output+2) <  out[0] ? (output+2) : out[0] ;
        for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution
        {
            sum += pow(( Layer_InNeurons_GPU[i*fr[0]*fc[0] + row*fc[0] + col]),2);
        }
        value = (Layer_InNeurons_GPU[output*fr[0]*fc[0] + row*fc[0] + col]) / (pow( 1 + ((alpha[0]/local_size[0]) *sum),beta[0]));
        sum = 0;
        Layer_OutNeurons_GPU[output*fr[0]*fc[0] + row*fc[0] + col] = value;
    
}

__kernel void  executeFCLayer(__global float *bias, __global float *Layer_InNeurons_GPU,__global float *Layer_Weights_GPU,__global float *Layer_OutNeurons_GPU,__global int *output, __global int *input,__global int *reLU,__global int *dropout)
{
    float product = 0.0;
    int out = (get_group_id(0) * get_local_size(0)) + get_local_id(0);
    int weight = out * input[0];
    {
        for(int in = 0; in < input[0]; in++)
        {
               product += Layer_InNeurons_GPU[in] * Layer_Weights_GPU[weight+in];
        }
        product += bias[out];
        if(reLU[0] == 1)
        {
            if(product < 0) /* ReLU Layer */
                product = 0;
        }

        Layer_OutNeurons_GPU[out] = product;
        product = 0.0;
    }
}

