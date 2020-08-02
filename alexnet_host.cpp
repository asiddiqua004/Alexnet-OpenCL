    /********************
     * FILENAME :        alexnet_host.cpp
     *
     * DESCRIPTION :
     *       Host side implementation of AlexNet network using OpenCL
     *
     * NOTES :
     *       This file includes OpenCL memory allocations and OpenCL
     *       memory copies to host.
     *       Invokes kernel from host.
     *       Reads inputs and weight from files
     *
     * AUTHOR :    1. Ayesha Siddiqua
     *             2.Ranjana Ajaykumar
     *********************/
    // standard utilities and systems includes
#include <oclUtils.h> // we can change it with time included for reference now
#include <shrQATest.h>
    
    // Name of the file with the source code for the computation kernel
    // *********************************************************************
    const char* SourceFile = "an_kernel.cl"; // Name of the kernel file
    
    // Host buffers for demo
    // *********************************************************************
    
    
    // OpenCL Vars
    cl_platform_id Platform;      // OpenCL platform
    cl_context cxGPUContext;        // OpenCL context
    cl_command_queue cqCommandQueue;// OpenCL command que
    cl_device_id Device;        // OpenCL device list
    cl_uint targetDevice = 0;           // Default Device to compute on
    cl_uint uiNumDevsUsed = 1;      // Number of devices used in this sample
    cl_program cpProgram;           // OpenCL program
    cl_event ceEvent;               // OpenCL event
    cl_int ciErrNum;                // Error code var
    char* cPathAndName = NULL;      // var for full paths to data, src, etc.
    char* SourceCL = NULL;         // Buffer to hold source for compilation
    
    //Alexnet has 5 Convolution layers 3 Max pooling layers and 5 fully connected layers
    //layer parameters
#define INPUT_SIZE 227*227*3
#define L1_KERNEL_SIZE 11*11*3
#define L1_OUT 96
#define L2_KERNEL_SIZE 5*5*48
#define L2_OUT 256
#define L3_KERNEL_SIZE 3*3*256
#define L3_OUT 384
#define L4_KERNEL_SIZE 3*3*192
#define L4_OUT 384
#define L5_KERNEL_SIZE 3*3*192
#define L5_OUT 256
#define L1_FMAP 55*55
#define L2_FMAP 27*27
#define L3_FMAP 13*13
#define L4_FMAP 13*13
#define L5_FMAP 13*13
#define POOL1_FMAP 27*27
#define POOL2_FMAP 13*13
#define POOL3_FMAP 6*6

/// CPU Execution function declaration

void executeFirstLayer(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int stride_width,int col_width,int feature_r,int feature_c,int out);

void pooling(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc);

void execute3Dconvolution(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group);

void executelrnNorm(float *Layer_InNeurons_GPU, float alpha, float beta,int local_size,int out,int fr,int fc,float *Layer_OutNeurons_GPU);

void executeFCLayer(float *bias,float *Layer_InNeurons_GPU,float *Layer_Weights_GPU,float *Layer_OutNeurons_GPU,int output, int input,bool reLU,bool dropout);

void executeSoftMax(float *Layer_In_Neurons_GPU);




    ////////////////////////////////////////////////////////////////////////////////
    // declaration, forward
    extern "C"
    void NeuralNetwork();
    unsigned g_verbose;
    unsigned NUM;
    void extract_weights(const char *pFileName,float *layer_weights,bool bias)
    {
        FILE * pFile1 = fopen (pFileName,"rb");
        char delim[2];
        if(bias == true)
            delim[0] = ' ';
        else
            delim[0] = '\n';
        delim[1] = 0;
        char *token;
        int count = 0;
        char *line = NULL;
        size_t len = 0;
        if (!(pFile1 != NULL))
            printf("File Not Found\n");
        if (pFile1 != NULL && (bias == false))
        {
            printf(" File FOUND %s\n",pFileName);
            {
                //fread(weights,sizeof(weights),1,pFile1);
                //token = strtok(weights,delim);
                //while(token != NULL)
                while (getline(&line, &len, pFile1) != -1)
                {
                    token = strtok(line,delim);
                    float temp_num = atof(token);
                    layer_weights[count] = temp_num;
                    //printf("%.8f\t",temp_num);
                    count++;
                    //    token = strtok(NULL,delim);
                }
            }
            printf("Final Count : %d\n",count);
            fclose(pFile1);
        }
        if (pFile1 != NULL && (bias == true))
        {
            printf(" File FOUND %s\n",pFileName);
            {
                
                char weights[94590] = "";
                fread(weights,sizeof(weights),1,pFile1);
                token = strtok(weights,delim);
                while(token != NULL)
                {
                    float temp_num = atof(token);
                    layer_weights[count] = temp_num;
                    //printf("%.8f\t",temp_num);
                    count++;
                    token = strtok(NULL,delim);
                }
            }
            printf("Final Count : %d\n",count);
            fclose(pFile1);
        }
        
    }
    
    int main(int argc, char** argv)
    {
        int i, commandline_error;
        commandline_error = 0;
        g_verbose = 0;
        if (argc >= 2) {
            NUM = atoi(argv[1]);
            for (i=2; i < argc;i++) {
                if (argv[i][0] == '-') {
                    switch (argv[i][1]) {
                        case 'v': g_verbose = 1;
                            break;
                        default: commandline_error=1;
                    }
                }
                else commandline_error=1;
            }
        } else commandline_error=1;
        if (commandline_error || !NUM) {
            printf("Usage: ./alexnet <NUM> [-v]\n");
            printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
            return 1;
        }
        NeuralNetwork();
    }
    
    void Fill_weights(float *Layer1_Weights_CPU,float *Layer2_Weights_CPU,float *Layer3_Weights_CPU,float *Layer4_Weights_CPU,float *Layer5_Weights_CPU,float *Layer6_Weights_CPU,float *Layer7_Weights_CPU,float *Layer8_Weights_CPU)
    {
        extract_weights("data/conv1.txt",Layer1_Weights_CPU,false);
        extract_weights("data/conv2.txt",Layer2_Weights_CPU,false);
        extract_weights("data/conv3.txt",Layer3_Weights_CPU,false);
        extract_weights("data/conv4.txt",Layer4_Weights_CPU,false);
        extract_weights("data/conv5.txt",Layer5_Weights_CPU,false);
        extract_weights("data/fc6.txt",Layer6_Weights_CPU,false);
        extract_weights("data/fc7.txt",Layer7_Weights_CPU,false);
        extract_weights("data/fc8.txt",Layer8_Weights_CPU,false);
        printf("Extracted Weights and Bias successfully\n");
    }
    void Fill_bias(float *bias_1,float *bias_2,float *bias_3,float *bias_4,float *bias_5,float *bias_6,float *bias_7,float *bias_8)
    {
        extract_weights("data/bias1.txt",bias_1,true);
        extract_weights("data/bias2.txt",bias_2,true);
        extract_weights("data/bias3.txt",bias_3,true);
        extract_weights("data/bias4.txt",bias_4,true);
        extract_weights("data/bias5.txt",bias_5,true);
        extract_weights("data/bias6.txt",bias_6,true);
        extract_weights("data/bias7.txt",bias_7,true);
        extract_weights("data/bias8.txt",bias_8,true);
    }
    void readIn(float *layer1)
    {
        FILE *fp = fopen ("data/input.txt","rb");
        size_t len;
        char delim[1];
        delim[0] = '\n';
        int count = 0;
        char *token;
        char *line = NULL;
        if (fp != NULL)
        {
            printf(" File FOUND\n");
            {
                while ((getline(&line, &len, fp)) != -1)
                {
                    token = strtok(line,delim);
                    layer1[count] = atof(token);
                    count++;
                }
                printf("READ INPUT Final Count :: %d\n",count);
            }
            fclose(fp);
        }
        else
        {
            printf(" File NOt FOUND\n");
        }
    }
    
    void NeuralNetwork()
    {
        //AlexNet architecture changes
        
        /* Read Input File 227*227*3 */
        float *Layer1_Neurons_CPU = (float*) malloc (INPUT_SIZE * sizeof(float));
        readIn(Layer1_Neurons_CPU);
        
        
        /* Declaration of Bias and Weights for CPU */
        float bias_1[96],bias_2[256],bias_3[384],bias_4[384],bias_5[256],bias_6[4096],bias_7[4096],bias_8[1000];
        float *Layer1_Weights_CPU = (float *)malloc(sizeof(float) *(L1_KERNEL_SIZE * L1_OUT));
        float *Layer2_Weights_CPU = (float *)malloc(sizeof(float) *(L2_KERNEL_SIZE * L2_OUT));
        float *Layer3_Weights_CPU = (float *)malloc(sizeof(float) *(L3_KERNEL_SIZE * L3_OUT));
        float *Layer4_Weights_CPU = (float *)malloc(sizeof(float) *(L4_KERNEL_SIZE * L4_OUT));
        float *Layer5_Weights_CPU = (float *)malloc(sizeof(float) *(L5_KERNEL_SIZE * L5_OUT));
        float *Layer6_Weights_CPU = (float *)malloc(sizeof(float) *(4096*256*6*6));
        float *Layer7_Weights_CPU = (float *)malloc(sizeof(float) *(4096*4096));
        float *Layer8_Weights_CPU = (float *)malloc(sizeof(float) *(4096*1000));
        
        /* Fill Bias and Weights */
        Fill_bias(bias_1,bias_2,bias_3,bias_4,bias_5,bias_6,bias_7,bias_8);
        Fill_weights(Layer1_Weights_CPU,Layer2_Weights_CPU,Layer3_Weights_CPU,Layer4_Weights_CPU,Layer5_Weights_CPU,Layer6_Weights_CPU,Layer7_Weights_CPU,Layer8_Weights_CPU);
        
       
        /*************************************************/
        //Get the NVIDIA platform
        printf("Get the Platform ID...\n");
        ciErrNum = clGetPlatformIDs(1,&Platform,NULL);
        if(ciErrNum != CL_SUCCESS)
        {
            fprintf(stderr, "Failed to get platform ID!\n");
            exit(EXIT_FAILURE);
        }
        // Set target device and Query number of compute units on targetDevice
        printf("Set one GPU Device as a target device...\n");
        ciErrNum = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_GPU, 1, &Device, NULL);
        if(ciErrNum != CL_SUCCESS)
        {
            fprintf(stderr, "Failed to get device ID!\n");
            exit(EXIT_FAILURE);
        }
        //Create the context
        printf("clCreateContext...\n");
        cxGPUContext = clCreateContext(NULL, 1, &Device, NULL, NULL, &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {
            fprintf(stderr, "Failed to create context!\n");
            exit(EXIT_FAILURE);
        }
        // Create a command-queue
        printf("clCreateCommandQueue...\n");
        cqCommandQueue = clCreateCommandQueue(cxGPUContext, Device, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {
            fprintf(stderr, "Failed to create command queue!\n");
            exit(EXIT_FAILURE);
        }
        // Read the OpenCL kernel in from source file
        FILE* srcFile = fopen(SourceFile, "r");
        fseek(srcFile, 0, SEEK_END);
        size_t srcSize = ftell(srcFile);
        rewind(srcFile);
        SourceCL = (char*) malloc (srcSize + 1);
        SourceCL[srcSize] = '\0';
        fread(SourceCL, sizeof(char), srcSize, srcFile);
        fclose(srcFile);
        // Create the program
        printf("clCreateProgramWithSource...\n");
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&SourceCL, NULL, &ciErrNum);
        // Build the program
        printf("clBuildProgram...\n");
        ciErrNum = clBuildProgram(cpProgram, 1, &Device,"-cl-fast-relaxed-math", NULL, NULL);
        if (ciErrNum != CL_SUCCESS)
        {
            // write out standard error, Build Log and PTX, then cleanup and exit
            printf("Kernel build failure (%d)\n", ciErrNum);
            size_t log_size;
            clGetProgramBuildInfo(cpProgram, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log  = (char*) malloc(log_size + 1);
            log[log_size] = '\0';
            clGetProgramBuildInfo(cpProgram, Device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("log size = %d: %s\n", log_size, log);
            free(log);
            exit(1);
        }
        // General creation of CL characteristics platform,device, context, command  queue , program ended.
        //////////////////////////////
        cl_kernel ckKernelL1a_1;cl_kernel ckKernelL1b_1; cl_kernel ckKernelL1a_2; cl_kernel ckKernelL1b_2;cl_kernel ckKernelL1a_3; cl_kernel ckKernelL1b_3;
        cl_kernel ckKernelL1_4;cl_kernel ckKernelL1_5;cl_kernel ckKernelL1_6;cl_kernel ckKernelL1_7;
        cl_kernel ckKernelL1a_5;cl_kernel ckKernelL1b_5; cl_kernel ckKernelL1a_6;cl_kernel ckKernelL1b_6; cl_kernel ckKernelL1a_7;cl_kernel ckKernelL1b_7;
        cl_kernel ckKernelL1_8; cl_kernel ckKernelL1_9;
        
        size_t  localWorkSize[2], globalWorkSize[2];
        
        //Implementation on the GPU using OPENCL
        
        /*Layer1 */// Layer1 Neurons -> Layer1_norm -> Layer1_pool -> Layer2_Neurons->
        cl_mem Layer1_bias_GPU, Layer1_Weights_GPU,Layer1_Neurons_GPU,Layer1_Norm_GPU,Layer1_pool_GPU,Layer2_Neurons_GPU;
        cl_mem r_offset, c_offset, threadx, thready;
        int *row_offset, *col_offset, *tdrx, *tdry;
        int row_offseta, col_offseta, tdrxa, tdrya;
        row_offset =&row_offseta; col_offset =&col_offseta; tdrx=&tdrxa; tdry=&tdrya;
        
        // Allocate the OpenCL buffer memory objects for layer1 inputs and result on the device GMEM
        Layer1_Neurons_GPU =    clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* INPUT_SIZE, NULL, &ciErrNum);
        Layer1_Weights_GPU =    clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*L1_KERNEL_SIZE * L1_OUT, NULL, &ciErrNum);
        Layer1_bias_GPU =       clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* L1_OUT, NULL, &ciErrNum);
        Layer1_Norm_GPU =       clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* (L1_OUT * L1_FMAP), NULL, &ciErrNum);
        
        r_offset = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        c_offset = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        threadx = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        thready = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        
        //Load data into the input buffer
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer1_Neurons_GPU, CL_FALSE, 0,sizeof(float)* INPUT_SIZE ,Layer1_Neurons_CPU , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer1_Weights_GPU, CL_FALSE, 0, sizeof(float)*L1_KERNEL_SIZE * L1_OUT,Layer1_Weights_CPU , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer1_bias_GPU , CL_FALSE, 0, sizeof(float)* L1_OUT, bias_1, 0, NULL, NULL);

        /* Output is 96*55*55 , hence launch as 96*32*16 + 96*32*16 + 96*16*23 + 96*16*23 +  96*23*23 */
       //Layer1a_1 ********(32,16)************ offset (0,0)
        size_t threadsPerGrid =96*512; size_t threadsPerBlock =512; tdrxa=32; tdrya=16; row_offseta=0; col_offseta=0;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int), row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,c_offset, CL_FALSE, 0, sizeof(int) , col_offset  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        ckKernelL1a_1 = clCreateKernel(cpProgram,"executeFirstLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
            {    fprintf(stderr, "Failed to create kernel!\n");      exit(EXIT_FAILURE);      }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1a_1,  0, sizeof(cl_mem), &Layer1_bias_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_1,  1, sizeof(cl_mem), &Layer1_Neurons_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_1,  2, sizeof(cl_mem), &Layer1_Weights_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_1,  3, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_1,  4, sizeof(cl_mem), &r_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_1,  5, sizeof(cl_mem), &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_1,  6, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1a_1,  7, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
            {    fprintf(stderr, "Failed to set kernel params!\n");  exit(EXIT_FAILURE);     }
        // Launch kernel
        cl_event event;
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1a_1, 1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to run kernel! %d\n", ciErrNum);     exit(EXIT_FAILURE);     }

        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);

        cl_ulong start_time;
        cl_ulong stop_time;
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        double nanosecond = stop_time-start_time;
        printf ("L1a_1 execution time is %0.3f ms \n",nanosecond/1000000.0);

        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed launch kernel L1a_1\n");    exit(EXIT_FAILURE);     }
        
        ////Layer1b_1 ********(32,16)***********offset (16,0)
        threadsPerGrid =96 * 512; threadsPerBlock =512; tdrxa=32; tdrya=16; row_offseta=16; col_offseta=0;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int), row_offset  , 0, NULL, NULL);
        
       
        ckKernelL1b_1 = clCreateKernel(cpProgram,"executeFirstLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to create kernel L1b_1!\n");     exit(EXIT_FAILURE);     }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1b_1,  0, sizeof(cl_mem), &Layer1_bias_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_1,  1, sizeof(cl_mem), &Layer1_Neurons_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_1,  2, sizeof(cl_mem), &Layer1_Weights_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_1,  3, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_1,  4, sizeof(cl_mem), &r_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1b_1,  5, sizeof(cl_mem), &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1b_1,  6, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1b_1,  7, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to set kernel params L1b_1!\n");     exit(EXIT_FAILURE);    }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1b_1, 1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1b_1! %d\n", ciErrNum);   exit(EXIT_FAILURE);    }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1b_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed launch kernel L1b_1\n"); exit(EXIT_FAILURE);  }
        
        
        //Layer1a_2 ******************(16,23)********** offset (32,0)
        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=16; tdrya=23; row_offseta=32; col_offseta=0;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int), row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
      
        ckKernelL1a_2 = clCreateKernel(cpProgram,"executeFirstLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to create kernel L1a_2!\n");    exit(EXIT_FAILURE);     }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1a_2,  0,sizeof(cl_mem) , &Layer1_bias_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_2,  1,sizeof(cl_mem) , &Layer1_Neurons_GPU);
        ciErrNum|= clSetKernelArg(ckKernelL1a_2,  2,sizeof(cl_mem) , &Layer1_Weights_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_2,  3,sizeof(cl_mem) , &Layer1_Norm_GPU);
        ciErrNum|= clSetKernelArg(ckKernelL1a_2,  4, sizeof(cl_mem), &r_offset);
        ciErrNum|= clSetKernelArg(ckKernelL1a_2,  5, sizeof(cl_mem), &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_2,  6, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1a_2,  7, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        { fprintf(stderr, "Failed to set kernel params L1a_2!\n"); exit(EXIT_FAILURE); }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1a_2,1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1a_2! %d\n", ciErrNum); exit(EXIT_FAILURE);  }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1a_2 execution time is %0.3f ms \n",nanosecond/1000000.0);
         ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        { fprintf(stderr, "Failed launch kernel L1a_2\n"); exit(EXIT_FAILURE);   }
        
        ////Layer1b_2 ******************(16,23)********** offset (32,16)*********
        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=16; tdrya=23; row_offseta=32; col_offseta=16;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, c_offset, CL_FALSE, 0, sizeof(int), col_offset  , 0, NULL, NULL);

        ckKernelL1b_2 = clCreateKernel(cpProgram,"executeFirstLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1b_2!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1b_2,  0,sizeof(cl_mem) , &Layer1_bias_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_2,  1,sizeof(cl_mem) , &Layer1_Neurons_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_2,  2,sizeof(cl_mem) , &Layer1_Weights_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_2,  3,sizeof(cl_mem) , &Layer1_Norm_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_2,  4, sizeof(cl_mem), &r_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1b_2,  5, sizeof(cl_mem), &c_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1b_2,  6, sizeof(cl_mem), &threadx);
         ciErrNum |= clSetKernelArg(ckKernelL1b_2,  7, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1b_2!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1b_2, 1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to run kernel L1b_2! %d\n", ciErrNum);    exit(EXIT_FAILURE); }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1b_2 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {       fprintf(stderr, "Failed launch kernel L1b_2\n");     exit(EXIT_FAILURE);       }

        
       ////Layer1a_3 ************************(23,16)********* offset(0,32)
        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=23; tdrya=16; row_offseta=0; col_offseta=32;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,c_offset, CL_FALSE, 0, sizeof(int) ,col_offset  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        ckKernelL1a_3 = clCreateKernel(cpProgram,"executeFirstLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1a_3!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1a_3,  0,sizeof(cl_mem) , &Layer1_bias_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1a_3,  1,sizeof(cl_mem) , &Layer1_Neurons_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1a_3,  2,sizeof(cl_mem) , &Layer1_Weights_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1a_3,  3,sizeof(cl_mem) , &Layer1_Norm_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1a_3,  4, sizeof(cl_mem), &r_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1a_3,  5, sizeof(cl_mem), &c_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1a_3,  6, sizeof(cl_mem), &threadx);
         ciErrNum |= clSetKernelArg(ckKernelL1a_3,  7, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1a_3!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1a_3,1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to run kernel L1a_3! %d\n", ciErrNum);    exit(EXIT_FAILURE); }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1a_3 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {     fprintf(stderr, "Failed launch kernel L1a_3\n");      exit(EXIT_FAILURE);      }
        
        ////Layer1b_3 *******(23,16)*********offset(16,32)
        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=23; tdrya=16; row_offseta=16; col_offseta=32;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  ,0, NULL, NULL);
        
        ckKernelL1b_3 = clCreateKernel(cpProgram,"executeFirstLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1b_3!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1b_3,  0,sizeof(cl_mem) , &Layer1_bias_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_3,  1,sizeof(cl_mem) , &Layer1_Neurons_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_3,  2,sizeof(cl_mem) , &Layer1_Weights_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_3,  3,sizeof(cl_mem) , &Layer1_Norm_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1b_3,  4, sizeof(cl_mem), &r_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1b_3,  5, sizeof(cl_mem), &c_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1b_3,  6, sizeof(cl_mem), &threadx);
         ciErrNum |= clSetKernelArg(ckKernelL1b_3,  7, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1b_3!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1b_3,1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1b_3! %d\n", ciErrNum);  exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1b_3 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed launch kernel L1b_3\n"); exit(EXIT_FAILURE);    }
        
        
      ////Layer1_4 ******************(23,23)********** offset (32,32)****
        threadsPerGrid =96 * 529; threadsPerBlock =529; tdrxa=23; tdrya=23; row_offseta=32; col_offseta=32;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        ckKernelL1_4 = clCreateKernel(cpProgram,"executeFirstLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1_4!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1_4,  0,sizeof(cl_mem) , &Layer1_bias_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1_4,  1,sizeof(cl_mem) , &Layer1_Neurons_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1_4,  2,sizeof(cl_mem) , &Layer1_Weights_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1_4,  3,sizeof(cl_mem) , &Layer1_Norm_GPU);
         ciErrNum |= clSetKernelArg(ckKernelL1_4,  4, sizeof(cl_mem), &r_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1_4,  5, sizeof(cl_mem), &c_offset);
         ciErrNum |= clSetKernelArg(ckKernelL1_4,  6, sizeof(cl_mem), &threadx);
         ciErrNum |= clSetKernelArg(ckKernelL1_4,  7, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1_4!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1_4,1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1_4! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1_4 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        { fprintf(stderr, "Failed launch kernel L1_4\n"); exit(EXIT_FAILURE);   }
        
        
        /*......................Normalisation ........................*/
        
        Layer1_pool_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* (L1_OUT * L1_FMAP), NULL, &ciErrNum);
        
        //******************(32,16)********** offset (0,0)
        threadsPerGrid =96 * 512; threadsPerBlock =512; tdrxa=32; tdrya=16; row_offseta=0; col_offseta=0;

        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,c_offset, CL_FALSE, 0, sizeof(int) ,col_offset  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        cl_mem alpha, beta;cl_mem local_size, out, fr, fc;
        float *alpha1,*beta1; int *local_size1, *out1, *fr1, *fc1;
        float alph=0.0001; float bet = 0.75; int locala = 5;   int outa   = 96; int fra    = 55;  int fca    = 55;
        alpha1=&alph;beta1=&bet;local_size1=&locala;out1=&outa; fr1=&fra; fc1=&fca;
        
        alpha      = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &ciErrNum);
        beta       = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &ciErrNum);
        local_size = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        out        = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        fr         = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        fc         = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, alpha, CL_FALSE, 0, sizeof(float) ,alpha1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,beta, CL_FALSE, 0, sizeof(float) ,beta1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, local_size, CL_FALSE, 0, sizeof(int) ,local_size1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,out, CL_FALSE, 0, sizeof(int) ,out1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, fr, CL_FALSE, 0, sizeof(int) ,fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fc, CL_FALSE, 0, sizeof(int) ,fc1  ,0, NULL, NULL);
        
        //Create Kernel
        ckKernelL1a_5 = clCreateKernel(cpProgram,"executelrnNormCl_split", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
         {   fprintf(stderr, "Failed to create kernel L1a_5 %d!\n",ciErrNum);    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1a_5,  0, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  1, sizeof(cl_mem), &alpha);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  2, sizeof(cl_mem), &beta);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  3, sizeof(cl_mem),   &local_size);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  4, sizeof(cl_mem),   &out);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  5, sizeof(cl_mem),   &fr);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  6, sizeof(cl_mem),   &fc);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  7, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  8, sizeof(cl_mem),   &r_offset);
        ciErrNum |=clSetKernelArg(ckKernelL1a_5,  9, sizeof(cl_mem),   &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_5,  10, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1a_5,  11, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1a_5 %d!\n", ciErrNum);    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1a_5,1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1a_5! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1a_5 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed launch kernel L1a_5\n");   exit(EXIT_FAILURE);     }
        
        //********************32 x16 **** (16,0)************************
        threadsPerGrid =96 * 512; threadsPerBlock =512; tdrxa=32; tdrya=16; row_offseta=16; col_offseta=0;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  , 0, NULL, NULL);
        
        //Create Kernel
        ckKernelL1b_5 = clCreateKernel(cpProgram,"executelrnNormCl_split", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1b_5!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1b_5,  0, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  1, sizeof(cl_mem), &alpha);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  2, sizeof(cl_mem), &beta);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  3, sizeof(cl_mem),   &local_size);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  4, sizeof(cl_mem),   &out);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  5, sizeof(cl_mem),   &fr);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  6, sizeof(cl_mem),   &fc);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  7, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  8, sizeof(cl_mem),   &r_offset);
        ciErrNum |=clSetKernelArg(ckKernelL1b_5,  9, sizeof(cl_mem),   &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1b_5,  10, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1b_5,  11, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1b_5!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1b_5, 1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1b_5! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1b_5 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed launch kernel L1a_5\n");   exit(EXIT_FAILURE);     }
        
        //*******16x32 ******** (32,0)*******************
      
        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=16; tdrya=23;  row_offseta=32; col_offseta=0;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int), row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,c_offset, CL_FALSE, 0, sizeof(int) , col_offset  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);

        ckKernelL1a_6 = clCreateKernel(cpProgram,"executelrnNormCl_split", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1a_6!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1a_6,  0, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  1, sizeof(cl_mem), &alpha);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  2, sizeof(cl_mem), &beta);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  3, sizeof(cl_mem),   &local_size);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  4, sizeof(cl_mem),   &out);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  5, sizeof(cl_mem),   &fr);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  6, sizeof(cl_mem),   &fc);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  7, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  8, sizeof(cl_mem),   &r_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  9, sizeof(cl_mem),   &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  10, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1a_6,  11, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1a_6!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1a_6, 1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1a_6! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1a_6 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed launch kernel L1a_6\n");    exit(EXIT_FAILURE);    }
        
        //**********16 x23 ****(32,16)**************/
        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=16; tdrya=23; row_offseta=32; col_offseta=16;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdry , 0, NULL, NULL);
        
        ckKernelL1b_6 = clCreateKernel(cpProgram,"executelrnNormCl_split", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1a_6!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1b_6,  0, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  1, sizeof(cl_mem), &alpha);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  2, sizeof(cl_mem), &beta);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  3, sizeof(cl_mem),   &local_size);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  4, sizeof(cl_mem),   &out);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  5, sizeof(cl_mem),   &fr);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  6, sizeof(cl_mem),   &fc);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  7, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  8, sizeof(cl_mem),   &r_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  9, sizeof(cl_mem),   &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  10, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1b_6,  11, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1b_6!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1b_6, 1, NULL, &threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1b_6! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1b_6 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed launch kernel L1b_6\n");    exit(EXIT_FAILURE);    }
        
        //********** 23 x16**** (0,32)  *********
        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=23; tdrya=16; row_offseta=0; col_offseta=32;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,c_offset, CL_FALSE, 0, sizeof(int) ,col_offset  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        

        ckKernelL1a_7 = clCreateKernel(cpProgram,"executelrnNormCl_split", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1a_7!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1a_7,  0, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  1, sizeof(cl_mem), &alpha);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  2, sizeof(cl_mem), &beta);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  3, sizeof(cl_mem),   &local_size);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  4, sizeof(cl_mem),   &out);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  5, sizeof(cl_mem),   &fr);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  6, sizeof(cl_mem),   &fc);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  7, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  8, sizeof(cl_mem),   &r_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  9, sizeof(cl_mem),   &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  10, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1a_7,  11, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1a_7!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
      ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1a_7, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1a_7! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1a_7 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L1a_7\n"); exit(EXIT_FAILURE);
        }
        
        //*************23x16 (16,32)**********************

        threadsPerGrid =96 * 368; threadsPerBlock =368; tdrxa=23; tdrya=16; row_offseta=16; col_offseta=32;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  , 0, NULL, NULL);
        
        ckKernelL1b_7 = clCreateKernel(cpProgram,"executelrnNormCl_split", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1b_7 %d!\n",ciErrNum );    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1b_7,  0, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  1, sizeof(cl_mem), &alpha);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  2, sizeof(cl_mem), &beta);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  3, sizeof(cl_mem),   &local_size);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  4, sizeof(cl_mem),   &out);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  5, sizeof(cl_mem),   &fr);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  6, sizeof(cl_mem),   &fc);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  7, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  8, sizeof(cl_mem),   &r_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  9, sizeof(cl_mem),   &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  10, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1b_7,  11, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1b_7!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1b_7, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1b_7! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1b_7 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L1b_7\n"); exit(EXIT_FAILURE);
        }
        
       
        threadsPerGrid =96 * 529; threadsPerBlock =529; tdrxa=23; tdrya=23; row_offseta=32; col_offseta=32;

        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, r_offset, CL_FALSE, 0, sizeof(int) ,row_offset  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        //**************23x23 (32,32)****************************
        ckKernelL1_8 = clCreateKernel(cpProgram,"executelrnNormCl_split", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
         {   fprintf(stderr, "Failed to create kernel L1_8!\n");    exit(EXIT_FAILURE);   }
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1_8,  0, sizeof(cl_mem), &Layer1_Norm_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  1, sizeof(cl_mem), &alpha);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  2, sizeof(cl_mem), &beta);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  3, sizeof(cl_mem),   &local_size);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  4, sizeof(cl_mem),   &out);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  5, sizeof(cl_mem),   &fr);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  6, sizeof(cl_mem),   &fc);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  7, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  8, sizeof(cl_mem),   &r_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  9, sizeof(cl_mem),   &c_offset);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  10, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1_8,  11, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L1_8!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1_8, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L1_8! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1_8 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed launch kernel L1_8\n");    exit(EXIT_FAILURE); }
        
        /*.................. Max Pool ()........................... */
        
        Layer2_Neurons_GPU= clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*L1_OUT * POOL1_FMAP , NULL, &ciErrNum);
        
        // Block(96,1,1)  Thread(27,27);
        threadsPerGrid =96 * 729; threadsPerBlock =729; tdrxa=27; tdrya=27;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        cl_mem out_fr, out_fc,   kernel1, stride_width, in_fr, in_fc;
        int *out_fr1, *out_fc1, * kernel11, *stride_width1, *in_fr1,  *in_fc1;
        outa   = 96;
        int out_fra = 27;
        int out_fca = 27;
        int kernel11a = 3;
        int stride_width1a = 2;
        int in_fra = 55;
        int in_fc1a = 55;
        out_fr1=&out_fra; out_fc1=&out_fca;   kernel11=&kernel11a; stride_width1=&stride_width1a; in_fr1=&in_fra; in_fc1=&in_fc1a;
        
        out = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        out_fr = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        out_fc = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        kernel1 = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        stride_width = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        in_fr = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        in_fc = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out, CL_FALSE, 0, sizeof(int) ,out1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out_fr, CL_FALSE, 0, sizeof(int) ,out_fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,out_fc, CL_FALSE, 0, sizeof(int) ,out_fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,   kernel1, CL_FALSE, 0, sizeof(int) ,  kernel11  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,stride_width, CL_FALSE, 0, sizeof(int) ,stride_width1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, in_fr, CL_FALSE, 0, sizeof(int) ,in_fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,in_fc, CL_FALSE, 0, sizeof(int) ,in_fc1  ,0, NULL, NULL);
        
        
        ckKernelL1_9 = clCreateKernel(cpProgram,"executepoolingCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L1_9 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL1_9,  0, sizeof(cl_mem), &Layer1_pool_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  1, sizeof(cl_mem), &Layer2_Neurons_GPU);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  2, sizeof(cl_mem), &out);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  3, sizeof(cl_mem), &out_fr);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  4, sizeof(cl_mem), &out_fc);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  5, sizeof(cl_mem),   &kernel1);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  6, sizeof(cl_mem),   &stride_width);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  7, sizeof(cl_mem),   &in_fr);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  8, sizeof(cl_mem),   &in_fc);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  9, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL1_9,  10, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
            {   fprintf(stderr, "Failed to set kernel params L1_9!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
            ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL1_9,  1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
            if(ciErrNum != CL_SUCCESS)
            {    fprintf(stderr, "Failed to run kernel L1_9! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L1_9 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L1_9\n"); exit(EXIT_FAILURE);
        }
        
        clReleaseMemObject(Layer1_Weights_GPU);
        clReleaseMemObject(Layer1_Neurons_GPU);
        clReleaseMemObject(Layer1_Norm_GPU);
        clReleaseMemObject(Layer1_pool_GPU);
        clReleaseKernel(ckKernelL1a_1); clReleaseKernel(ckKernelL1b_1);
        clReleaseKernel(ckKernelL1a_2);clReleaseKernel(ckKernelL1b_2);
        clReleaseKernel(ckKernelL1a_3);clReleaseKernel(ckKernelL1b_3);
        clReleaseKernel(ckKernelL1_4);
        clReleaseKernel(ckKernelL1a_5);clReleaseKernel(ckKernelL1b_5);
        clReleaseKernel(ckKernelL1a_6);clReleaseKernel(ckKernelL1b_6);
        clReleaseKernel(ckKernelL1a_7);clReleaseKernel(ckKernelL1b_7);
        clReleaseKernel(ckKernelL1_8);
        clReleaseKernel(ckKernelL1_9);
        
        
        /*************** Second Layer convolution + ReLU + pooling *************************/
        cl_mem Layer2_bias_GPU,Layer2_Weights_GPU,Layer2_Norm_GPU,Layer2_pool_GPU,Layer3_Neurons_GPU;

        Layer2_Weights_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*L2_KERNEL_SIZE * L2_OUT, NULL, &ciErrNum);
        Layer2_bias_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* L2_OUT, NULL, &ciErrNum);
        Layer2_Norm_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* (L2_OUT * L2_FMAP), NULL, &ciErrNum);
        
       
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer2_Weights_GPU, CL_FALSE, 0,sizeof(float)*(L2_KERNEL_SIZE * L2_OUT) , Layer2_Weights_CPU, 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer2_bias_GPU , CL_FALSE, 0,sizeof(float)* L2_OUT,bias_2 , 0, NULL, NULL);
        
        cl_kernel ckKernelL2_1; cl_kernel ckKernelL2_2; cl_kernel ckKernelL2_3;
        cl_kernel ckKernelL2_4;
        
        /* Group = 2 and each group is 128*27*27 */
        //Block(128,1,1) Thread(27,27);
       threadsPerGrid =128 * 729; threadsPerBlock =729; tdrxa=27; tdrya=27;
        cl_mem  pad, in_output, group;
        int *pad1, *in_output1, *group1;
        int pada;
        int in_outputa1;
        int groupa;
        pada = 2;
        in_outputa1 = 48;
        groupa = 2;
        outa   = 128;
        fra = 27;
        fca = 27;
        kernel11a = 5;
        stride_width1a = 1;
        pad1 = &pada;
        in_output1 = &in_outputa1;
        group1 = &groupa;
        
        pad = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        in_output = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        group = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out, CL_FALSE, 0, sizeof(int) ,out1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fr, CL_FALSE, 0, sizeof(int) ,fr1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fc, CL_FALSE, 0, sizeof(int) ,fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,   kernel1, CL_FALSE, 0, sizeof(int) ,  kernel11  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,stride_width, CL_FALSE, 0, sizeof(int) ,stride_width1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, pad, CL_FALSE, 0, sizeof(int) ,pad1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,in_output, CL_FALSE, 0, sizeof(int) ,in_output1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,group, CL_FALSE, 0, sizeof(int) ,group1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        ckKernelL2_1 = clCreateKernel(cpProgram,"execute3DconvolutionCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L2_1 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL2_1,  0, sizeof(cl_mem), &Layer2_bias_GPU);
        ciErrNum = clSetKernelArg(ckKernelL2_1,  1, sizeof(cl_mem), &Layer2_Neurons_GPU);
        ciErrNum = clSetKernelArg(ckKernelL2_1,  2, sizeof(cl_mem), &Layer2_Weights_GPU);
        ciErrNum = clSetKernelArg(ckKernelL2_1,  3,sizeof(cl_mem) , &Layer2_Norm_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  7, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  8, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  9, sizeof(cl_mem), &pad);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  10, sizeof(cl_mem), &in_output);
        ciErrNum |=clSetKernelArg(ckKernelL2_1,  11, sizeof(cl_mem), &group);
        ciErrNum |= clSetKernelArg(ckKernelL2_1,  12, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL2_1,  13, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L2_1!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL2_1, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L2_1! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L2_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L2_1\n"); exit(EXIT_FAILURE);
        }

        ckKernelL2_2 = clCreateKernel(cpProgram,"execute3Dconvolutiongroup2Cl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L2_2 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL2_2,  0, sizeof(cl_mem), &Layer2_bias_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  1, sizeof(cl_mem), &Layer2_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  2, sizeof(cl_mem), &Layer2_Weights_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  3,sizeof(cl_mem) , &Layer2_Norm_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  7, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  8, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  9, sizeof(cl_mem), &pad);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  10, sizeof(cl_mem), &in_output);
        ciErrNum |=clSetKernelArg(ckKernelL2_2,  11, sizeof(cl_mem), &group);
        ciErrNum |= clSetKernelArg(ckKernelL2_2,  12, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL2_2,  13, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L2_2!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL2_2, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L2_2! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L2_2 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L2_2\n"); exit(EXIT_FAILURE);
        }
        
        /********************Normalisation *****************/

        Layer2_pool_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* (L2_OUT * L2_FMAP), NULL, &ciErrNum);
        
        //Block(256,1,1) Thread(27,27);
        threadsPerGrid =256 * 729; threadsPerBlock =729; tdrxa=27; tdrya=27;
       
        cl_mem func_call; int *func_call1;
        alph=0.0001;
        bet = 0.75;
        locala = 5;
        outa   = 256;
        fra    = 27;
        fca    = 27;
        int func_call1a=0;
        func_call1=&func_call1a;
        func_call = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, alpha, CL_FALSE, 0, sizeof(float) ,alpha1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,beta, CL_FALSE, 0, sizeof(float) ,beta1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, local_size, CL_FALSE, 0, sizeof(int) ,local_size1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,out, CL_FALSE, 0, sizeof(int) ,out1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, fr, CL_FALSE, 0, sizeof(int) ,fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fc, CL_FALSE, 0, sizeof(int) ,fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,func_call, CL_FALSE, 0, sizeof(int) ,func_call1,0, NULL, NULL);
        
     
        ckKernelL2_3 = clCreateKernel(cpProgram,"executelrnNormCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L2_3 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL2_3,  0, sizeof(cl_mem), &Layer2_Norm_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  1, sizeof(cl_mem), &alpha);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  2, sizeof(cl_mem), &beta);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  3, sizeof(cl_mem), &local_size);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  7,sizeof(cl_mem), &Layer2_pool_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_3,  8, sizeof(cl_mem), &func_call);
        ciErrNum |= clSetKernelArg(ckKernelL2_3,  9, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL2_3,  10, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L2_3!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL2_3,  1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L2_3! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L2_3 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L2_3\n"); exit(EXIT_FAILURE);
        }
        
        /******************* Max Pool *****************/
     
        Layer3_Neurons_GPU  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* L2_OUT * POOL2_FMAP, NULL, &ciErrNum);
        
        //dim3 pool2_Block(256,1,1);//dim3 pool2_Thread(13,13);
        threadsPerGrid =256 * 169; threadsPerBlock =169; tdrxa=13; tdrya=13;
        outa   = 256;
        out_fra = 13;
        out_fca = 13;
        kernel11a = 3;
        stride_width1a = 2;
        in_fra = 27;
        in_fc1a = 27;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out, CL_FALSE, 0, sizeof(int) ,out1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out_fr, CL_FALSE, 0, sizeof(int) ,out_fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,out_fc, CL_FALSE, 0, sizeof(int) ,out_fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,   kernel1, CL_FALSE, 0, sizeof(int) ,  kernel11  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,stride_width, CL_FALSE, 0, sizeof(int) ,stride_width1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, in_fr, CL_FALSE, 0, sizeof(int) ,in_fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,in_fc, CL_FALSE, 0, sizeof(int) ,in_fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        ckKernelL2_4 = clCreateKernel(cpProgram,"executepoolingCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L2_4 %d!\n",ciErrNum );}
        
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL2_4,  0, sizeof(cl_mem), &Layer2_pool_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  1, sizeof(cl_mem), &Layer3_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  2, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  3, sizeof(cl_mem), &out_fr);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  4, sizeof(cl_mem), &out_fc);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  5, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  6, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  7, sizeof(cl_mem), &in_fr);
        ciErrNum |=clSetKernelArg(ckKernelL2_4,  8, sizeof(cl_mem), &in_fc);
        ciErrNum |= clSetKernelArg(ckKernelL2_4,  9, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL2_4,  10, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L2_4!\n");    exit(EXIT_FAILURE);  }
        
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL2_4, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L2_4! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L2_4 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L2_4\n"); exit(EXIT_FAILURE);
        }
        
        clReleaseMemObject(Layer2_bias_GPU);
        clReleaseMemObject(Layer2_Weights_GPU);
        clReleaseMemObject(Layer2_Neurons_GPU);
        clReleaseMemObject(Layer2_Norm_GPU);
        clReleaseMemObject(Layer2_pool_GPU);
        
        clReleaseKernel(ckKernelL2_1);
        clReleaseKernel(ckKernelL2_2);
        clReleaseKernel(ckKernelL2_3);
        clReleaseKernel(ckKernelL2_4);
        
        /* Third Layer convolution + ReLU  */
        cl_mem Layer3_bias_GPU,Layer3_Weights_GPU,Layer4_Neurons_GPU;
        
        Layer3_Weights_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*L3_KERNEL_SIZE * L3_OUT, NULL, &ciErrNum);
        Layer3_bias_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* L3_OUT, NULL, &ciErrNum);
        Layer4_Neurons_GPU  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* L3_OUT * L3_FMAP, NULL, &ciErrNum);
        
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer3_Weights_GPU, CL_FALSE, 0,sizeof(float)*(L3_KERNEL_SIZE * L3_OUT) , Layer3_Weights_CPU, 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer3_bias_GPU , CL_FALSE, 0,sizeof(float)* L3_OUT,bias_3 , 0, NULL, NULL);
        
        //Block(384,1,1) Thread(13,13);
        threadsPerGrid =384 * 169; threadsPerBlock =169; tdrxa=13; tdrya=13;
        
        int fr1a,fc1a;
        outa   = 384;
        fr1a=13;
        fc1a=13;
        kernel11a = 3;
        stride_width1a = 1;
        pada = 1;
        in_outputa1=256;
        groupa=1;
        fr1=&fr1a;fc1=&fc1a;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out, CL_FALSE, 0, sizeof(int) ,out1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fr, CL_FALSE, 0, sizeof(int) ,fr1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fc, CL_FALSE, 0, sizeof(int) ,fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,   kernel1, CL_FALSE, 0, sizeof(int) ,  kernel11  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,stride_width, CL_FALSE, 0, sizeof(int) ,stride_width1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, pad, CL_FALSE, 0, sizeof(int) ,pad1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,in_output, CL_FALSE, 0, sizeof(int) ,in_output1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,group, CL_FALSE, 0, sizeof(int) ,group1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        cl_kernel ckKernelL3_1;
        
        
        ckKernelL3_1 = clCreateKernel(cpProgram,"execute3DconvolutionCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L3_1 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL3_1,  0, sizeof(cl_mem), &Layer3_bias_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  1, sizeof(cl_mem), &Layer3_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  2, sizeof(cl_mem), &Layer3_Weights_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  3, sizeof(cl_mem), &Layer4_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  7, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  8, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  9, sizeof(cl_mem), &pad);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  10, sizeof(cl_mem), &in_output);
        ciErrNum |=clSetKernelArg(ckKernelL3_1,  11, sizeof(cl_mem), &group);
        ciErrNum |= clSetKernelArg(ckKernelL3_1,  12, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL3_1,  13, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L3_1!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL3_1, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L3_1! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L3_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS)
        { fprintf(stderr, "Failed launch kernel L3_1\n"); exit(EXIT_FAILURE); }
        
        clReleaseMemObject(Layer3_bias_GPU);
        clReleaseMemObject(Layer3_Weights_GPU);
        clReleaseMemObject(Layer3_Neurons_GPU);
        clReleaseKernel(ckKernelL3_1);
        
        /* Fourth Layer convolution + ReLU  */
        cl_mem Layer4_bias_GPU,Layer4_Weights_GPU,Layer5_Neurons_GPU;
        
       
        Layer4_Weights_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*L4_KERNEL_SIZE * L4_OUT, NULL, &ciErrNum);
        Layer4_bias_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* L4_OUT, NULL, &ciErrNum);
        Layer5_Neurons_GPU  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* L4_OUT * L4_FMAP, NULL, &ciErrNum);
        
       
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer4_Weights_GPU, CL_FALSE, 0,sizeof(float)*(L4_KERNEL_SIZE * L4_OUT) , Layer4_Weights_CPU, 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer4_bias_GPU , CL_FALSE, 0,sizeof(float)* L4_OUT,bias_4 , 0, NULL, NULL);
        
        //Block(192,1,1) Thread(13,13);
        threadsPerGrid =192 * 169; threadsPerBlock =169; tdrxa=13; tdrya=13;
        
        outa=192;  fr1a=13;  fc1a=13;  stride_width1a=1;  kernel11a=3;  pada=1;  in_outputa1=192;  groupa=2;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out, CL_FALSE, 0, sizeof(int) ,out1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fr, CL_FALSE, 0, sizeof(int) ,fr1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fc, CL_FALSE, 0, sizeof(int) ,fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,stride_width, CL_FALSE, 0, sizeof(int) ,stride_width1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, kernel1, CL_FALSE, 0, sizeof(int) ,kernel11  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, pad, CL_FALSE, 0, sizeof(int) ,pad1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,in_output, CL_FALSE, 0, sizeof(int) ,in_output1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,group, CL_FALSE, 0, sizeof(int) ,group1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        cl_kernel ckKernelL4_1; cl_kernel ckKernelL4_2;
       
        ckKernelL4_1 = clCreateKernel(cpProgram,"execute3DconvolutionCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L4_1 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL4_1,  0, sizeof(cl_mem), &Layer4_bias_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  1, sizeof(cl_mem), &Layer4_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  2, sizeof(cl_mem), &Layer4_Weights_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  3, sizeof(cl_mem), &Layer5_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  7, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  8, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  9, sizeof(cl_mem), &pad);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  10, sizeof(cl_mem), &in_output);
        ciErrNum |=clSetKernelArg(ckKernelL4_1,  11, sizeof(cl_mem), &group);
        ciErrNum |= clSetKernelArg(ckKernelL4_1,  12, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL4_1,  13, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L4_1!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL4_1, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L4_1! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L4_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L4_1\n"); exit(EXIT_FAILURE);
        }
        
        
        ckKernelL4_2 = clCreateKernel(cpProgram,"execute3Dconvolutiongroup2Cl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L4_2 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL4_2,  0, sizeof(cl_mem), &Layer4_bias_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  1, sizeof(cl_mem), &Layer4_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  2, sizeof(cl_mem), &Layer4_Weights_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  3, sizeof(cl_mem), &Layer5_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  7, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  8, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  9, sizeof(cl_mem), &pad);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  10, sizeof(cl_mem), &in_output);
        ciErrNum |=clSetKernelArg(ckKernelL4_2,  11, sizeof(cl_mem), &group);
        ciErrNum |= clSetKernelArg(ckKernelL4_2,  12, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL4_2,  13, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L4_2!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL4_2, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L2_2! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L4_2 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L4_2\n"); exit(EXIT_FAILURE);
        }
        
        clReleaseMemObject(Layer4_bias_GPU);
        clReleaseMemObject(Layer4_Weights_GPU);
        clReleaseMemObject(Layer4_Neurons_GPU);
        
        clReleaseKernel(ckKernelL4_1);
        clReleaseKernel(ckKernelL4_2);
        
        /* Fifth Layer convolution + ReLU + pooling */
        cl_mem Layer5_bias_GPU,Layer5_Weights_GPU,Layer5_pool_GPU,Layer6_Neurons_GPU;
        
       
        Layer5_Weights_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*L5_KERNEL_SIZE * L5_OUT, NULL, &ciErrNum);
        Layer5_bias_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* L5_OUT,NULL , &ciErrNum);
        Layer5_pool_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* (L5_OUT * L5_FMAP), NULL, &ciErrNum);
        
 
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer5_Weights_GPU, CL_FALSE, 0,sizeof(float)*(L5_KERNEL_SIZE * L5_OUT) , Layer5_Weights_CPU, 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer5_bias_GPU , CL_FALSE, 0,sizeof(float)* L5_OUT,bias_5 , 0, NULL, NULL);
        
        //Block(128,1,1) Thread(13,13);
        threadsPerGrid =128 * 169; threadsPerBlock =169; tdrxa=13; tdrya=13;
        
        outa=128;  fr1a=13;  fc1a=13;  stride_width1a=1;  kernel11a=3;  pada=1;  in_outputa1=192;  groupa=2;
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out, CL_FALSE, 0, sizeof(int) ,out1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fr, CL_FALSE, 0, sizeof(int) ,fr1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,fc, CL_FALSE, 0, sizeof(int) ,fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,stride_width, CL_FALSE, 0, sizeof(int) ,stride_width1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, kernel1, CL_FALSE, 0, sizeof(int) ,kernel11  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, pad, CL_FALSE, 0, sizeof(int) ,pad1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,in_output, CL_FALSE, 0, sizeof(int) ,in_output1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,group, CL_FALSE, 0, sizeof(int) ,group1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);
        
        cl_kernel ckKernelL5_1; cl_kernel ckKernelL5_2; cl_kernel ckKernelL5_3;
        
        ckKernelL5_1 = clCreateKernel(cpProgram,"execute3DconvolutionCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L5_1 %d!\n",ciErrNum );}
        
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL5_1,  0, sizeof(cl_mem), &Layer5_bias_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  1, sizeof(cl_mem), &Layer5_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  2, sizeof(cl_mem), &Layer5_Weights_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  3, sizeof(cl_mem), &Layer5_pool_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  7, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  8, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  9, sizeof(cl_mem), &pad);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  10, sizeof(cl_mem), &in_output);
        ciErrNum |=clSetKernelArg(ckKernelL5_1,  11, sizeof(cl_mem), &group);
        ciErrNum |= clSetKernelArg(ckKernelL5_1,  12, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL5_1,  13, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L5_1!\n");    exit(EXIT_FAILURE);  }
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL5_1, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L5_1! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L5_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L5_1\n"); exit(EXIT_FAILURE);
        }

        ckKernelL5_2 = clCreateKernel(cpProgram,"execute3Dconvolutiongroup2Cl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L5_2 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL5_2,  0, sizeof(cl_mem), &Layer5_bias_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  1, sizeof(cl_mem), &Layer5_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  2, sizeof(cl_mem), &Layer5_Weights_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  3, sizeof(cl_mem), &Layer5_pool_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  4, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  5, sizeof(cl_mem), &fr);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  6, sizeof(cl_mem), &fc);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  7, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  8, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  9, sizeof(cl_mem), &pad);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  10, sizeof(cl_mem), &in_output);
        ciErrNum |=clSetKernelArg(ckKernelL5_2,  11, sizeof(cl_mem), &group);
        ciErrNum |= clSetKernelArg(ckKernelL5_2,  12, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL5_2,  13, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L5_2!\n");    exit(EXIT_FAILURE);  }
        
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL5_2, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L5_2! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L5_2 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L5_2\n"); exit(EXIT_FAILURE);
        }
        
         //***** POOL3_FMAP AND POOL5_FMAP SWAP *****
        Layer6_Neurons_GPU  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* L5_OUT * POOL3_FMAP, NULL, &ciErrNum);
        
        //Block(256,1,1) Thread(6,6);
       threadsPerGrid =256 * 36; threadsPerBlock =36; tdrxa=6; tdrya=6;
        outa   = 256;
        out_fra = 6;
        out_fca = 6;
        kernel11a = 3;
        stride_width1a = 2;
        in_fra = 13;
        in_fc1a = 13;
        
        out1=&outa;
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out, CL_FALSE, 0, sizeof(int) ,out1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, out_fr, CL_FALSE, 0, sizeof(int) ,out_fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,out_fc, CL_FALSE, 0, sizeof(int) ,out_fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, kernel1, CL_FALSE, 0, sizeof(int) ,kernel11  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,stride_width, CL_FALSE, 0, sizeof(int) ,stride_width1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, in_fr, CL_FALSE, 0, sizeof(int) ,in_fr1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,in_fc, CL_FALSE, 0, sizeof(int) ,in_fc1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,threadx , CL_FALSE, 0, sizeof(int),  tdrx , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,thready, CL_FALSE, 0, sizeof(int) , tdry  ,0, NULL, NULL);

        
        ckKernelL5_3 = clCreateKernel(cpProgram,"executepoolingCl", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L5_3 %d!\n",ciErrNum );}
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL5_3,  0, sizeof(cl_mem), &Layer5_pool_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  1, sizeof(cl_mem), &Layer6_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  2, sizeof(cl_mem), &out);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  3, sizeof(cl_mem), &out_fr);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  4, sizeof(cl_mem), &out_fc);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  5, sizeof(cl_mem), &kernel1);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  6, sizeof(cl_mem), &stride_width);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  7, sizeof(cl_mem), &in_fr);
        ciErrNum |=clSetKernelArg(ckKernelL5_3,  8, sizeof(cl_mem), &in_fc);
        ciErrNum |= clSetKernelArg(ckKernelL5_3,  9, sizeof(cl_mem), &threadx);
        ciErrNum |= clSetKernelArg(ckKernelL5_3,  10, sizeof(cl_mem), &thready);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L5_3 = %d!\n",ciErrNum);    exit(EXIT_FAILURE);  }
        
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL5_3, 1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L5_3! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L5_3 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L5_3\n"); exit(EXIT_FAILURE);
        }
        
        clReleaseMemObject(Layer5_bias_GPU);
        clReleaseMemObject(Layer5_Weights_GPU);
        clReleaseMemObject(Layer5_Neurons_GPU);
        clReleaseMemObject(Layer5_pool_GPU);
        clReleaseKernel(ckKernelL5_2);
        clReleaseKernel(ckKernelL5_3);
        
        /* Sixth Layer Fully connected + ReLU */
        cl_mem Layer6_bias_GPU, Layer6_Weights_GPU, Layer7_Neurons_GPU;
    
        Layer6_Weights_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*4096*256*6*6, NULL, &ciErrNum);
        Layer6_bias_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* 4096, NULL, &ciErrNum);
        Layer7_Neurons_GPU  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* 4096, NULL, &ciErrNum);
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer6_Weights_GPU, CL_FALSE, 0, sizeof(float)*4096*256*6*6 , Layer6_Weights_CPU, 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer6_bias_GPU, CL_FALSE, 0,sizeof(float)* 4096,bias_6 , 0, NULL, NULL);
        
        //Block(4096,1,1) Thread(1,1)
        threadsPerGrid =4096; threadsPerBlock =1024;
        cl_mem output, input; cl_mem reLU, dropout;
        int *output1, *input1; int *reLU1, *dropout1;
        int output1a=4096;int input1a=256*6*6;int reLU1a=1;int dropout1a=0;
        output1=&output1a; input1=&input1a; reLU1=&reLU1a;dropout1=&dropout1a;
        
        output = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        input = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        reLU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        dropout = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &ciErrNum);
        
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, output, CL_FALSE, 0, sizeof(int) ,output1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,input, CL_FALSE, 0, sizeof(int) ,input1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,reLU, CL_FALSE, 0, sizeof(int) ,reLU1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,dropout, CL_FALSE, 0, sizeof(int) ,dropout1  ,0, NULL, NULL);
        
        cl_kernel ckKernelL6_1;

        
        ckKernelL6_1 = clCreateKernel(cpProgram,"executeFCLayer", &ciErrNum);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to create kernel L6_1 %d!\n",ciErrNum );}
        // Set the Argument values
        
        ciErrNum = clSetKernelArg(ckKernelL6_1,  0, sizeof(cl_mem), &Layer6_bias_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL6_1,  1, sizeof(cl_mem), &Layer6_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL6_1,  2, sizeof(cl_mem), &Layer6_Weights_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL6_1,  3, sizeof(cl_mem), &Layer7_Neurons_GPU);
        ciErrNum |=clSetKernelArg(ckKernelL6_1,  4, sizeof(cl_mem), &output);
        ciErrNum |=clSetKernelArg(ckKernelL6_1,  5, sizeof(cl_mem), &input);
        ciErrNum |=clSetKernelArg(ckKernelL6_1,  6, sizeof(cl_mem), &reLU);
        ciErrNum |=clSetKernelArg(ckKernelL6_1,  7, sizeof(cl_mem), &dropout);
        if(ciErrNum != CL_SUCCESS)
        {   fprintf(stderr, "Failed to set kernel params L6_1!\n");    exit(EXIT_FAILURE);  }

        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL6_1,1, NULL,&threadsPerGrid, &threadsPerBlock , 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L6_1! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L6_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L6_1\n"); exit(EXIT_FAILURE);
        }
        
        clReleaseKernel(ckKernelL6_1);
        
        /* Seventh Layer Fully connected + ReLU */
        cl_mem Layer7_bias_GPU, Layer7_Weights_GPU,Layer8_Neurons_GPU;
        
        Layer7_Weights_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*4096*4096,NULL , &ciErrNum);
        Layer7_bias_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* 4096, NULL, &ciErrNum);
        Layer8_Neurons_GPU  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* 4096, NULL, &ciErrNum);
        
 
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer7_Weights_GPU, CL_FALSE, 0, sizeof(float)*4096*4096 , Layer7_Weights_CPU, 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer7_bias_GPU , CL_FALSE, 0,sizeof(float)* 4096,bias_7 , 0, NULL, NULL);
        
        
        // Block(4096,1,1)  Thread(1,1)
        localWorkSize[0] = 1; localWorkSize[1] = 1; globalWorkSize[0] = 4096; globalWorkSize[1] = 1;
        
        output1a=4096;input1a=4096;reLU1a=1;dropout1a=0;
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, output, CL_FALSE, 0, sizeof(int) ,output1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,input, CL_FALSE, 0, sizeof(int) ,input1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,reLU, CL_FALSE, 0, sizeof(int) ,reLU1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,dropout, CL_FALSE, 0, sizeof(int) ,dropout1  ,0, NULL, NULL);
        
        cl_kernel ckKernelL7_1;
       
        ckKernelL7_1 = clCreateKernel(cpProgram,"executeFCLayer", &ciErrNum);
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL7_1,  0, sizeof(cl_mem), &Layer7_bias_GPU);
        ciErrNum = clSetKernelArg(ckKernelL7_1,  1, sizeof(cl_mem), &Layer7_Neurons_GPU);
        ciErrNum = clSetKernelArg(ckKernelL7_1,  2, sizeof(cl_mem), &Layer7_Weights_GPU);
        ciErrNum = clSetKernelArg(ckKernelL7_1,  3, sizeof(cl_mem), &Layer8_Neurons_GPU);
        ciErrNum = clSetKernelArg(ckKernelL7_1,  4, sizeof(cl_mem), &output);
        ciErrNum = clSetKernelArg(ckKernelL7_1,  5, sizeof(cl_mem), &input);
        ciErrNum = clSetKernelArg(ckKernelL7_1,  6, sizeof(cl_mem), &reLU);
        ciErrNum = clSetKernelArg(ckKernelL7_1,  7, sizeof(cl_mem), &dropout);
        
        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL7_1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L7_1! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L7_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L7_1\n"); exit(EXIT_FAILURE);
        }
        
        
        /* Eigth Layer Fully connected + ReLU */
        cl_mem Layer8_bias_GPU, Layer9_Neurons_GPU, Layer8_Weights_GPU;

        Layer8_Weights_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)*4096*1000, NULL, &ciErrNum);
        Layer8_bias_GPU = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(float)* 1000, NULL, &ciErrNum);
        Layer9_Neurons_GPU  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,  sizeof(float)* 1000, NULL, &ciErrNum);

        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, Layer8_Weights_GPU, CL_FALSE, 0, sizeof(float)*4096*1000 , Layer8_Weights_CPU, 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,Layer8_bias_GPU , CL_FALSE, 0,sizeof(float)* 1000,bias_8 , 0, NULL, NULL);
        
        // Block(1000,1,1) Thread(1,1)
        localWorkSize[0] = 1; localWorkSize[1] = 1; globalWorkSize[0] = 1000; globalWorkSize[1] = 1;
        
        output1a=1000;input1a=4096;reLU1a=0;dropout1a=0;
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, output, CL_FALSE, 0, sizeof(int) ,output1  , 0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,input, CL_FALSE, 0, sizeof(int) ,input1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,reLU, CL_FALSE, 0, sizeof(int) ,reLU1  ,0, NULL, NULL);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue,dropout, CL_FALSE, 0, sizeof(int) ,dropout1  ,0, NULL, NULL);
        cl_kernel ckKernelL8_1;
    
        ckKernelL8_1 = clCreateKernel(cpProgram,"executeFCLayer", &ciErrNum);
        // Set the Argument values
        ciErrNum = clSetKernelArg(ckKernelL8_1,  0, sizeof(cl_mem), &Layer8_bias_GPU);
        ciErrNum = clSetKernelArg(ckKernelL8_1,  1, sizeof(cl_mem), &Layer8_Neurons_GPU);
        ciErrNum = clSetKernelArg(ckKernelL8_1,  2, sizeof(cl_mem), &Layer8_Weights_GPU);
        ciErrNum = clSetKernelArg(ckKernelL8_1,  3, sizeof(cl_mem), &Layer9_Neurons_GPU);
        ciErrNum = clSetKernelArg(ckKernelL8_1,  4, sizeof(cl_mem), &output);
        ciErrNum = clSetKernelArg(ckKernelL8_1,  5, sizeof(cl_mem), &input);
        ciErrNum = clSetKernelArg(ckKernelL8_1,  6, sizeof(cl_mem), &reLU);
        ciErrNum = clSetKernelArg(ckKernelL8_1,  7, sizeof(cl_mem), &dropout);

        // Launch kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernelL8_1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);
        if(ciErrNum != CL_SUCCESS)
        {    fprintf(stderr, "Failed to run kernel L8_1! %d\n",  ciErrNum);   exit(EXIT_FAILURE);     }
        clWaitForEvents(1,&event);
        clFinish(cqCommandQueue);
  
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(start_time),&start_time,NULL);
        clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(stop_time),&stop_time,NULL);

        nanosecond = stop_time-start_time;
        printf ("L8_1 execution time is %0.3f ms \n",nanosecond/1000000.0);
        ciErrNum =clEnqueueBarrier(cqCommandQueue);
        if(ciErrNum != CL_SUCCESS){
            fprintf(stderr, "Failed launch kernel L8_1\n"); exit(EXIT_FAILURE);
        }
        
        float *fc9_Neurons_CPU = (float *)malloc(sizeof(float) * (1000));
        
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue,Layer9_Neurons_GPU, CL_TRUE, 0, sizeof(float)*(1000), fc9_Neurons_CPU,0, NULL, NULL);
        if(ciErrNum != CL_SUCCESS)
        {
            fprintf(stderr, "Failed to read output data %d!\n", ciErrNum);
            exit(EXIT_FAILURE);
        }
        
        /* Check the output */
        float max = 0.0;int index = 0;
        for(int i =0; i < 1000; i++)
        {
            if(max < fc9_Neurons_CPU[i])
            {
                max = fc9_Neurons_CPU[i];
                index = i;
            }
        }
        printf("INDEX = %d\n",index);

        printf("Starting Cleanup...\n");
        free(SourceCL);
        clReleaseProgram(cpProgram);
        clReleaseCommandQueue(cqCommandQueue);
        clReleaseContext(cxGPUContext);
        
        
///******************
        /* if CPU = 1 then CPU version of code ,else OpenCl code */
             //******************
    #ifdef CPU
        /* First Layer convolution + ReLU + pooling */
        float *Layer2_Neurons_CPU = (float *)malloc(sizeof(float) *(96*55*55));
        executeFirstLayer(bias_1,Layer1_Neurons_CPU,Layer1_Weights_CPU,Layer2_Neurons_CPU,4,227,55,55,96);
        /*Normalisation */
        float *Layer2_Norm_CPU = (float *)malloc(sizeof(float) *(96*55*55));
        executelrnNorm(Layer2_Neurons_CPU,0.0001,0.75,5,96,55,55,Layer2_Norm_CPU);
        /* Max Pool */
        float *Layer2_pool_CPU = (float *)malloc(sizeof(float) *(96*27*27));
        pooling(Layer2_Norm_CPU,Layer2_pool_CPU,96,27,27,3,2,55,55);
        
        /* Second Layer */
        float *Layer3_Neurons_CPU = (float *)malloc(sizeof(float) *(256*27*27));
        execute3Dconvolution(bias_2,Layer2_pool_CPU,Layer2_Weights_CPU,Layer3_Neurons_CPU,256,27,27,1,5,2,96,2);
        /*Normalisation */
        float *Layer3_Norm_CPU = (float *)malloc(sizeof(float) *(256*27*27));
        executelrnNorm(Layer3_Neurons_CPU,0.0001,0.75,5,256,27,27,Layer3_Norm_CPU);
        /* Max Pool */
        float *Layer3_pool_CPU = (float *)malloc(sizeof(float) *(256*13*13));
        pooling(Layer3_Norm_CPU,Layer3_pool_CPU,256,13,13,3,2,27,27);
        
        /* Third Layer convolution + ReLU  */
        float *Layer4_Neurons_CPU = (float *)malloc(sizeof(float) *(384*13*13));
        execute3Dconvolution(bias_3,Layer3_pool_CPU,Layer3_Weights_CPU,Layer4_Neurons_CPU,384,13,13,1,3,1,256,1);
        
        /* Fourth Layer convolution + ReLU  */
        float *Layer5_Neurons_CPU = (float *)malloc(sizeof(float) *(384*13*13));
        execute3Dconvolution(bias_4,Layer4_Neurons_CPU,Layer4_Weights_CPU,Layer5_Neurons_CPU,384,13,13,1,3,1,384,2);
        
        /* Fifth Layer convolution + ReLU + pooling */
        float *fc6_Neurons_CPU = (float *)malloc(sizeof(float) *(256*13*13));
        execute3Dconvolution(bias_5,Layer5_Neurons_CPU,Layer5_Weights_CPU,fc6_Neurons_CPU,256,13,13,1,3,1,384,2);
        float *fc6_pool_CPU = (float *)malloc(sizeof(float) *(256*6*6));
        pooling(fc6_Neurons_CPU,fc6_pool_CPU,256,6,6,3,2,13,13);
        
        /* Sixth Layer Fully connected + ReLU */
        float *fc7_Neurons_CPU = (float *)malloc(sizeof(float) * (4096));
        executeFCLayer(bias_6,fc6_pool_CPU,Layer6_Weights_CPU,fc7_Neurons_CPU,4096,(256*6*6),true,true);
        
        /* Seventh Layer Fully connected + ReLU */
        float *fc8_Neurons_CPU = (float *)malloc(sizeof(float) * (4096));
        executeFCLayer(bias_7,fc7_Neurons_CPU,Layer7_Weights_CPU,fc8_Neurons_CPU,4096,4096,true,true);
        
        /*Eigth Layer */
        float *fc9cpu_Neurons_CPU = (float *)malloc(sizeof(float) * (1000));
        executeFCLayer(bias_8,fc8_Neurons_CPU,Layer8_Weights_CPU,fc9cpu_Neurons_CPU,1000,4096,false,false);
        
        /* Check the output */
        float max1 = 0.0;int index1 = 0;
        for(int i =0; i < 1000; i++)
        {
            if(max1 < fc9cpu_Neurons_CPU[i])
            {
                max1 = fc9cpu_Neurons_CPU[i];
                index1 = i;
            }
        }
        printf("INDEX from CPU = %d\n",index1);
        
#endif

        
        /////////////
        
        
        free(Layer1_Weights_CPU);
        free(Layer2_Weights_CPU);
        free(Layer3_Weights_CPU);
        free(Layer4_Weights_CPU);
        free(Layer5_Weights_CPU);
        free(Layer6_Weights_CPU);
        free(Layer7_Weights_CPU);
        free(Layer8_Weights_CPU);
         printf("Enf of Cleanup...\n");
        /* SoftMax */
        //Confirm the functionality of SoftMax ,extract_weights("data/fc8_out.txt",fc9_Neurons_CPU,false);
        //executeSoftMax(fc9_Neurons_CPU);
        exit(0);
    }

void executeFirstLayer(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int stride_width,int col_width,int feature_r,int feature_c,int out)
{
    float product = 0.0;
    int stride = 0,colstride = 0;
    {
        for(int output =0;output < out ;output++)
        {
            for(int row =0; row < feature_r ;row++)
            {
                colstride = 3*row*stride_width*col_width;
                stride = 0;
                for(int col =0; col < feature_c ;col++)
                {
                    product = 0;
                    /* RGB weights and input 11*11*3 , kernel is 11*11 */
                    for(int i = 0; i < 11; i++)
                    {
                        for(int j = 0; j < 11; j++)
                        {
                            product +=        ((Layer1_Neurons_GPU[i*col_width*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*11 + j + (output * 11*11*3)])
                                               + (Layer1_Neurons_GPU[i*col_width*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11 + j+ (output * 11*11*3)])
                                               + (Layer1_Neurons_GPU[i*col_width*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11*2 + j+ (output * 11*11*3)]));
                        }
                    }
                    product += bias[output];
                    if(product < 0) /* RELU Layer */
                        product = 0; // max(0,x)
                    Layer2_Neurons_GPU[output*feature_r*feature_c + row*feature_c + col] = product;
#ifdef LAYER1_DEBUG
                    printf("%f\n",product);
#endif
                    product = 0.0;
                    stride+= stride_width*3;
                }
            }
        }
    }
}
void pooling(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    printf("pooling Activation layer \n");
    float max = 0.0;
    int downsample = 0;
    int stride = 0,colstride = 0;
    {
        for(int output =0;output < out ;output++)
        {
            for(int row =0; row < out_fr ;row++)
            {
                colstride = row * stride_width*in_fc;
                stride = 0;
                for(int col =0; col < out_fc ;col++)
                {
                    for(int i = 0; i < kernel; i++)
                    {
                        for(int j = 0; j < kernel; j++)
                        {
                            if(max < ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])))
                                max =   ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])) ;
                            //          if(output == 141)
                            //              printf("%f %d\t",Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride],((output*in_fr*in_fc) + i*in_fc + j + stride + colstride)) ;
                            
                        }
                    }
                    Layer2_pool_GPU[downsample] = max;
#ifdef POOL_DEBUG
                    printf("\n %f %d\n",max,downsample);
#endif
                    max = 0.0;
                    downsample++;
                    stride+= stride_width;
                }
            }
        }
    }
}

void execute3Dconvolution(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group)
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    printf(" 3D convolution with group %d,output %d,feature %d x %d ,stride %d, kernel %d, pad %d, input %d\n",group,out,fr,fc,stride_width,kernel,pad,in_output);
    if(group == 2)
    {
        out = out >> 1;
        in_output = in_output >> 1;
    }
    int stride = 0,colstride = 0;
    {
        for(int output =0;output < out  ;output++) /* out = 256 */
        {
            colstride = 0;
            for(int row =0; row < fr ; row++) /* out = 256 */
            {
                stride = 0;
                if(row > pad)
                    colstride = (row - pad) * fr;
                for(int col =0; col < fc ;col++) /* out = 256 */
                {
                    x_pad = 0; y_pad = 0;
                    /* set the loops value */
                    loopc = kernel;loopr = kernel;
                    /* take care of padding in left hand side of image*/
                    if( row < pad)
                    {
                        x_pad = pad - row;
                        loopr = kernel - x_pad;
                    }
                    /* take care of padding in upper side of image*/
                    if( col < pad )
                    {
                        y_pad = pad - col;
                        loopc = kernel - y_pad;
                    }
                    /* take care of padding in right side of image*/
                    if(col >= fc - pad)
                        loopc =  fc + pad - col;
                    /* take care of padding in bottom of image */
                    if(row >= fr - pad)
                        loopr =  fr + pad - row;
                    for(int feature =0; feature < in_output ; feature++) // calculate the feature maps
                    {
                        for(int i =0; i < loopr ; i++) // kernel convolution
                        {
                            for(int j =0; j < loopc ; j++) // kernel convolution
                            {
                                product += ( Layer2_Neurons_GPU[feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + feature*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]);
                            }
                        }
                    }
                    product += bias[output];
                    if(product < 0) /* ReLU Layer */
                        product = 0;
#ifdef LAYER2_DEBUG
                    printf("%f\n",product);
#endif
                    //                    if((group == 2) && (out == 128) && (in_output == 192))
                    //                        printf("%f\n",product);
                    Layer3_Neurons_GPU[output*fr*fc + row*fc + col] = product;
                    product = 0.0;
                    if(col >= pad)
                        stride+=stride_width;
                }
            }
            
        }
        if(group == 2)
        {
            /* Execute second set of inputs */
            for(int output = out ;output < (out << 1)   ;output++) /* out = 256 */
            {
                colstride = 0;
                for(int row =0; row < fr; row++) /* out = 256 */
                {
                    stride = 0;
                    if(row > pad)
                        colstride = (row - pad) * fr;
                    for(int col =0; col < fc ;col++) /* out = 256 */
                    {
                        x_pad = 0; y_pad = 0;
                        /* set the loops value */
                        loopc = kernel;loopr = kernel;
                        /* take care of padding in left hand side of image*/
                        if( row < pad)
                        {
                            x_pad = pad - row;
                            loopr = kernel - x_pad;
                        }
                        /* take care of padding in upper side of image*/
                        if( col < pad )
                        {
                            y_pad = pad - col;
                            loopc = kernel - y_pad;
                        }
                        /* take care of padding in right side of image*/
                        if(col >= fc - pad)
                            loopc =  fc + pad - col;
                        /* take care of padding in bottom of image */
                        if(row >= fr - pad)
                            loopr =  fr + pad - row;
                        for(int feature = in_output ; feature < (in_output << 1) ; feature++) // calculate the feature maps
                        {
                            for(int i =0; i < loopr ; i++) // kernel convolution
                            {
                                for(int j =0; j < loopc ; j++) // kernel convolution
                                {
                                    product += (( Layer2_Neurons_GPU[feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + (feature-in_output)*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]));
                                }
                            }
                        }
                        product += bias[output];
                        if(product < 0) /* ReLU Layer */
                            product = 0;
#ifdef LAYER2_DEBUG
                        printf("%f\n",product);
#endif
                        //                        if((group == 2) && (out == 128) && (in_output == 192))
                        //                            printf("%f\n",product);
                        Layer3_Neurons_GPU[output*fr*fc + row*fc + col] = product;
                        product = 0.0;
                        if(col >= pad)
                            stride+=stride_width;
                    }
                }
                
            }
        }
        
    }
}
void executelrnNorm(float *Layer_InNeurons_GPU, float alpha, float beta,int local_size,int out,int fr,int fc,float *Layer_OutNeurons_GPU)
{
    printf(" Exexcute Norm Layer\n");
    int nStart = 0, nEnd = 0;
    float value = 0.0;float sum = 0.0;
    for(int row =0; row < fr; row++)
    {
        for(int col =0; col < fc ;col++)
        {
            for(int output = 0 ;output < out   ;output++)
            {
                nStart=(output-floor(local_size/2)) > 1 ? (output-floor(local_size/2)) : 1 ;
                nEnd=(output+floor(local_size/2)) <  out ? (output+floor(local_size/2)) : out ;
                for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution
                {
                    sum += pow(( Layer_InNeurons_GPU[i*fr*fc + row*fc + col]),2);
                }
                value = (Layer_InNeurons_GPU[output*fr*fc + row*fc + col]) / (pow( 1 + ((alpha/local_size) *sum),beta));
                sum = 0;
                Layer_OutNeurons_GPU[output*fr*fc + row*fc + col] = value;
            }
        }
        
    }
#ifdef NORM_LAYER
    for(int N = 0; N < out; N++)
    {
        
        for(int W = 0; W < fr; W++)
        {
            for(int H = 0; H < fc; H++)
            {
                printf("%f\n",Layer_OutNeurons_GPU[N*fr*fc + W*fc + H]);;
            }
        }
    }
#endif
}
void executeFCLayer(float *bias,float *Layer_InNeurons_GPU,float *Layer_Weights_GPU,float *Layer_OutNeurons_GPU,int output, int input,bool reLU,bool dropout)
{
    printf("Execute FC Layer of output : %d input %d\n",output,input);
    float product = 0.0,max = 0.0; int weight = 0,index = 0;
    for(int out=0; out < output ; out++)
    {
        for(int in = 0; in < input; in++)
        {
            product += Layer_InNeurons_GPU[in] * Layer_Weights_GPU[weight++];
        }
        product += bias[out];
        if(reLU == true)
        {
            if(product < 0) /* ReLU Layer */
                product = 0;
        }
        else
        {
            if(max < product)
            {
                index = out;
                max = product;
            }
        }
        if(dropout == true)
        {
            
        }
        Layer_OutNeurons_GPU[out] = product;
#ifdef FC_DEBUG
        printf("%f\n",product);
#endif
        product = 0.0;
    }
    printf(" MAX from FC layer = %d\n",index);
}

void executeSoftMax(float *Layer_In_Neurons_GPU)
{
    printf("executeSoftMax \n");
    float max = 0,sum = 0;
    float output[1000] = {0};
    for(int i = 0; i < 1000; i++)
    {
        if(Layer_In_Neurons_GPU[i] > max)
            max = Layer_In_Neurons_GPU[i];
    }
#ifdef SOFTMAX_DEBUG
    printf("Max = %10e\n",max);
#endif
    for(int i = 0; i < 1000; i++)
    {
        output[i] = exp(Layer_In_Neurons_GPU[i] - max);
        sum += output[i];
    }
#ifdef SOFTMAX_DEBUG
    printf("Sum =  %10e\n",sum);
#endif
    for(int i = 0; i < 1000; i++)
    {
        output[i] *= (1/sum);
#ifdef SOFTMAX_DEBUG
        printf("%10e\n",output[i]);
#endif
    }
    
}

