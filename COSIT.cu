/**********************************************************************************************
 * This code provides 2 CUDA Optimised Statistical Inference Techniques: Jackknife and Blocked Bootstrap.
 * Statistical Inference is an integral part of a huge range of data analysis and applications are far reaching.
 * Both techniques use resampling as a means of computing statistics of interest from an original data set.
 * In this code I look at the mean value of the data set, and derive Jackknife and Blocked Bootstrap estimates for it's error.
 * In addition to the CUDA kernels, I have supplied CPU versions of the program, and an openMP version of the Jackknife.
 * Several Bootstrapping alternative kernels are given, with improved accuracy (due to less bias) but slower execution time.
 * 	CUDA Jackknife:
 * Compute the mean using "reduction" kernel.
 * Compute the standard error using a second "squaresreduction" kernel.
 * 	CUDA Blocked Bootstrap:
 * Block the data set using the same "reduction" kernel, or "reduction2" if smaller bins are required.
 * Compute the Bootstrap resamples of the blocked data in "bootstrap"
 ***********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// cutil_inline.h includes the elapsed_time() function which records time since previous elapsed_time() call
#include "cutil_inline.h"
//#include "params.h"
//params.h contains the parameters required. (These are #defined below)
#include <cuda.h>
#include <curand.h>
#include <omp.h>

#define TBS 256 //this is the threads per block for the jackknife reduction kernel
#define ELS 10240000 //number of elements
#define NTHREADS 13 //only valid for Open MP Jackknife, dependent on the system number of cores available
#define EPB 128 //elements per bin in bootstrap
#define ELPB 128 //threads per block (elements per bootstrap) in bootstrap
#define BOOTS 16384 //number of bootstraps
// see main() below for what happens when each of the following is 0/1
#define CPU_JACK 0
#define OMP_JACK 0
#define GPU_JACK 0
#define CPU_BOOT 1
#define LOOP_BOOT 0 //loop_boot=1 and gpu_boot=1 causes us to loop over different bin sizes
#define GPU_BOOT 1
#define BSTRAP 1 //this is which bootstrap method we run, 1 is the quickest but has some bias, 3 is slowest and has no bias, 2 is between 1 and 3.



__constant__ int num_els;
__constant__ float sum;
__constant__ float mean;
__constant__ int bins;

// The reduction takes two elements during load, so requires half the expected number of blocks (els/blockDim). (This taken care of on the host)
// The reduction also uses templates and unrolling as in the Mark Harris Tutorial http://people.maths.ox.ac.uk/gilesm/cuda/prac4/reduction.pdf

// ************************************* Jackknife Kernels ************************************** //

template <unsigned int blockSize>
__global__ void reduction(float *g_idata, float *g_odata) {

extern __shared__ volatile float sdata[];
// each thread loads two elements from global to shared mem, and does an addition in the loading
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x + gridDim.x*blockDim.x*blockIdx.y*2;
sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
__syncthreads();
// perform reduction on shared mem (unrolled loops)
if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid<32) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
// transfer results back to global memory
if (tid == 0) g_odata[blockIdx.x+gridDim.x*blockIdx.y] = sdata[0];
}

// Another reduction for the square sums does calculations while loading in the data

template <unsigned int blockSize>
__global__ void squaresreduction(float *g_idata, float *g_odata) {
extern __shared__ volatile float sdata[];
// each thread loads two elements from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x + gridDim.x*blockDim.x*blockIdx.y*2;
sdata[tid] = (((sum-g_idata[i])/(num_els-1.0f))-mean)*(((sum-g_idata[i])/(num_els-1.0f))-mean)
		+ (((sum-g_idata[i+blockDim.x])/(num_els-1.0f))-mean)*(((sum-g_idata[i+blockDim.x])/(num_els-1.0f))-mean);
//sdata[tid] = (((sum-g_idata[i])/(num_els-1.0f))-mean)*(((sum-g_idata[i])/(num_els-1.0f))-mean);


__syncthreads();

// do reduction in shared mem
if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid<32) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// write result for this block to global memory
if (tid == 0) g_odata[blockIdx.x+gridDim.x*blockIdx.y] = sdata[0];
}

// ************************************* Jackknife CPU CODE ************************************** //

// CPU jackknife calculates the mean of the data array; then finds the jackknife mean corresponding to each element; then computes the standard deviation using the formula found on http://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0CC8QFjAA&url=http%3A%2F%2Fwww.int.washington.edu%2Ftalks%2FWorkShops%2Fint_07_2b%2FPeople%2FJoo_B%2FJoo5.pdf&ei=fSwfUq-_C8qu7AaApoCADA&usg=AFQjCNFFo_WXYNNuhQGv5-maSee4JR3tKQ&bvm=bv.51495398,d.ZGU

void cpu_jackknife (float *data_array, unsigned int len,double &cpu_calcs){
  double timeSinceLastTimer, timerCPU;
  elapsed_time(&timerCPU);

  float *data;
  data = (float*) malloc (len*sizeof(float));

  float cpu_sd=0.0f;
  float cpu_sum=0.0f;

  timeSinceLastTimer = elapsed_time(&timerCPU);
  printf("\nTime to declare CPU variables:\t%f \n",timeSinceLastTimer);

	  for (int i = 0; i<len; i++)
	    {
	    cpu_sum += data_array[i];
	    }
	  for (int i = 0; i<len; i++)
	    {
		  data[i]=(cpu_sum-data_array[i])/(len-1);
	    }
	  cpu_sum/=len;
	  for (int i = 0; i<len; i++)
	    {
	    cpu_sd += (data[i]-cpu_sum)*(data[i]-cpu_sum);
	    }
	  cpu_sd=sqrt((len-1.0f)*cpu_sd/(len));

  timeSinceLastTimer = elapsed_time(&timerCPU);
  cpu_calcs=timeSinceLastTimer;
 printf("Time to complete CPU calculations:\t%f \n",timeSinceLastTimer);

//  printf("CPU:\n");
  printf("CPU sd = \t%f\n",cpu_sd);
  printf("CPU mean = \t%f\n",cpu_sum);
//  free (data);

//  timeSinceLastTimer = elapsed_time(&timerCPU);
//  printf("Time to free data used in CPU calcs:\t%f \n",timeSinceLastTimer);

}

// ************************************* Jackknife Open MP CODE ************************************** //

void omp_jackknife (float *data_array , unsigned int len , double &omp_calcs){

// Compute the jackknife using openMP methods: using reductions for both the mean and sd calculations, and using a schedule directive for modifying every element of the data array to be the jackknife mean corresponding to that element.

  double timeSinceLastTimer, timerOMP;
  elapsed_time(&timerOMP);

  float *data;
  data = (float*) malloc (len*sizeof(float));

  float omp_sd=0.0f;
  float omp_sum=0.0f;

  int chunk=len/NTHREADS;
//int chunk=1000 ;
  int i;
  int nThreads=NTHREADS;
omp_set_num_threads(nThreads);
  printf("\nUsing nThreads=%d\nChunk size=%d\n",nThreads,chunk);
  timeSinceLastTimer = elapsed_time(&timerOMP);
  printf("Time to declare OMP variables:\t%f \n",timeSinceLastTimer);

#pragma omp parallel for reduction(+:omp_sum)
         for (i = 0; i< len; i++)
           {
              (omp_sum) = omp_sum + data_array[i];
           }

#pragma omp parallel for schedule ( static,chunk)
         for (i = 0; i< len; i++)
           {
                data[i]=((omp_sum)-data_array [i])/(len -1.0f);
           }
#pragma omp parallel for reduction(+:omp_sd)
         for (i = 0; i< len; i++)
           {
           (omp_sd) = omp_sd + ( data[i]-(omp_sum)/len )*(data[i]-(omp_sum)/len);
           }


   (omp_sd)=sqrt(( len-1.0f)*(omp_sd)/(len));
   (omp_sum)=omp_sum/ len;


  timeSinceLastTimer = elapsed_time(&timerOMP);
  omp_calcs=timeSinceLastTimer;
 printf("Time to complete OMP calculations:\t%f \n",timeSinceLastTimer);

//  printf("CPU:\n");
  printf("OMP sd = \t%f\n",omp_sd);
  printf("OMP mean = \t%f\n",omp_sum);
//  free (data);

//  timeSinceLastTimer = elapsed_time(&timerOMP);
//  printf("Time to free data used in CPU calcs:\t%f \n",timeSinceLastTimer);


}

// **************** Determine grid dimension for kernels in both BS and JK ***************** //
// if our grid-x-dimension exceeds 65536 the kernel fails, so introduce a grid-y-dimension to stop this happening.
// grid-y-dimension must be chosen suitably to maintain the same total grid dimension
void determine_grid (int blocknum, int &num_blocks_y){

  if (blocknum>65536) {
  if (blocknum%2==0 & blocknum < 2*65536) num_blocks_y=2;
  else if (blocknum%3==0 & blocknum < 3*65536) num_blocks_y=3;
  else if (blocknum%4==0 & blocknum < 4*65536) num_blocks_y=4;
  else if (blocknum%5==0 & blocknum < 5*65536) num_blocks_y=5;
  else if (blocknum%8==0 & blocknum < 8*65536) num_blocks_y=8;
  else if (blocknum%10==0 & blocknum < 10*65536) num_blocks_y=10;
  else if (blocknum%16==0 & blocknum < 16*65536) num_blocks_y=16;
  else if (blocknum%32==0 & blocknum < 32*65536) num_blocks_y=32;
  else if (blocknum%64==0 & blocknum < 64*65536) num_blocks_y=64;
  else if (blocknum%100==0 & blocknum < 100*65536) num_blocks_y=100;
  else if (blocknum%128==0 & blocknum < 128*65536) num_blocks_y=128;
  else if (blocknum%200==0 & blocknum < 200*65536) num_blocks_y=200;
  else if (blocknum%256==0 & blocknum < 256*65536) num_blocks_y=256;
  else if (blocknum%500==0 & blocknum < 500*65536) num_blocks_y=500;
  else if (blocknum%512==0 & blocknum < 512*65536) num_blocks_y=512;
  else if (blocknum%1000==0 & blocknum < 1000*65536) num_blocks_y=1000;
  else if (blocknum%1024==0 & blocknum < 1024*65536) num_blocks_y=1024;
  else printf("Too many blocks have been submitted, add more possibilities for num_blocks_y");
  }
  if(blocknum/1024>=65536) printf("Too many blocks submitted: increase num_blocks_y");

}

// ************************************* Jackknife GPU CODE ************************************** //

void gpu_jackknife (float *data_array, unsigned int num_elements, double &gpu_calcs){

  // Declare GPU variables (blocks,threadblocks,memory e.t.c)
  double timeSinceLastTimer, timerGPU, timeRed;
  elapsed_time(&timerGPU);

  int num_blocks, mem_size, shared_mem_size, block_mem_size;
  int num_threadsperblock=TBS;
  num_blocks=(int) ceil((float) num_elements/num_threadsperblock);

// Compute the number of blocks from #els/#threads per block: then compute half this to find the actual number of blocks required in our kernal (because each thread reads in two elements)
// Reqnum_elements is the number of elements we have to give to the kernal
  int Reqnum_elements=num_threadsperblock*(int) ceil((float) num_blocks/2)*2;

  float gpu_sum=0.0f, gpu_mean;
  float gpu_sd=0.0f;

  float *data;
  float *d_idata, *d_odata;
  float *partial_sums;

  int halved_blocknum = (int) ceil((float) num_blocks/2);
  // to avoid problems with unspecified launch failures (too many x-dimension blocks), find a suitable value for num_blocks_y:
  int num_blocks_y=1;

  determine_grid (halved_blocknum,num_blocks_y);

  dim3 dimGrid(halved_blocknum/num_blocks_y,num_blocks_y);

  block_mem_size = halved_blocknum * sizeof(float);
  mem_size     = Reqnum_elements * sizeof(float);
  shared_mem_size = num_threadsperblock * sizeof(float);

  printf("\nnum_elements=%d\n",num_elements);
  printf("Reqnum_elements=%d\n",Reqnum_elements);
  printf("dimGrid(%d,%d)\n",halved_blocknum/num_blocks_y,num_blocks_y);
  printf("num_threadsperblock=%d\n\n",num_threadsperblock);

//if we dont need to zero pad, we can save a lot of time by not creating a new array
  if (Reqnum_elements-num_elements!=0){

    data= (float*) malloc (mem_size);
    partial_sums = (float*) malloc (block_mem_size);
    for(int i = 0; i < num_elements; i++) data[i] = data_array[i];
    for(int i = num_elements; i < Reqnum_elements; i++) data[i]=0.0f;

  // allocate device memory input and output arrays

    cudaSafeCall(cudaMalloc((void**)&d_idata, mem_size));
    cudaSafeCall(cudaMalloc((void**)&d_odata, block_mem_size));

    // copy host data to device input array

    cudaSafeCall(cudaMemcpy(d_idata, data, mem_size, cudaMemcpyHostToDevice));
    free(data);
  }
  else {
    partial_sums = (float*) malloc (block_mem_size);
    cudaSafeCall(cudaMalloc((void**)&d_idata, mem_size));
    cudaSafeCall(cudaMalloc((void**)&d_odata, block_mem_size));
    cudaSafeCall(cudaMemcpy(d_idata, data_array, mem_size, cudaMemcpyHostToDevice));
  }

  cudaSafeCall(cudaDeviceSynchronize());
  timeSinceLastTimer = elapsed_time(&timerGPU);
  printf("1: Time to declare gpu variables and swap memory:\t%f \n",timeSinceLastTimer);

  // First execute the reduction kernel to find the sum of the data (Use switch case and templates to speed up the execution)

  switch (num_threadsperblock)
  {
  case 1024:
  	  reduction<1024><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 512:
	  reduction<512><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 256:
	  reduction<256><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 128:
	  reduction<128><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 64:
	  reduction<64><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 32:
	  reduction<32><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 16:
	  reduction<16><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 8:
	  reduction<8><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 4:
	  reduction<4><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 2:
	  reduction<2><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 1:
	  reduction<1><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  }
  cudaCheckMsg("reduction kernel execution failed");
  cudaSafeCall(cudaDeviceSynchronize());
  timeRed = elapsed_time(&timerGPU);
  printf("2: Time to run reduction:\t%f \n",timeRed);

// Copy partial sums back and sum them
  cudaSafeCall(cudaMemcpy(partial_sums, d_odata, block_mem_size, cudaMemcpyDeviceToHost));
  for (int i = 0 ; i<halved_blocknum; i++) gpu_sum += partial_sums[i];

//  for (int i = 1 ; i<(int) ceil ((double) num_blocks/2);i++) printf("p_s[%d]=%f\n", i,partial_sums[i]);
// printf("sum=%f\n",partial_sums[0]);
// printf("numels=%d\n",num_elements);

// make sum and mean known to the device
  gpu_mean=gpu_sum/(num_elements);
  cudaSafeCall( cudaMemcpyToSymbol(sum,    &gpu_sum,    sizeof(float)) );
  cudaSafeCall( cudaMemcpyToSymbol(mean,    &gpu_mean,    sizeof(float)) );

  // Finally, execute the reduction kernel again, where it is edited so that we are summing to give the variance as output.
  cudaSafeCall(cudaDeviceSynchronize());
  timeSinceLastTimer = elapsed_time(&timerGPU);
  gpu_calcs=timeRed+timeSinceLastTimer;
  printf("3: Time to sum partials from reduction:\t%f \n",timeSinceLastTimer);

  switch (num_threadsperblock)
   {
  case 1024:
  	  squaresreduction<1024><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 512:
 	  squaresreduction<512><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 256:
	   squaresreduction<256><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 128:
	   squaresreduction<128><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 64:
	   squaresreduction< 64><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 32:
	   squaresreduction< 32><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 16:
	   squaresreduction< 16><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 8:
	   squaresreduction< 8><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 4:
	   squaresreduction< 4><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 2:
	   squaresreduction< 2><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   case 1:
	   squaresreduction< 1><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
   }
  cudaCheckMsg("squares reduction kernel execution failed");
  cudaSafeCall(cudaDeviceSynchronize());
  timeSinceLastTimer = elapsed_time(&timerGPU);
 //printf("Time to run reduction:\t%f \n",time1);
  printf("4: Time to run squares reduction:\t%f \n",timeSinceLastTimer);
//  printf("Time to run both reductions:\t%f\n", timeRed+timeSinceLastTimer);
  gpu_calcs+=timeSinceLastTimer;

// Copy partial sums back and find the s.d.
  cudaSafeCall(cudaMemcpy(partial_sums, d_odata, block_mem_size, cudaMemcpyDeviceToHost));
//  for (int i = 1 ; i<(int) ceil ((double) num_blocks/2);i++) printf("p_s[%d]=%f\n", i,partial_sums[i]);
  for (int i = 1 ; i<halved_blocknum; i++) partial_sums[0] += partial_sums[i];
  gpu_sd=((num_elements-1)*partial_sums[0]/(num_elements));
  float error=(gpu_mean/(num_elements-1))*(gpu_mean/(num_elements-1))*(Reqnum_elements-num_elements);
  gpu_sd=sqrt(gpu_sd-error);

  timeSinceLastTimer = elapsed_time(&timerGPU);
  gpu_calcs+=timeSinceLastTimer;
  printf("5: Total GPU calculation time:\t%f \n",gpu_calcs);

  free(partial_sums);

  cudaSafeCall(cudaFree(d_idata));
  cudaSafeCall(cudaFree(d_odata));

// We get an error in the GPU sd calculation from zero-padding. The easiest fix is to take off the introduced error at the end
//  printf("GPU:\n");
  printf("GPU sd =\t%f\n",gpu_sd);
  printf("GPU mean =\t%f\n",gpu_mean);

  timeSinceLastTimer = elapsed_time(&timerGPU);
  printf("Time to free data used in GPU calcs:\t%f \n",timeSinceLastTimer);


}

// ************************************* Bootstrap: Simple Reduction Kernel ************************************** //

// this simple binary reduction deals with cases that the other reduction doesn't cope with (e.g. small threadblocksize).
__global__ void reduction2(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory
    extern  __shared__  volatile float s_data[];
    int tid = threadIdx.x;
    int id = tid + blockDim.x*blockIdx.x+ gridDim.x*blockDim.x*blockIdx.y;

    // first, each thread loads data into shared memory
    s_data[tid] = g_idata[id];

    // next, we perform binary tree reduction
    for (int d = blockDim.x>>1;d>0; d >>= 1) {
      __syncthreads();  // ensure previous step completed
      if (tid<d) {
    	  s_data[tid] += s_data[tid+d];
      }
    }
    // finally, first thread puts result into global memory
    __syncthreads();
    if (tid==0) {
        g_odata[blockIdx.x+gridDim.x*blockIdx.y]=s_data[0];
   }
}


// ************************************* Bootstrap Kernel ************************************** //
//__constant__ int bins;
// most bias, but quickest (rand in (0,bins-blockDim.x))
__global__ void bootstrap(float *g_odata, float *g_idata, unsigned int *g_irand)
{
	float myResample;

	int constant = (4294967295/(bins-blockDim.x));
	int constant2= blockIdx.x*bins;
	for(int i=0;i<bins;i++) {

	   	int rid = g_irand[constant2+i]/constant;

		myResample+=g_idata[rid+threadIdx.x];
	}

	g_odata[threadIdx.x+blockDim.x*blockIdx.x] = myResample/num_els;

}

//some bias, slower than bootstrap (rand in (0,bins-1) but with modulo to loop)
__global__ void bootstrap2(float *g_odata, float *g_idata, unsigned int *g_irand)
{
	float myResample;

	int constant = (4294967295/(bins));
	int constant2= blockIdx.x*bins;
	for(int i=0;i<bins;i++) {

	   	int rid = g_irand[constant2+i]/constant;

		myResample+=g_idata[(rid+threadIdx.x)%bins];
	}

	g_odata[threadIdx.x+blockDim.x*blockIdx.x] = myResample/num_els;

}

//least amount of bias, slower than bootstrap2 (rand array is much bigger, each thread chooses a sample independent of other threads).
__global__ void bootstrap3(float *g_odata, float *g_idata, unsigned int *g_irand)
{
	float myResample;

	int constant = (4294967295/(bins));
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	for(int i=0;i<bins;i++) {

	   	int rid = g_irand[id*bins+i]/constant;

		myResample+=g_idata[rid];
	}

	g_odata[threadIdx.x+blockDim.x*blockIdx.x] = myResample/num_els;

}

// ************************************* CPU Blocked Bootstrap ************************************** //

void cpu_blocked_bootstrap (float *data_array, unsigned int num_els, unsigned int num_bins, unsigned int num_boots, double &cpu_calcs){
double cpu_timer, timeSinceLastTimer;
elapsed_time(&cpu_timer);

float partial_squaresum=0.0f;
float cpu_mean=0.0f, cpu_sd=0.0f,mean_data=0.0f;

//bin array will contain the partial sum of many numbers
    float *bin_array;
    bin_array = (float*) malloc (num_bins*sizeof(float));
    for (int j=0;j<num_bins;j++) bin_array[j]=0.0f;

//bootstrap mean will contain the bootstrap resamples
    float *bootstrap_mean;
    bootstrap_mean = (float*) malloc(num_boots*sizeof(float));
    for (int j = 0; j<num_boots; j++) bootstrap_mean[j]=0.0f;

//care should be taken here when num_bins*num_boots is large
    int *rand_array;
    rand_array = (int*) malloc (num_bins*num_boots*sizeof(int));
    for (int j=0;j<num_bins*num_boots;j++) rand_array[j]=0;

    timeSinceLastTimer = elapsed_time(&cpu_timer);
    printf("CPU Time to assign memory:\t%f \n",timeSinceLastTimer);

//rand array contains a list of discrete random numbers between 0 and binsize-1
    for (int i = 0; i<num_bins*num_boots;i++) {
	rand_array[i]= num_bins*(rand()/(float) RAND_MAX);
    }

//    for (int i=0; i<256;i++) printf("rand_array[%d]=%d",i,rand_array[i]);

    timeSinceLastTimer = elapsed_time(&cpu_timer);
    printf("CPU Time to create random array:\t%f \n",timeSinceLastTimer);


//bin array contains the sum of the numbers in each bin
    for (int j = 0; j<num_bins; j++){
    	for (int i = 0; i< (num_els/num_bins); i++){
	    bin_array[j]+= data_array[i+j*num_els/num_bins];
        }
    }

    timeSinceLastTimer = elapsed_time(&cpu_timer);
    printf("CPU Time to create bin array:\t%f \n",timeSinceLastTimer);


//bootstrap array contains an array of bootstrap means
//for each bootstrap, sum the num_bins sums into bootstrap_array; then divide these sums by num_elements
    for (int j = 0; j<num_boots; j++){
    	for (int i = 0; i< num_bins; i++){
	    bootstrap_mean[j]+=bin_array[rand_array[i+j*num_bins]];
        }
    }
    for (int j=0; j<num_boots; j++) bootstrap_mean[j]/=num_els;

    cpu_calcs = elapsed_time(&cpu_timer);
    printf("CPU Time to compute bootstrap means:\t%f \n",cpu_calcs);

//compute mean of bootstrap means
    for (int j = 0; j<num_boots; j++){
        cpu_mean += bootstrap_mean[j];
    }
    cpu_mean/=num_boots;

// compute sd of the bootstrap means
    for (int j = 0; j<num_boots; j++){
	partial_squaresum += (bootstrap_mean[j]-cpu_mean)*(bootstrap_mean[j]-cpu_mean);
    }
    cpu_sd=sqrt((partial_squaresum)/(num_boots-1));

// compute mean of data
    for (int i = 0; i<num_bins; i++){
    	mean_data += bin_array[i];
    }
    mean_data/= num_els;

    timeSinceLastTimer = elapsed_time(&cpu_timer);
    printf("CPU Time to calculate statistics:\t%f \n",timeSinceLastTimer);
    printf("\nCPU data mean = %f\n",mean_data);
    printf("CPU bootstrap_mean = %f\n",cpu_mean);
    printf("CPU bootstrap_sd = %f\n",cpu_sd);

//free data
	  free(bin_array);
	  free(bootstrap_mean);
	  free(rand_array);
}

// ************************************* GPU Blocked Bootstrap ************************************** //

void gpu_blocked_bootstrap (float *data, unsigned int num_els, unsigned int num_bins, unsigned int num_boots, double &gpu_calcs, int &DoReduction, float *subbin_array, int num_subbins){
	double gpu_timer, timeSinceLastTimer;
	elapsed_time(&gpu_timer);

	if (num_boots%ELPB!=0) printf("bootstrapping will give the wrong result because ELPB doesn't divide num_boots");
	int num_subboots=num_boots/ELPB;

	int num_blocks, num_threadsperblock, shared_mem_size;
	// float gpu_sum=0.0f;
	float sd_boots=0.0f, mean_data=0.0f, mean_boots=0.0f;

	float *d_odata2, *d_idata2;
	unsigned int *d_irand;
	//d_irand will contain uniformly distr unsigned ints between 0 and 4294967295

	float *bin_array;
	bin_array = ( float*) malloc((num_bins) * sizeof( float));
	for (int i = 0 ; i<num_bins; i++) bin_array[i]=0.0f;

	float *boots_array;
	boots_array = (float*) malloc((num_boots)*sizeof(float));
	for (int i = 0 ; i<num_boots; i++) boots_array[i]=0.0f;

	cudaSafeCall(cudaMalloc((void**)&d_odata2,  (num_bins) * sizeof(float)));
	cudaSafeCall(cudaMalloc((void**)&d_idata2,  (num_boots) * sizeof(float)));

	timeSinceLastTimer = elapsed_time(&gpu_timer);
//	printf("Time to create variables:\t%f \n",timeSinceLastTimer);

//Our Array exclusive_random is used to assigns subbin sums into random bins
	int *exclusive_random;
	exclusive_random=(int*) malloc(num_bins* sizeof(int));
	for(int i = 0; i < num_bins; i++)	exclusive_random[i]=i;

//REDUCTION METHOD, ONLY DONE ONCE IF LOOPING: saves us reducing every time, we use the last subbin array to create our bin array from.

if (DoReduction==1){

  num_threadsperblock=num_els/num_subbins;
//printf("numthreadsperblock=%d",num_threadsperblock);

float *d_idata, *d_odata;
cudaSafeCall(cudaMalloc((void**)&d_idata,(num_els)*sizeof(float)));
cudaSafeCall(cudaMemcpy(d_idata, data, (num_els)*sizeof(float), cudaMemcpyHostToDevice));

if (num_threadsperblock>=128)
{
  int threadblocksize = num_threadsperblock/2;
  if ((threadblocksize!=64)&&(threadblocksize!=128)&&(threadblocksize!=256)&&(threadblocksize!=512)) printf("reduction is giving wrong result because an invalid threadblock size was entered");
// use 256 thread block size; so 512 elements per block.
  int num_threadsperblock=threadblocksize;
  int shared_mem_size = sizeof(float) * num_threadsperblock;

// num_blocks required is typically num_elements/num_threadsperblock, but half this since we read in twice the number of elements
  int num_blocks=(int) ceil((float) num_els/num_threadsperblock);
  int halved_blocknum = (int) ceil((float) num_blocks/2);

// to avoid problems with unspecified launch failures (too many x-dimension blocks), find a suitable value for num_blocks_y:
  int num_blocks_y=1;
  determine_grid (halved_blocknum,num_blocks_y);
  dim3 dimGrid(halved_blocknum/num_blocks_y,num_blocks_y);
  printf("dimGrid(%d,%d)\n",halved_blocknum/num_blocks_y,num_blocks_y);

  cudaSafeCall(cudaMalloc((void**)&d_odata,  (halved_blocknum) * sizeof(float)));

  cudaSafeCall( cudaDeviceSynchronize() );

  switch (num_threadsperblock)
  {
  case 512:
	  reduction<512><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 256:
	  reduction<256><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 128:
	  reduction<128><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 64:
	  reduction< 64><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 32:
	  reduction< 32><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 16:
	  reduction< 16><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 8:
	  reduction< 8><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 4:
	  reduction< 4><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 2:
	  reduction< 2><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  case 1:
	  reduction< 1><<< dimGrid,num_threadsperblock,shared_mem_size>>>(d_idata,d_odata); break;
  }
  cudaCheckMsg("reduction kernel execution failed");
  cudaSafeCall( cudaDeviceSynchronize() );
}

else {

  cudaSafeCall(cudaMalloc((void**)&d_odata,  (num_subbins) * sizeof(float)));
  num_blocks=num_subbins;

  int num_blocks_y=1;
  determine_grid (num_blocks,num_blocks_y);
  dim3 dimGrid(num_blocks/num_blocks_y,num_blocks_y);
  printf("dimGrid(%d,%d)\n",num_blocks/num_blocks_y,num_blocks_y);

	  // size of the memory shared between threads of the same block
  shared_mem_size = sizeof(float) * num_threadsperblock;
  cudaSafeCall( cudaDeviceSynchronize() );
  reduction2<<<dimGrid,num_threadsperblock,shared_mem_size>>>(d_odata,d_idata);
  cudaCheckMsg("reduction kernel execution failed");
  cudaSafeCall( cudaDeviceSynchronize() );

}

timeSinceLastTimer = elapsed_time(&gpu_timer);
printf("Time to run reduction:\t%f \n",timeSinceLastTimer);

cudaSafeCall(cudaMemcpy(subbin_array, d_odata, sizeof(float)*num_subbins, cudaMemcpyDeviceToHost));
cudaSafeCall(cudaFree(d_idata));
cudaSafeCall(cudaFree(d_odata));

DoReduction=0;

}

elapsed_time(&gpu_timer);
//Fischer shuffle algorithm to replace our bins randomly
//sample without replacement once for indexes to replace
	for(int i=0;i<num_bins-1;i++){
		int Rand=(((num_bins-i)*rand())/(float) RAND_MAX );
		int temp=exclusive_random[num_bins-i-1];
		exclusive_random[num_bins-i-1]=exclusive_random[Rand];
		exclusive_random[Rand]=temp;
	}
	timeSinceLastTimer = elapsed_time(&gpu_timer);
	printf("Time to do shuffle:\t%f \n",timeSinceLastTimer);

  //compute partial sums of subbins and place into randomly chosen bins

  for (int j=0; j<num_bins; j++) {
	  //for (int i=0; i<((num_subbins)/(num_bins)); i++) bin_array[j]+=subbin_array[i+j*num_subbins/num_bins];
	  for (int i=0; i<((num_subbins)/(num_bins)); i++) bin_array[exclusive_random[j]]+=subbin_array[i+j*num_subbins/num_bins];
  }

  timeSinceLastTimer = elapsed_time(&gpu_timer);
  printf("Time to create bin array:\t%f \n",timeSinceLastTimer);

	  // create array of random numbers: we need num_subboots*num_bins random numbers
	  
 	  curandGenerator_t gen;
if (BSTRAP==3) cudaSafeCall(cudaMalloc((void**)&d_irand,  (num_bins)*(num_boots) * sizeof(unsigned int)));
else  cudaSafeCall(cudaMalloc((void**)&d_irand,  (num_bins)*(num_subboots) * sizeof(unsigned int)));

	  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
if (BSTRAP==3) 	curandGenerate(gen, d_irand, (num_bins)*(num_boots));
else curandGenerate(gen, d_irand, (num_bins)*(num_subboots));
	  cudaSafeCall( cudaDeviceSynchronize() );

	  timeSinceLastTimer = elapsed_time(&gpu_timer);
	  printf("Time to generate random numbers:\t%f \n samples/sec: %e\n",timeSinceLastTimer,(num_bins)*(num_subboots)/timeSinceLastTimer);
// if few random numbers are needed, consider creating the random numbers "on-the-fly" on the device
	  
	// submit a bootstrap kernel where each thread computes a resample
	  num_blocks=num_subboots;
	  num_threadsperblock=ELPB;
// We checked that BOOTS/ELPB is an integer at the start of the function

	  cudaSafeCall(cudaMemcpy(d_odata2, bin_array, sizeof(float)*num_bins, cudaMemcpyHostToDevice));

	cudaSafeCall( cudaDeviceSynchronize() );
if (BSTRAP==1) bootstrap<<<num_blocks,num_threadsperblock>>>(d_idata2,d_odata2,d_irand);
if (BSTRAP==2) bootstrap2<<<num_blocks,num_threadsperblock>>>(d_idata2,d_odata2,d_irand);
if (BSTRAP==3) bootstrap3<<<num_blocks,num_threadsperblock>>>(d_idata2,d_odata2,d_irand);
	cudaCheckMsg("bootstrap kernel execution failed");
	cudaSafeCall( cudaDeviceSynchronize() );
	gpu_calcs = elapsed_time(&gpu_timer);
	printf("Time to bootstrap:\t%f \n",gpu_calcs);
    cudaSafeCall(cudaMemcpy(boots_array, d_idata2, sizeof(float)*num_boots, cudaMemcpyDeviceToHost));

	  //finally calculate the mean and standard deviation for the bootstraps

	  elapsed_time(&gpu_timer);


	  for (int i = 0 ; i<num_boots;i++) {
		mean_boots+=boots_array[i];
	  }
	  mean_boots/=num_boots;

	  for (int i = 0 ; i<num_bins;i++) {
		mean_data+=bin_array[i];
	  }
	mean_data/=num_els;

	  for (int i = 0 ; i<num_boots;i++) {
	  	sd_boots+=(boots_array[i]-mean_boots)*(boots_array[i]-mean_boots);
	  }

	  sd_boots=sqrt((sd_boots/(num_boots-1)));

	  timeSinceLastTimer = elapsed_time(&gpu_timer);
	  printf("Time to calculate statistics:\t%f \n",timeSinceLastTimer);

// These are checks:
//	  for (int i = 0 ; i<num_bins; i++) if (i%1000==0) printf("bin_array[%d]=%f\n",i,bin_array[i]);
//	  for (int i = 0 ; i<num_boots; i++) if (i%100==0) printf("boots_array[%d]=%f\n",i,boots_array[i]);
//	  for (int i = 0 ; i<num_subbins; i++) if (i%10==0) printf("subbin_array[%d]=%f\n",i,subbin_array[i]);

	  cudaSafeCall(cudaFree(d_idata2));
	  cudaSafeCall(cudaFree(d_odata2));
	  cudaSafeCall(cudaFree(d_irand));

	  //free(subbin_array); //this is freed in the main()
	  free(exclusive_random);
	  free(bin_array);
	  free(boots_array);

	  if (num_bins/32<1) printf ("Too few bins");
	  else{
	  printf("\nGPU data mean =\t%f\n",mean_data);
	  printf("GPU bootstrap_mean =\t%f\n",mean_boots);
	  printf("GPU bootstrap_sd =\t%f\n\n",sd_boots);
	  }

}

// *********************** main() ****************** //

int main( int argc, char** argv)
{

  /* these are #defined at the top
  int CPU_BOOT=0;
  int GPU_BOOT=0;
  int LOOP_BOOT=0;
  int CPU_JACK=0; //decides whether to run CPU code or not.
  int OMP_JACK=0; //decides whether to run OMP code or not.
  int GPU_JACK=0; //decides whether to run GPU code or not.
*/
  double timer, init_time, reset_time, data_time;  // timer variable and elapsed time
  double cpu_boot_elapsed, gpu_boot_elapsed, cpu_jack_elapsed, omp_jack_elapsed, gpu_jack_elapsed;

  int num_elements=ELS;
  int *num_elementsP= &num_elements;

  elapsed_time(&timer);

  float *h_data;
  h_data = ( float*) malloc((num_elements) * sizeof( float));

  // create an array of random numbers
  for(int i = 0; i < num_elements; i++) {
    h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
//    h_data[i] = 1.0f;
  }

// create an array of correlated random numbers
//  for(int i = 0; i < num_elements/128;i++) {
//  for (int j = 0; j<128;j++) 
//  h_data[i*128+j]=floorf(100*((i%2)-(rand())/(float)RAND_MAX)); //each bin, i, of length 128 contains random numbers between -100 and 0 or between 0 and 100
//  }

for (int i = 0 ; i < 512; i++) if(i%1==0) printf("h_data[%d]=%f\n",i,h_data[i]);

  data_time = elapsed_time(&timer);
  printf("Time to create data array:\t%f \n",data_time);

//Bootstrapping variables:

  int num_boots=BOOTS;
  int num_elsperbin=EPB;

  int num_bins=num_elements/num_elsperbin;
  int *num_binsP= &num_bins;

if (CPU_BOOT==1) cpu_blocked_bootstrap(h_data, num_elements, num_bins, num_boots,cpu_boot_elapsed);

  printf("\nnum_els = %d\n",num_elements);
  printf("num_elsperbin = %d\n",num_elsperbin);
  printf("num_bins = %d\n",num_bins);
  printf("num_boots = %d\n",num_boots);

if (GPU_BOOT==1) { // create an array to store partial sums (These come from the reduction kernel).
	// We will use the faster reduction method if num_elspersubbin = 128,256,512,1024
	int num_elspersubbin=1;
	if (num_elsperbin%1024==0&&num_elsperbin>=1024) num_elspersubbin=1024;
	if (num_elsperbin%512==0&&num_elsperbin>=512) num_elspersubbin=512;
	else if (num_elsperbin%256==0&&num_elsperbin>=256) num_elspersubbin=256;
	else if (num_elsperbin%128==0&&num_elsperbin>=128) num_elspersubbin=128;
	else if (num_elsperbin%64==0&&num_elsperbin>=64) num_elspersubbin=64;
	else if (num_elsperbin%32==0&&num_elsperbin>=32) num_elspersubbin=32;
	else if (num_elsperbin%16==0&&num_elsperbin>=16) num_elspersubbin=16;
	else if (num_elsperbin%8==0&&num_elsperbin>=8) num_elspersubbin=8;
	else if (num_elsperbin%4==0&&num_elsperbin>=4) num_elspersubbin=4;
	else if (num_elsperbin%2==0&&num_elsperbin>=2) num_elspersubbin=2;
	else num_elspersubbin=1;


	int num_subbins=num_elements/num_elspersubbin;
	printf("num_subbins=%d\n",num_subbins);

	float *subbin_array;
	subbin_array = ( float*) malloc((num_subbins) * sizeof( float));

	int FirstTime=1; //keeps a track of whether this is the first time through the method.

	elapsed_time(&timer);
	cutilDeviceInit(argc, argv);
	//  cudaFree(0);
	init_time = elapsed_time(&timer);
	printf("Time to initialise device: \t%f \n",init_time);
	cudaSafeCall( cudaMemcpyToSymbol(num_els,    num_elementsP,    sizeof(int)) );
//if we want to loop over different bin sizes, we must run the blocked bootstrap with a for loop
//must also have num elements divisible by 1000*Y (Y=minimum num elements per bin)
//start the loop with els per bin = Y, then 2*Y, then 5*Y then 10*Y ...
//to save time, we want to perform the h_data reduction to a bin array once, then store this bin array somewhere, to create future bin arrays from. By rerandomising after each loop, we lose some of the bias introduced in the kernel.

	if ((LOOP_BOOT==1)&&(num_elements%(num_elsperbin*1000)==0)){
		int num_elsperbinonthisiteration;
		int x=1;
		for (int j=0;j<10;j++){
			num_elsperbinonthisiteration=num_elsperbin*x;
			*num_binsP=num_elements/num_elsperbinonthisiteration;
			cudaSafeCall( cudaMemcpyToSymbol(bins,    num_binsP,    sizeof(int)) );
			printf("\nnum_bins = %d\n",num_bins);
			printf("num_elsperbinonthisiteration = %d\n",num_elsperbinonthisiteration);
			//check num_bins >=32 else it's daft to do statistics
			if (num_bins>=32)
				gpu_blocked_bootstrap(h_data, num_elements, num_bins, num_boots, gpu_boot_elapsed, FirstTime, subbin_array, num_subbins);
			if (CPU_BOOT==1 && GPU_BOOT==1) printf("speedup:\t%f\n", cpu_boot_elapsed/gpu_boot_elapsed);
			if (j%3==0) x=x*2;
			else if (j%3==1) x=(x*5)/2;
			else x=x*2;
		}
	}
	else{
	cudaSafeCall( cudaMemcpyToSymbol(bins,    num_binsP,    sizeof(int)) );
	gpu_blocked_bootstrap(h_data, num_elements, num_bins, num_boots, gpu_boot_elapsed, FirstTime, subbin_array, num_subbins);
	}
	cudaDeviceReset();
	reset_time = elapsed_time(&timer);
	printf("Device Reset Time:\t\t%f \n",reset_time);
	free(subbin_array);
	if (CPU_BOOT==1 && GPU_BOOT==1) printf("speedup:\t%f\n", cpu_boot_elapsed/gpu_boot_elapsed);

}

  if (GPU_BOOT==1) {
	printf("Bootstrap Bandwidth in GB/s:\t%f\n", ((((num_bins/1073741824)*num_boots*sizeof(int))/ELPB)+(num_bins/1073741824)*num_boots*sizeof(float)+num_boots*sizeof(float))/(gpu_boot_elapsed));
	printf("Percentage of peak attained:\t%f\n", 100*(((((num_bins/1073741824)*num_boots*sizeof(int))/ELPB)+(num_bins/1073741824)*num_boots*sizeof(float)+num_boots*sizeof(float))/(gpu_boot_elapsed))/150.0);
	printf("Bootstrap GOPS/s:\t%f\n", ((num_bins*num_boots*4/(gpu_boot_elapsed*1000000000))));
	printf("Percentage of peak attained:\t%f\n",100*((num_bins*num_boots*4/(gpu_boot_elapsed*1000000000))	/1030 ));
  }

//Jackknife Codes:

if (CPU_JACK==1) {
 // elapsed_time(&timer);
  cpu_jackknife(h_data,num_elements, cpu_jack_elapsed);
 // cpu_elapsed = elapsed_time(&timer);
 // printf("Time to complete CPU calculations: \t%f \n",cpu_elapsed);
}

if (OMP_JACK==1) {
 // elapsed_time(&timer);
  omp_jackknife(h_data,num_elements, omp_jack_elapsed);
 // cpu_elapsed = elapsed_time(&timer);
 // printf("Time to complete CPU calculations: \t%f \n",cpu_elapsed);
}

if (GPU_JACK==1) {
  // Measure time to initialise the device
  elapsed_time(&timer);
  cutilDeviceInit(argc, argv);
  init_time = elapsed_time(&timer);
  printf("Time to initialise device: \t%f \n",init_time);

 // make number of elements known to the device (This could already have been done by GPU boot)
  cudaSafeCall( cudaMemcpyToSymbol(num_els,    num_elementsP,    sizeof(int)) );
  gpu_jackknife(h_data,num_elements,gpu_jack_elapsed);

  // Measure time to reset device
  elapsed_time(&timer);
  cudaDeviceReset();
  reset_time = elapsed_time(&timer);
  printf("Device Reset Time:\t\t%f \n",reset_time);
}

if (GPU_JACK==1 && CPU_JACK==1)
  printf("Jackknife CPU/OMP Speedup = %f\n", cpu_jack_elapsed/omp_jack_elapsed);
if (GPU_JACK==1 && CPU_JACK==1)
  printf("Jackknife CPU/CUDA Speedup = %f\n", cpu_jack_elapsed/gpu_jack_elapsed);
if (GPU_JACK==1 && OMP_JACK==1)
  printf("Jackknife OMP/CUDA Speedup = %f\n", omp_jack_elapsed/gpu_jack_elapsed);


free(h_data);

}

