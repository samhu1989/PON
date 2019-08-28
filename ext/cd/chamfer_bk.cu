
#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void NmDistanceKernel(int b,int n,const int dim,const float * xyz,int m,const float * xyz2,float * result,int * result_i,const int batch,float* buf){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*dim;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*dim+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				const float* xyz1 = &(xyz[(i*n+j)*dim]);
				int best_i=0;
				float best=0;
				int end_ka=end_k-(end_k&3);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[k*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[(k+1)*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[(k+2)*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[(k+3)*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[k*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[(k+1)*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[(k+2)*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
                            float d = 0.0;
							for(int di=0;di<dim;++di)
                            {
                                float dif = buf[(k+3)*dim+di] - xyz1[di];
                                d += dif*dif;
                            }
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
                    float d = 0.0;
					for(int di=0;di<dim;++di)
                    {
                        float dif = buf[k*dim+di] - xyz1[di];
                        d += dif*dif;
                    }
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}
// int chamfer_cuda_forward(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream){
int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2,at::Tensor idx1,at::Tensor idx2,at::Tensor buf){

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
    const auto dim = xyz1.size(2);
	const auto m = xyz2.size(1); //num_points point cloud B
    const auto buf_batch_size = buf.size(0);
    if( (dim != xyz2.size(2)) || (dim != buf.size(1)) ){
        printf("dim do not match in chamfer_cuda_forward\n");
        return 0;
    }
	NmDistanceKernel<<<dim3(32,16,1),512>>>(batch_size, n, dim, xyz1.data<float>(), m, xyz2.data<float>(), dist1.data<float>(), idx1.data<int>() , buf_batch_size , buf.data<float>());
	NmDistanceKernel<<<dim3(32,16,1),512>>>(batch_size, m, dim, xyz2.data<float>(), n, xyz1.data<float>(), dist2.data<float>(), idx2.data<int>() , buf_batch_size , buf.data<float>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;
}
__global__ void NmDistanceGradKernel(int b,int n,const int dim,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
            int j2=idx1[i*n+j];
            float g=grad_dist1[i*n+j]*2;
            for(int di = 0;di<dim;++di)
			{
                float x1=xyz1[(i*n+j)*dim+di];
                float x2=xyz2[(i*m+j2)*dim+di];
                atomicAdd(&(grad_xyz1[(i*n+j)*dim+di]),g*(x1-x2));
                atomicAdd(&(grad_xyz2[(i*m+j2)*dim+di]),-(g*(x1-x2)));
            }
		}
	}
}
// int chamfer_cuda_backward(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream){
int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2){
	// cudaMemset(grad_xyz1,0,b*n*3*4);
	// cudaMemset(grad_xyz2,0,b*m*3*4);
	
	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
    const auto dim = xyz1.size(2);
	const auto m = xyz2.size(1); //num_points point cloud B
    if(dim != xyz2.size(2)){
        printf("dim do not match in chamfer_cuda_backward\n");
        return 0;
    }
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(batch_size,n,dim,xyz1.data<float>(),m,xyz2.data<float>(),graddist1.data<float>(),idx1.data<int>(),gradxyz1.data<float>(),gradxyz2.data<float>());
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(batch_size,m,dim,xyz2.data<float>(),n,xyz1.data<float>(),graddist2.data<float>(),idx2.data<int>(),gradxyz2.data<float>(),gradxyz1.data<float>());
	
	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;
	
}

__device__ inline void swapf(float & a, float & b)
{   
    float tmp = a;
    a = b;
    b = tmp;
}

__device__ inline void swap(int & a, int & b)
{
    int tmp = a;
    a = b ;
    b = tmp;
}

__global__ void KnnKernel(int b,const int n,const int dim,const float * xyz,const int k,float * result,int * result_i){
    const int size = 4096;
    __shared__ float dist[size];
    __shared__ int idx[size];
    assert( n <= size );
    for ( int bi = blockIdx.x ; bi < b ; bi += gridDim.x )
    {
        for ( int i = blockIdx.y ;  i < n  ; i += gridDim.y )
        {
            for ( int j = threadIdx.x ; j < n ; j += blockDim.x )
            {
                if( i == j ){
                    dist[j] = 0;
                    idx[j]  = j;
                    continue;
                }
                float d = 0.0;
                for ( int di = 0 ; di < dim ; ++di )
                {
                    float dif = xyz[(bi*n+i)*dim+di] - xyz[(bi*n+j)*dim+di];
                    d += dif*dif;
                }
                dist[j] = d;
                idx[j] = j;
            }
            __syncthreads();
            //odd-even sort
	    int pownum = int(log2(float(n)));
	    if ( n != pow(2, pownum) ){
            for ( int cnt = 0 ; cnt < ( n + 1 ) / 2 ; ++cnt )
            {
                for ( int j = 2*threadIdx.x + 1 ; j < n ; j += 2*blockDim.x )
                {
                    if ( dist[j] < dist[ j - 1 ] )
                    {
                        swapf(dist[j], dist[j-1]);
                        swap(idx[j], idx[j-1]);
                    }
                }
                __syncthreads();
                for ( int j = 2*threadIdx.x + 2 ; j < n ; j += 2*blockDim.x )
                {
                    if ( dist[j] < dist[ j - 1 ] )
                    {
                        swapf(dist[j], dist[j-1]);
                        swap(idx[j], idx[j-1]);
                    }
                }
                __syncthreads();
            }
	    }else{	
            //Bitonic Sort
            for (unsigned int t = 2; t <= n ; t *= 2)
            {
                // Bitonic merge:
                for (unsigned int j = t / 2; j>0; j /= 2)
                {	
			for (unsigned int tid = threadIdx.x ; tid < n ; tid += blockDim.x )
                    	{
				unsigned int ixj = tid ^ j;
                    		if (ixj > tid)
                    		{
                        		if ((tid & t) == 0)
                        		{
                            			if (dist[tid] > dist[ixj])
                            			{
                                			swapf(dist[tid], dist[ixj]);
                                			swap(idx[tid], idx[ixj]);
                            			}
                        		}
                        		else
                        		{
                            			if (dist[tid] < dist[ixj])
                            			{
                                			swapf(dist[tid], dist[ixj]);
                                			swap(idx[tid], idx[ixj]);
                            			}
                        		}
                    		}
                    		
			}
			__syncthreads();	
                }
            }
	    }
            __syncthreads();
            //copy result
            for ( int j = threadIdx.x ; j < k  ; j += blockDim.x )
            {
                result[(bi*n+i)*k+j] = dist[j+1];
                result_i[ ((bi*n+i)*k+j)*2+0 ] = bi;
                result_i[ ((bi*n+i)*k+j)*2+1 ] = idx[j+1];
            }
            
        }
    }
}
int knn_cuda(at::Tensor xyz,at::Tensor k,at::Tensor dist,at::Tensor idx)
{
    const auto bs = xyz.size(0);
	const auto n = xyz.size(1); //num_points point cloud
    const auto d = xyz.size(2);
    int k_ = k.data<int>()[0];
	KnnKernel<<<dim3(bs,16,1),512>>>(bs,n,d,xyz.data<float>(),k_,dist.data<float>(),idx.data<int>());
	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd Knn: %s\n", cudaGetErrorString(err));
	    return 0;
	  }
	  return 1;
}

