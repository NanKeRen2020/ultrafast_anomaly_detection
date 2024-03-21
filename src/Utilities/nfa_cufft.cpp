#include"iostream"
#include"cuda_runtime_api.h"
#include"device_launch_parameters.h"
#include"cufft.h"

using namespace std;
//FFT反变换后，用于规范化的函数
__global__ void normalizing(cufftComplex* data, int data_len)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	data[idx].x /= data_len;
	data[idx].y /= data_len;
}
void Check(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		cout << "行号:" << __LINE__ << endl;
		cout << "错误:" << cudaGetErrorString(status) << endl;
	}
}
int main()
{
	const int Nt = 256;
	const int BATCH = 1;
	//BATCH用于批量处理一批一维数据，当BATCH=2时
	//则将0-1024，1024-2048作为两个一维信号做FFT处理变换
	cufftComplex* host_in, *host_out, *device_in, *device_out;
	//主机内存申请及初始化--主机锁页内存
	Check(cudaMallocHost((void**)&host_in, Nt * sizeof(cufftComplex)));
	Check(cudaMallocHost((void**)&host_out, Nt * sizeof(cufftComplex)));
	for (int i = 0; i < Nt; i++)
	{
		host_in[i].x = i + 1;
		host_in[i].y = i + 1;
	}
	//设备内存申请
	Check(cudaMalloc((void**)&device_in, Nt * sizeof(cufftComplex)));
	Check(cudaMalloc((void**)&device_out, Nt * sizeof(cufftComplex)));
	//数据传输--H2D
	Check(cudaMemcpy(device_in, host_in, Nt * sizeof(cufftComplex), cudaMemcpyHostToDevice));
 
	//创建cufft句柄
	cufftHandle cufftForwrdHandle, cufftInverseHandle;
	cufftPlan1d(&cufftForwrdHandle, Nt, CUFFT_C2C, BATCH);
	cufftPlan1d(&cufftInverseHandle, Nt, CUFFT_C2C, BATCH);
 
	//执行fft正变换
	cufftExecC2C(cufftForwrdHandle, device_in, device_out, CUFFT_FORWARD);
 
	//数据传输--D2H
	Check(cudaMemcpy(host_out, device_out, Nt * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
 
	//设置输出精度--正变换结果输出
	cout << "正变换结果:" << endl;
	cout.setf(20);
	for (int i = 0; i < Nt; i++)
	{
		cout << host_out[i].x << "+j*" << host_out[i].y << endl;
	}
 
	//执行fft反变换
	cufftExecC2C(cufftInverseHandle, device_out, device_in, CUFFT_INVERSE);
 
	//IFFT结果是真值的N倍，因此要做/N处理
	dim3 grid(Nt / 128);
	dim3 block(128);
	normalizing << <grid, block >> > (device_in, Nt);
 
	//数据传输--D2H
	Check(cudaMemcpy(host_in, device_in, Nt * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
 
	//设置输出精度--反变换结果输出
	cout << "反变换结果:" << endl;
	cout.setf(20);
	for (int i = 0; i < Nt; i++)
	{
		cout << host_in[i].x << "+j*" << host_in[i].y << endl;
	}
	cin.get();
	return 0;
}