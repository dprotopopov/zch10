
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <locale.h>
#include <iostream>

int n = 3; // ����������� ������������
double a[] = { 0, 0, 0 }; // ������ ������� �������� ����������
double b[] = { 100, 100, 100 }; // ������� ������� �������� ����������
int R = 10000; // ���������� ��������
double e = 1e-7; // �������� ���������� 

double p[] = { 10, 20, 30 };

// ����������� �������
// ����� ������ ���� ������� �������� ������� s
__global__ void f(double *x, double *p, double *s)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	s[id] = (x[id] - p[id])*(x[id] - p[id]);
}

// ������������� ���������� ��������������� �����
__global__ void setuprand(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x; /* Each thread gets same seed, a different sequence number, no offset */
	curand_init(1234, id, 0, &state[id]);
}

// ��������� ���������������� �������
__global__ void randvector(double *x, double *a, double *b, curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = state[id]; 
	double p = curand_uniform_double(&localState);
	state[id] = localState; 
	x[id] = a[id] + p*(b[id] - a[id]);
}

int main(int argc, char* argv[])
{
	// ��������� ��������� � ������� Windows
	// ������� setlocale() ����� ��� ���������, ������ �������� - ��� ��������� ������, � ����� ������ LC_TYPE - ����� ��������, ������ �������� � �������� ������. 
	// ������ ������� ��������� ����� ������ "Russian", ��� ��������� ������ ������� �������, ����� ����� �������� ����� ����� �� ��� � � ��.
	setlocale(LC_ALL, "");

	curandState *devStates;
	cudaMalloc((void **)&devStates, n*sizeof(curandState));
	setuprand <<<1, n>>>(devStates);

	double *devA, *devB, *devX, *devX1, *devP, *devS;
	double *x, *s;
	cudaMalloc((void **)&devA, n*sizeof(double));
	cudaMalloc((void **)&devB, n*sizeof(double));
	cudaMalloc((void **)&devX, n*sizeof(double));
	cudaMalloc((void **)&devX1, n*sizeof(double));
	cudaMalloc((void **)&devP, n*sizeof(double));
	cudaMalloc((void **)&devS, n*sizeof(double));
	x = (double *)malloc(n*sizeof(double));
	s = (double *)malloc(n*sizeof(double));

	cudaMemcpy(devA, a, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devP, p, n*sizeof(double), cudaMemcpyHostToDevice);

	// ����� ��������� �����
	randvector <<<1, n>>>(devX, devA, devB, devStates);
	f <<<1, n>>>(devX, devP, devS);
	cudaMemcpy(s, devS, n*sizeof(double), cudaMemcpyDeviceToHost);
	double fx = 0; for (int i = 0; i < n; i++) fx += s[i];

	for (auto r = 0; r < R; r++)
	{
		// ����� ��������� �����
		randvector <<<1, n>>>(devX1, devA, devB, devStates);
		f <<<1, n>>>(devX1, devP, devS);
		cudaMemcpy(s, devS, n*sizeof(double), cudaMemcpyDeviceToHost);
		double fx1 = 0; for (int i = 0; i < n; i++) fx1 += s[i];

		if (fx < fx1) continue;
		if (abs(fx - fx1)<e)
		{
			cudaMemcpy(devX, devX1, n*sizeof(double), cudaMemcpyDeviceToDevice);
			fx = fx1;
			break;
		}
		cudaMemcpy(devX, devX1, n*sizeof(double), cudaMemcpyDeviceToDevice);
		fx = fx1;
	}

	// ����� �����������
	cudaMemcpy(x, devX, n*sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << "����� �������� : ";
	for (auto i = 0; i < n; i++)
	{
		std::cout << x[i];
		if (i < n - 1) std::cout << ",";
	}
	std::cout << std::endl;

	std::cout << "�������� �������� : " << fx << std::endl;

	free(x);
	free(s);
	cudaFree(devX);
	cudaFree(devX1);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devP);
	cudaFree(devS);
	cudaFree(devStates);

	getchar(); // �������� ����� � ���������� ����� ����������� ���������
	return 0;
}

