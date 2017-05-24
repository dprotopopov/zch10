
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <locale.h>
#include <iostream>

int n = 3; // размерность пространства
double a[] = { 0, 0, 0 }; // нижняя граница значения переменных
double b[] = { 100, 100, 100 }; // верхняя граница значения переменных
int R = 10000; // количество итераций
double e = 1e-7; // точность вычисления 

double p[] = { 10, 20, 30 };

// Исследуемая функция
// после вызова надо сложить элементы массива s
__global__ void f(double *x, double *p, double *s)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	s[id] = (x[id] - p[id])*(x[id] - p[id]);
}

// Инициализация генератора псевдослучайных чисел
__global__ void setuprand(curandState *state)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x; /* Each thread gets same seed, a different sequence number, no offset */
	curand_init(1234, id, 0, &state[id]);
}

// Генератор псевдослучайного вектора
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
	// Поддержка кириллицы в консоли Windows
	// Функция setlocale() имеет два параметра, первый параметр - тип категории локали, в нашем случае LC_TYPE - набор символов, второй параметр — значение локали. 
	// Вместо второго аргумента можно писать "Russian", или оставлять пустые двойные кавычки, тогда набор символов будет такой же как и в ОС.
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

	// Выбор начальной точки
	randvector <<<1, n>>>(devX, devA, devB, devStates);
	f <<<1, n>>>(devX, devP, devS);
	cudaMemcpy(s, devS, n*sizeof(double), cudaMemcpyDeviceToHost);
	double fx = 0; for (int i = 0; i < n; i++) fx += s[i];

	for (auto r = 0; r < R; r++)
	{
		// Выбор следующей точки
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

	// Вывод результатов
	cudaMemcpy(x, devX, n*sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << "Точка минимума : ";
	for (auto i = 0; i < n; i++)
	{
		std::cout << x[i];
		if (i < n - 1) std::cout << ",";
	}
	std::cout << std::endl;

	std::cout << "Значение минимума : " << fx << std::endl;

	free(x);
	free(s);
	cudaFree(devX);
	cudaFree(devX1);
	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devP);
	cudaFree(devS);
	cudaFree(devStates);

	getchar(); // Ожидание ввода с клавиатуры перед завершением программы
	return 0;
}

