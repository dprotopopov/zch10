// montecarlo.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <vector>
#include <ctime>
#include <locale.h>
#include <assert.h>
#include <iostream>

int n = 3; // размерность пространства
std::vector<double> a = { 0, 0, 0 }; // нижняя граница значения переменных
std::vector<double> b = { 100, 100, 100 }; // верхняя граница значения переменных
int R = 10000; // количество итераций
double e = 1e-7; // точность вычисления 

// Исследуемая функция
double f(const std::vector<double> &x)
{
	const std::vector<double> p = { 10, 20, 30 };
	assert(x.size() == n);
	assert(p.size() == n);

	double s = 0;
	for (auto i = 0; i < n; i++) s += (x[i] - p[i])*(x[i] - p[i]);
	return s;
}

// Генератор псевдослучайных чисел в диапазоне [0;1]
// Реализован на основе стандартного генератора псевдослучайных чисел
double drand()
{
	double x = 0;
	double y = 1;
	for (auto i = 0; i < sizeof(double); i++)
	{
		x += (y /= 256)*(rand() % 256);
	}
	return x;
}

// Генератор псевдослучайного вектора
void randvector(std::vector<double> &x, const std::vector<double> &a, const std::vector<double> &b, int n)
{
	assert(a.size() == n);
	assert(b.size() == n);
	x.clear();
	for (auto i = 0; i < n; i++)
	{
		x.push_back(a[i] + drand()*(b[i] - a[i]));
	}
}

int main(int argc, char* argv[])
{
	// Иницициализация генератора псевдослучайных чисел значением текущего времени
	srand(time(nullptr));

	// Поддержка кириллицы в консоли Windows
	// Функция setlocale() имеет два параметра, первый параметр - тип категории локали, в нашем случае LC_TYPE - набор символов, второй параметр — значение локали. 
	// Вместо второго аргумента можно писать "Russian", или оставлять пустые двойные кавычки, тогда набор символов будет такой же как и в ОС.
	setlocale(LC_ALL, "");

	std::vector<double> x;

	// Выбор начальной точки
	randvector(x, a, b, n);
	auto fx = f(x);

	for (auto r = 0; r < R; r++)
	{
		std::vector<double> x1;

		// Выбор следующей точки
		randvector(x1, a, b, n);
		auto fx1 = f(x1);
		if (fx < fx1) continue;
		if (abs(fx - fx1)<e)
		{
			x = x1;
			fx = fx1;
			break;
		}
		x = x1;
		fx = fx1;
	}

	// Вывод результатов

	std::cout << "Точка минимума : ";
	for (auto i = 0; i < n; i++)
	{
		std::cout << x[i];
		if (i < n - 1) std::cout << ",";
	}
	std::cout << std::endl;

	std::cout << "Значение минимума : " << fx << std::endl;

	getchar(); // Ожидание ввода с клавиатуры перед завершением программы
	return 0;
}

