// gradient.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <assert.h>
#include <vector>
#include <locale.h>
#include <ctime>
#include <ostream>
#include <iostream>
#include <algorithm>

int n = 3; // размерность пространства
std::vector<double> a = { 0, 0, 0 }; // нижняя граница значения переменных
std::vector<double> b = { 100, 100, 100 }; // верхняя граница значения переменных
int R = 10000; // количество итераций
double e1 = 1e-7; // точность вычисления 
double e2 = 1e-7; // точность вычисления 
double t = 0.5; // Величина шага

const std::vector<double> p = { 10, 20, 30 };
// Исследуемая функция
double f(const std::vector<double> &x)
{
	assert(x.size() == n);
	assert(p.size() == n);

	double s = 0;
	for (auto i = 0; i < n; i++) s += (x[i] - p[i])*(x[i] - p[i]);
	return s;
}

// Градиент исследуемой функции
void gradientvector(std::vector<double> &g, const std::vector<double> &x)
{
	assert(x.size() == n);
	assert(p.size() == n);

	g.clear();
	for (auto i = 0; i < n; i++)
	{
		g.push_back(2.0*(x[i] - p[i]));
	}
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
	std::vector<double> g;

	// Выбор начальной точки
	randvector(x, a, b, n);
	auto fx = f(x);

	for (;;)
	{
		gradientvector(g, x);
		double s = 0;
		for (auto i = 0; i < n; i++) s += g[i] * g[i];
		s = sqrt(s);
		if (s < e1) break;
		std::vector<double> x1;
		auto fx1 = fx;
		
		// Выбор следующей точки
		double l = 0;
		for (auto t1 = t;; t1 /= 2)
		{
			x1.clear();
			for (auto i = 0; i < n; i++) x1.push_back(std::max(a[i],std::min(x[i] - t1*g[i],b[i])));
			fx1 = f(x1);
			l = 0;
			for (auto i = 0; i < n; i++) l += (x[i] - x1[i]) * (x[i] - x1[i]);
			l = sqrt(l);
			if (fx1 < fx) break;
			if (l < e2) break;
		}
		if (l < e2 && abs(fx - fx1) < e2)
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

