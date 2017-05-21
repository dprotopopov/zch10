// montecarlo.cpp: ���������� ����� ����� ��� ����������� ����������.
//

#include "stdafx.h"
#include <vector>
#include <ctime>
#include <locale.h>
#include <assert.h>
#include <iostream>

int n = 3; // ����������� ������������
std::vector<double> a = { 0, 0, 0 }; // ������ ������� �������� ����������
std::vector<double> b = { 100, 100, 100 }; // ������� ������� �������� ����������
int R = 10000; // ���������� ��������
double e = 1e-7; // �������� ���������� 

// ����������� �������
double f(const std::vector<double> &x)
{
	const std::vector<double> p = { 10, 20, 30 };
	assert(x.size() == n);
	assert(p.size() == n);

	double s = 0;
	for (auto i = 0; i < n; i++) s += (x[i] - p[i])*(x[i] - p[i]);
	return s;
}

// ��������� ��������������� ����� � ��������� [0;1]
// ���������� �� ������ ������������ ���������� ��������������� �����
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

// ��������� ���������������� �������
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
	// ��������������� ���������� ��������������� ����� ��������� �������� �������
	srand(time(nullptr));

	// ��������� ��������� � ������� Windows
	// ������� setlocale() ����� ��� ���������, ������ �������� - ��� ��������� ������, � ����� ������ LC_TYPE - ����� ��������, ������ �������� � �������� ������. 
	// ������ ������� ��������� ����� ������ "Russian", ��� ��������� ������ ������� �������, ����� ����� �������� ����� ����� �� ��� � � ��.
	setlocale(LC_ALL, "");

	std::vector<double> x;

	// ����� ��������� �����
	randvector(x, a, b, n);
	auto fx = f(x);

	for (auto r = 0; r < R; r++)
	{
		std::vector<double> x1;

		// ����� ��������� �����
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

	// ����� �����������

	std::cout << "����� �������� : ";
	for (auto i = 0; i < n; i++)
	{
		std::cout << x[i];
		if (i < n - 1) std::cout << ",";
	}
	std::cout << std::endl;

	std::cout << "�������� �������� : " << fx << std::endl;

	getchar(); // �������� ����� � ���������� ����� ����������� ���������
	return 0;
}

