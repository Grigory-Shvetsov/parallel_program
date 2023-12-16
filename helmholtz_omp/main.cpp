
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <omp.h>

using namespace std;

const double pi = 3.14159265359;

const double eps = 0.0001;

double f(double x, double y) {

	double k = 5000.0;
	return (2 * sin(pi * y) + k * k * (1 - x) * x * sin(pi * y) + pi * pi * (1 - x) * x * sin(pi * y));
}

double u(double x, double y) {

	return ((1 - x) * x * sin(pi * y));
}

void Jacobi(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0);
void Jacobi_2(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0);
void Zeydel(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0);
void Zeydel_2(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0);


double norma(const double* u, const double* y, const size_t n) {

	double sum = 0;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n * n; ++i)
		sum += (u[i] - y[i]) * (u[i] - y[i]);

	return sqrt(sum);
}

void vec_u(const size_t n, double h, double (*u)(double, double), double* y) {
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			y[i * n + j] = u(j * h, i * h);
}

int main()
{
	size_t n = 5000;
	double k = 5000.0;
	double h = 1.0 / n;
	double* y;
	y = new double[(n + 1) * (n + 1)];

	double* U;
	U = new double[(n + 1) * (n + 1)];
	double* y0 = new double[(n + 1) * (n + 1)];


	vec_u(n + 1, h, u, U);

	int count = 0;

	cout << "N = " << n << "\teps = " << eps << endl;

	cout << setw(15) << "Jacobi" << endl;

	omp_set_num_threads(1);
	double start = omp_get_wtime();
	Jacobi(k, f, n + 1, h, y, eps, count, U, y0);
	double finish = omp_get_wtime();
	double total_time = finish - start;



	cout << " threads: " << 1 << "\tcount: " <<
		count << "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time << endl;

	for (int i = 2; i <= 18; i += 2) {

		omp_set_num_threads(i);
		start = omp_get_wtime();
		Jacobi(k, f, n + 1, h, y, eps, count, U, y0);
		finish = omp_get_wtime();
		double total_time2 = finish - start;

		cout
			<< " threads: " << i
			<< "\tcount: " << count
			<< "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time2 << "\tt_start/t: " << total_time / total_time2 << endl;
		cout << endl;
	}
	cout << endl;


	cout << setw(15) << "Jacobi_2" << endl;

	omp_set_num_threads(1);
	start = omp_get_wtime();
	Jacobi_2(k, f, n + 1, h, y, eps, count, U, y0);
	finish = omp_get_wtime();
	double total_time3 = finish - start;



	cout << " threads: " << 1 << "\tcount: " <<
		count << "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time3 << endl;

	for (int i = 2; i <= 18; i += 2) {

		omp_set_num_threads(i);
		start = omp_get_wtime();
		Jacobi_2(k, f, n + 1, h, y, eps, count, U, y0);
		finish = omp_get_wtime();
		double total_time4 = finish - start;

		cout
			<< " threads: " << i
			<< "\tcount: " << count
			<< "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time4 << "\tt_start/t: " << total_time3 / total_time4 << endl;
		cout << endl;
	}
	cout << endl;

	cout << setw(15) << "Zeydel" << endl;

	omp_set_num_threads(1);
	start = omp_get_wtime();
	Zeydel(k, f, n + 1, h, y, eps, count, U, y0);
	finish = omp_get_wtime();
	double total_time5 = finish - start;

	cout << " threads: " << 1 << "\tcount: " <<
		count << "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time5 << endl;


	for (int i = 2; i <= 4; i += 2) {

		omp_set_num_threads(i);
		start = omp_get_wtime();
		Zeydel(k, f, n + 1, h, y, eps, count, U, y0);
		finish = omp_get_wtime();
		double total_time6 = finish - start;

		cout << " threads: " << i << "\tcount: " <<
			count << "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time6 << "\tt_start/t: " << total_time5 / total_time6 << endl;
		cout << endl;
	}

	cout << setw(15) << "Zeydel_2" << endl;

	omp_set_num_threads(1);
	start = omp_get_wtime();
	Zeydel_2(k, f, n + 1, h, y, eps, count, U, y0);
	finish = omp_get_wtime();
	double total_time7 = finish - start;

	cout << " threads: " << 1 << "\tcount: " <<
		count << "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time5 << endl;


	for (int i = 2; i <= 4; i += 2) {

		omp_set_num_threads(i);
		start = omp_get_wtime();
		Zeydel_2(k, f, n + 1, h, y, eps, count, U, y0);
		finish = omp_get_wtime();
		double total_time8 = finish - start;

		cout << " threads: " << i << "\tcount: " <<
			count << "\tnorma: " << norma(U, y, n + 1) << "\tt: " << total_time8 << "\tt_start/t: " << total_time7 / total_time8 << endl;
		cout << endl;
	}


	delete[] y;
	delete[] U;
	delete[] y0;
}


void Jacobi(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0)
{
	count = 0;
	for (size_t i = 0; i < n * n; ++i)
		y[i] = 0;

	double a = 1.0 / (4 + k * k * h * h);


	do {
		swap(y, y0);
		count++;
#pragma omp parallel for 
		for (int i = 1; i < n - 1; ++i)
			for (int j = 1; j < n - 1; ++j)
				y[i * n + j] = a * (y0[i * n + (j - 1)] + y0[(i - 1) * n + j] + y0[(i + 1) * n + j] + y0[i * n + (j + 1)] + h * h * f(j * h, i * h));

	} while (norma(y0, y, n) > eps);

}

void Jacobi_2(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0)
{
	count = 0;
	for (size_t i = 0; i < n * n; ++i)
		y[i] = 0;

	double a = 1.0 / (4 + k * k * h * h);
	double sum;


	do {
		sum = 0.0;
		swap(y, y0);
		count++;
#pragma omp parallel for reduction(+:sum)
		for (int i = 1; i < n - 1; ++i)
			for (int j = 1; j < n - 1; ++j)
			{
				y[i * n + j] = a * (y0[i * n + (j - 1)] + y0[(i - 1) * n + j] + y0[(i + 1) * n + j] + y0[i * n + (j + 1)] + h * h * f(j * h, i * h));
				sum += (y0[i * n + j] - y[i * n + j]) * (y0[i * n + j] - y[i * n + j]);
			}

	} while ((sqrt(sum) > eps));

}

void Zeydel(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0) {

	for (size_t i = 0; i < n * n; ++i)
		y[i] = 0;

	double a = 1.0 / (4 + k * k * h * h);
	count = 0;
	do {

		swap(y, y0);
		count++;

#pragma omp parallel for 
		for (int i = 1; i < n - 1; ++i)
			for (int j = (i + 1) % 2 + 1; j < n - 1; j += 2)
				y[i * n + j] = a * (y0[i * n + (j - 1)] + y0[(i - 1) * n + j] + y0[(i + 1) * n + j] + y0[i * n + (j + 1)] + h * h * f(j * h, i * h));

#pragma omp parallel for 
		for (int i = 1; i < n - 1; ++i)
			for (int j = i % 2 + 1; j < n - 1; j += 2)
				y[i * n + j] = a * (y[i * n + (j - 1)] + y[(i - 1) * n + j] + y[(i + 1) * n + j] + y[i * n + (j + 1)] + h * h * f(j * h, i * h));

	} while (norma(y0, y, n) > eps);

}


void Zeydel_2(const double k, double (*f)(double, double), const size_t n, const double h, double* y, const double eps, int& count, const double* U, double* y0) {

	for (size_t i = 0; i < n * n; ++i)
		y[i] = 0;

	double a = 1.0 / (4 + k * k * h * h);
	count = 0;
	double sum;
	do {
		sum = 0.0;

		swap(y, y0);
		count++;

#pragma omp parallel for 
		for (int i = 1; i < n - 1; ++i)
			for (int j = (i + 1) % 2 + 1; j < n - 1; j += 2)
				y[i * n + j] = a * (y0[i * n + (j - 1)] + y0[(i - 1) * n + j] + y0[(i + 1) * n + j] + y0[i * n + (j + 1)] + h * h * f(j * h, i * h));

#pragma omp parallel for 
		for (int i = 1; i < n - 1; ++i)
#pragma omp parallel for reduction(+:sum)
			for (int j = i % 2 + 1; j < n - 1; j += 2)
			{
				y[i * n + j] = a * (y[i * n + (j - 1)] + y[(i - 1) * n + j] + y[(i + 1) * n + j] + y[i * n + (j + 1)] + h * h * f(j * h, i * h));
				sum += (y0[i * n + j] - y[i * n + j]) * (y0[i * n + j] - y[i * n + j]);
			}
	} while (sqrt(sum) > eps);

}