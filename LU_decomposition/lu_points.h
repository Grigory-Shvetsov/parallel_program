#pragma once

#include"matrix.h"
#include "omp.h"


//(НЕБЛОЧНОЕ - ПОСЛЕДОВАТЕЛЬНОЕ) LU - разложение матрицы A (m * n)
void algorithm2_9(double* A, const int m, const int b, const int i0, const int j0)
{
	//double t = -omp_get_wtime();
	for (int i = 0; i < std::min(m - 1, b); ++i) {
		for (int j = i + 1; j < m; ++j)
			A[(j + i0) * N + i + j0] /= A[(i + i0) * N + i + j0];
		//print_vec(A, true);

		if (i < b - 1) {
			for (int j = i + 1; j < m; ++j)
				for (int k = i + 1; k < b; ++k)
					A[(j + i0) * N + k + j0] -= A[(j + i0) * N + i + j0] * A[(i + i0) * N + k + j0];
			//print_vec(A, true);
		}

	}
	/*t += omp_get_wtime();
	return t;*/
}

//(БЛОЧНЫЙ - ПОСЛЕДОВАТЕЛЬНЫЙ)
double LU_BLOCK(double* A)
{
	double t = -omp_get_wtime();
	double* L00 = new double[b * b];
	double* L10 = new double[(N - b) * b];
	double* U01 = new double[b * (N - b)];

	/*vec L00;
	L00.resize(b * b);
	vec L10;
	L10.resize((N - b) * b);

	vec U01;
	U01.resize(b * (N - b));*/

	for (int i = 0; i < N - 1; i += b) {
		std::cout << "------------------------------" << std::endl;
		std::cout << "Цикл по i: " << i << std::endl;
		std::cout << "------------------------------" << std::endl;
		algorithm2_9(A, N - i, b, i, i);

		print_vec(A, N, N, true);


		for (int p = 0; p < b; p++)
			for (int q = 0; q < b; q++)
				L00[b * p + q] = A[(p + i) * N + q + i];
		print_vec(L00, b, b, true);
		
		for (int p = b + i; p < N; p++)
			for (int q = 0; q < b; q++) {
				/*std::cout << p * N + q + i << std::endl;
				std::cout << A[p * N + q + i] << std::endl;*/
				L10[b * (p - b) + q] = A[p * N + q + i];
			}
		print_vec(L10, N - b - i, b, true);

		//решаем СЛАУ (обратный ход)
		for (int j = 0; j < N - b - i; j++)
			for (int p = 0; p < b; p++) {
				double sum = 0.0;
				for (int q = 0; q < p; q++)
				{
					sum += U01[j + (N - b) * q] * L00[q + b * p];
				}
				U01[j + (N - b - i) * p] = A[(j + b + i) + N * (p + i)] - sum;
				/*std::cout << j + (N - b) * p << std::endl;
				std::cout << U01[j + (N - b) * p] << std::endl;*/
			}
		print_vec(U01, b, N - b - i, true);


		for (int j = 0; j < N - b - i; j++)
			for (int p = 0; p < b; p++)
				A[(j + b + i) + N * (p + i)] = U01[j + (N - b - i) * p];

		print_vec(A, N, N, true);

		for (int j = b + i; j < N; ++j)
			for (int p = 0; p < b; ++p)
				for (int q = b + i; q < N; ++q)
					A[j * N + q] -= L10[(j - i - b) * b + p] * U01[p * (N - i - b) + q - (i + b)];



		print_vec(A, N, N, true);

	}
	delete[] L00;
	delete[] U01;
	delete[] L10;

	t += omp_get_wtime();

	//print_vec(A, N, N, true);

	return t;
}




//(НЕБЛОЧНЫЙ - ПАРАЛЛЕЛЬНЫЙ) LU - разложение матрицы A (n * n)
double LU_parallel(double* A)
{
	double t = -omp_get_wtime();
	for (int i = 0; i < N - 1; ++i) {
#pragma omp parallel
		{
#pragma omp  for
			for (int j = i + 1; j < N; ++j)
				A[N * j + i] /= A[i * N + i];

#pragma omp  for
			for (int j = i + 1; j < N; ++j)
				for (int k = i + 1; k < N; ++k)
					A[N * j + k] -= A[N * j + i] * A[N * i + k];
		}
	}
	t += omp_get_wtime();
	//print_vec(A, true);
	return t;
}



//(НЕБЛОЧНЫЙ - ПОСЛЕДОВАТЕЛЬНЫЙ) LU - разложение матрицы A (n * n)
double LU(double* A)
{
	double t = -omp_get_wtime();
	for (int i = 0; i < N - 1; ++i) {
		for (int j = i + 1; j < N; ++j)
			A[N * j + i] /= A[i * N + i];

		for (int j = i + 1; j < N; ++j)
			for (int k = i + 1; k < N; ++k)
				A[N * j + k] -= A[N * j + i] * A[N * i + k];
	}
	t += omp_get_wtime();
	print_vec(A, N, N, true);
	return t;
}
