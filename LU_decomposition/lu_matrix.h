#pragma once
#include"matrix.h"
#include "omp.h"



//---------------------------------------------------------------------------------------------
// ХРАНЕНИЕ В МАТРИЦАХ!!!
//---------------------------------------------------------------------------------------------


//(НЕБЛОЧНОЕ - ПАРАЛЛЛЬНОЕ) LU - разложение матрицы A (m * n)
double algorithm2_9_parallel(matrix& A)
{
	double t = -omp_get_wtime();
	int m = A.size();
	int n = A[0].size();
	for (int i = 0; i < std::min(m - 1, n); ++i) {
#pragma omp parallel
		{
#pragma omp for
			for (int j = i + 1; j < m; ++j)
				A[j][i] = A[j][i] / A[i][i];
			if (i < n - 1)
#pragma omp for 
				for (int j = i + 1; j < m; ++j)
					for (int k = i + 1; k < n; ++k)
						A[j][k] = A[j][k] - A[j][i] * A[i][k];
		}
	}
	t += omp_get_wtime();

	//print_vec(A);
	return t;
}



//(НЕБЛОЧНОЕ - ПОСЛЕДОВАТЕЛЬНОЕ) LU - разложение матрицы A (m * n)
double algorithm2_9(matrix& A)
{
	double t = -omp_get_wtime();
	int m = A.size();
	int n = A[0].size();
	for (int i = 0; i < std::min(m - 1, n); ++i) {
		for (int j = i + 1; j < m; ++j)
			A[j][i] = A[j][i] / A[i][i];
		if (i < n - 1)
			for (int j = i + 1; j < m; ++j)
				for (int k = i + 1; k < n; ++k)
					A[j][k] = A[j][k] - A[j][i] * A[i][k];
	}
	t += omp_get_wtime();
	//print_vec(A);
	return t;
}



//(НЕБЛОЧНЫЙ - ПАРАЛЛЕЛЬНЫЙ) LU - разложение матрицы A (n * n)
double LU_parallel(matrix& A)
{
	const double N = A.size();
	double t = -omp_get_wtime();
	for (int i = 0; i < N - 1; ++i) {
#pragma omp parallel
		{
#pragma omp for
			for (int j = i + 1; j < N; ++j)
				A[j][i] /= A[i][i];
#pragma omp for 
			for (int j = i + 1; j < N; ++j)
				for (int k = i + 1; k < N; ++k)
					A[j][k] -= A[j][i] * A[i][k];
		}
	}
	t += omp_get_wtime();
	//print_vec(A);
	return t;


}


//(НЕБЛОЧНЫЙ - ПОСЛЕДОВАТЕЛЬНЫЙ) LU - разложение матрицы A (n * n)
matrix LU(matrix& A)
{
	double N = A.size();
	//double t = -omp_get_wtime();
	for (int i = 0; i < N - 1; ++i) {
		for (int j = i + 1; j < N; ++j)
			A[j][i] /= A[i][i];

		for (int j = i + 1; j < N; ++j)
			for (int k = i + 1; k < N; ++k)
				A[j][k] -= A[j][i] * A[i][k];
	}
	//t += omp_get_wtime();
	//print_vec(A);
	return A;
}


//LU - разложение (понадобятся две матрицы L и U)
void LU_traditional(const matrix& A, matrix& L, matrix& U)
{
	U = A;
	int N = U.size();

	for (int i = 0; i < N; ++i)
		for (int j = i; j < N; ++j)
			L[j][i] = U[j][i] / U[i][i];

	for (int k = 1; k < N; ++k)
	{
		for (int i = k - 1; i < N; ++i)
			for (int j = i; j < N; ++j)
				L[j][i] = U[j][i] / U[i][i];

		for (int i = k; i < N; ++i)
			for (int j = k - 1; j < N; ++j)
				U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j];
	}
	print_vec(L);
	print_vec(U);

}
