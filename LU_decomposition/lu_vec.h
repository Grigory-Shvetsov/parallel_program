#pragma once
#include"matrix.h"
#include "omp.h"


//---------------------------------------------------------------------------------------------
// ХРАНЕНИЕ В ВЕКТОРАХ!!!
//---------------------------------------------------------------------------------------------


//(НЕБЛОЧНОЕ - ПАРАЛЛЕЛЬНОЕ) LU - разложение матрицы A (m * n)
void algorithm2_9_parallel(vec& A, const int m, const int b, const int i0, const int j0)
{
	//double t = -omp_get_wtime();
	for (int i = 0; i < std::min(m - 1, b); ++i)
	{
#pragma omp parallel
		{
#pragma omp for 
			for (int j = i + 1; j < m; ++j)
				A[(j + i0) * N + i + j0] /= A[(i + i0) * N + i + j0];

			if (i < b - 1)
			{
#pragma omp  for 
				for (int j = i + 1; j < m; ++j)
					for (int k = i + 1; k < b; ++k)
						A[(j + i0) * N + k + j0] -= A[(j + i0) * N + i + j0] * A[(i + i0) * N + k + j0];
			}

		}
		/*t += omp_get_wtime();
		return t;*/
	}
}



//(НЕБЛОЧНОЕ - ПОСЛЕДОВАТЕЛЬНОЕ) LU - разложение матрицы A (m * n)
void algorithm2_9(vec& A, const int m, const int b, const int i0, const int j0)
{
	//double t = -omp_get_wtime();
	for (int i = 0; i < std::min(m - 1, b); ++i)
	{
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


//(БЛОЧНОЕ - ПАРАЛЛЕЛЬНОЕ) LU - разложение матрицы A (m * n)
double LU_BLOCK_v1_parallel(vec& A)
{
	double t = -omp_get_wtime();
	vec U23, L32;
	U23.resize(b * (N - b));
	L32.resize(b * (N - b));

	for (int i = 0; i < N - 1; i += b) {


		algorithm2_9_parallel(A, N - i, b, i, i);

#pragma omp parallel
		{
#pragma omp for
			for (int j = 0; j < N - b - i; ++j)
				for (int k = 0; k < b; ++k)
				{
					double sum = 0;
					for (int q = 0; q < k; ++q)
						sum += A[(i + k) * N + (i + q)] * A[(q + i) * N + b + i + j];
					A[(k + i) * N + b + i + j] -= sum;
				}


#pragma omp for 
			for (int j = 0; j < N - b - i; ++j)
				for (int k = 0; k < b; ++k)
				{
					U23[k * (N - i - b) + j] = A[(i + k) * N + j + i + b];
					L32[j * b + k] = A[(i + j + b) * N + k + i];
				}

#pragma omp for 
			for (int j = b + i; j < N; ++j)
				for (int c = 0; c < b; ++c)
					for (int k = b + i; k < N; ++k)
						A[j * N + k] -= L32[(j - i - b) * b + c] * U23[c * (N - i - b) + k - (i + b)];
		}
	}
	t += omp_get_wtime();
	//std::cout << (A[7421757]) << "\n";
	//print_vec(A, true);
	return t;
}



//(БЛОЧНОЕ - НЕПАРАЛЛЕЛЬНОЕ) LU - разложение матрицы A (m * n)
double LU_BLOCK_v1(vec& A)
{
	double t = -omp_get_wtime();
	vec U23, L32;
	U23.resize(b * (N - b));
	L32.resize(b * (N - b));

	for (int i = 0; i < N - 1; i += b) {

		//std::cout << "Цикл по i: " << i << std::endl;
		algorithm2_9(A, N - i, b, i, i);
		//print_vec(A, true);

		for (int j = 0; j < N - b - i; ++j)
			for (int k = 0; k < b; ++k) {
				double sum = 0;
				for (int q = 0; q < k; ++q)
					sum += A[(i + k) * N + (i + q)] * A[(q + i) * N + b + i + j];
				A[(k + i) * N + b + i + j] -= sum;
			}

		//print_vec(A, true);

		for (int j = 0; j < N - b - i; ++j)
			for (int k = 0; k < b; ++k)
			{
				U23[k * (N - i - b) + j] = A[(i + k) * N + j + i + b];
				L32[j * b + k] = A[(i + j + b) * N + k + i];
			}
		/*std::cout << "Посчитали U23 и L32:" << std::endl;
		print_vec(U23, true);
		print_vec(L32, true);*/

		for (int j = b + i; j < N; ++j)
			for (int c = 0; c < b; ++c)
				for (int k = b + i; k < N; ++k)
					A[j * N + k] -= L32[(j - i - b) * b + c] * U23[c * (N - i - b) + k - (i + b)];
		/*std::cout << "Изменилась подматрица" << std::endl;
		print_vec(A, true);*/
	}
	t += omp_get_wtime();
	//print_vec(A, true);
	//std::cout << (A[7421757]) << "\n";
	return t;

}



//(БЛОЧНОЕ - НЕПАРАЛЛЕЛЬНОЕ) LU - разложение матрицы A (m * n)
double LU_BLOCK_v2_parallel(vec& A)
{
	double t = -omp_get_wtime();
	vec U23, L32, L22;
	U23.resize(b * (N - b));
	L32.resize(b * (N - b));
	L22.resize(b * b);

	for (int j = 0; j < N / b; ++j) {
#pragma omp parallel
		{
#pragma omp for
			for (int p = 0; p < b; p++)
				for (int q = 0; q < b; q++)
					L22[b * p + q] = A[N * (p + j * b) + q + j * b];
#pragma omp for
			for (int p = b + j * b; p < N; p++)
				for (int q = 0; q < b; q++)
					L32[b * (p - b - j * b) + q] = A[N * p + q + j * b];

#pragma omp for
			for (int p = 0; p < b; p++)
				for (int q = b + j * b; q < N; q++)
					U23[p * (N - b - j * b) + q - j * b - b] = A[N * (j * b + p) + q];
		}
		//---------------------------------------------------------------

		//LU - разложение------------------------------------------------ 
		for (int p = 0; p < std::min(N - 1, b); ++p) {
#pragma omp parallel 
			{
#pragma omp for
				for (int m = p + 1; m < b; ++m)
					L22[m * b + p] = L22[m * b + p] / L22[p * b + p];
#pragma omp for
				for (int m = b; m < N; ++m)
					L32[(m - b) * b + p] = L32[(m - b) * b + p] / L22[p * b + p];


				if (p < b)
				{
#pragma omp for
					for (int m = p + 1; m < b; ++m)
						for (int k = p + 1; k < b; ++k)
							L22[m * b + k] = L22[m * b + k] - L22[p * b + k] * L22[m * b + p];
#pragma omp for
					for (int m = b; m < N; ++m)
						for (int k = p + 1; k < b; ++k)
							L32[(m - b) * b + k] = L32[(m - b) * b + k] - L22[p * b + k] * L32[(m - b) * b + p];
				}
			}

		}
		//---------------------------------------------------------------

		//Считаем U23----------------------------------------------------
#pragma omp parallel	
		{
#pragma omp  for
			for (int p = 0; p < N - b - (b * j); ++p)
				for (int m = 1; m < b; ++m)
					for (int k = 0; k < m; ++k)
						U23[m * (N - b - j * b) + p] -= U23[k * (N - b - j * b) + p] * L22[m * b + k];
		}
		//-----------------------------------------------------------------

		//Записываем в А
#pragma omp parallel 
		{
#pragma omp for
			for (int k = 0; k < b; k++)
				for (int m = 0; m < b; m++)
					A[(j * b + k) * N + j * b + m] = L22[k * b + m];
#pragma omp for
			for (int k = (1 + j) * b; k < N; ++k)
				for (int m = j * b; m < (1 + j) * b; ++m)
					A[k * N + m] = L32[(k - (1 + j) * b) * b + m - j * b];
#pragma omp for
			for (int k = b * (j + 1); k < N; ++k)
				for (int m = b * j; m < b * (j + 1); ++m)
					A[m * N + k] = U23[(m - b * j) * (N - b - j * b) + k - b * (j + 1)];

			//Считаем подматрицу 
#pragma omp for
			for (int m = b * (j + 1); m < N; ++m)
				for (int k = 0; k < b; ++k)
					for (int p = b * (j + 1); p < N; ++p)
						A[m * N + p] -= L32[(m - b * (j + 1)) * b + k] * U23[k * (N - b - j * b) + p - b * (j + 1)];
		}
	}
	t += omp_get_wtime();
	//print_vec(A, true);
	//std::cout << (A[1047576]) << "\n";
	return t;
}


//(БЛОЧНОЕ - НЕПАРАЛЛЕЛЬНОЕ) LU - разложение матрицы A (m * n)
double LU_BLOCK_v2(vec& A)
{
	double t = -omp_get_wtime();
	vec U23, L32, L22;
	U23.resize(b * (N - b));
	L32.resize(b * (N - b));
	L22.resize(b * b);

	for (int j = 0; j < N / b; ++j) {
		//std::cout << "Цикл по j: " << j << std::endl;

		//записываем из А----------------------------------------------
		for (int p = 0; p < b; p++)
			for (int q = 0; q < b; q++)
				L22[b * p + q] = A[N * (p + j * b) + q + j * b];
		//print_vec(L22, b, b, true);

		for (int p = b + j * b; p < N; p++)
			for (int q = 0; q < b; q++)
				L32[b * (p - b - j * b) + q] = A[N * p + q + j * b];
		//print_vec(L32, N - b - j * b, b, true);


		for (int p = 0; p < b; p++)
			for (int q = b + j * b; q < N; q++)
				U23[p * (N - b - j * b) + q - j * b - b] = A[N * (j * b + p) + q];
		//print_vec(U23, b, N - b - j * b, true);
		//---------------------------------------------------------------

		//LU - разложение------------------------------------------------ 
		for (int p = 0; p < std::min(N - 1, b); ++p) {

			for (int m = p + 1; m < b; ++m)
				L22[m * b + p] = L22[m * b + p] / L22[p * b + p];
			for (int m = b; m < N; ++m)
				L32[(m - b) * b + p] = L32[(m - b) * b + p] / L22[p * b + p];


			if (p < b)
			{
				for (int m = p + 1; m < b; ++m)
					for (int k = p + 1; k < b; ++k)
						L22[m * b + k] = L22[m * b + k] - L22[p * b + k] * L22[m * b + p];
				for (int m = b; m < N; ++m)
					for (int k = p + 1; k < b; ++k)
						L32[(m - b) * b + k] = L32[(m - b) * b + k] - L22[p * b + k] * L32[(m - b) * b + p];
			}

		}
		//---------------------------------------------------------------
		/*std::cout << "Матрица L22" << std::endl;
		print_vec(L22, b, b, true);
		std::cout << "Матрица L32" << std::endl;
		print_vec(L32, N - b - j * b, b, true);*/


		//Считаем U23----------------------------------------------------
		for (int p = 0; p < N - b - (b * j); ++p)
			for (int m = 1; m < b; ++m)
				for (int k = 0; k < m; ++k)
					U23[m * (N - b - j * b) + p] -= U23[k * (N - b - j * b) + p] * L22[m * b + k];
		//-----------------------------------------------------------------
		/*std::cout << "Матрица U23" << std::endl;
		print_vec(U23, b, N - b - j * b, true);*/

		//Записываем в А
		for (int k = 0; k < b; k++)
			for (int m = 0; m < b; m++)
				A[(j * b + k) * N + j * b + m] = L22[k * b + m];

		for (int k = (1 + j) * b; k < N; ++k)
			for (int m = j * b; m < (1 + j) * b; ++m)
				A[k * N + m] = L32[(k - (1 + j) * b) * b + m - j * b];

		for (int k = b * (j + 1); k < N; ++k)
			for (int m = b * j; m < b * (j + 1); ++m)
				A[m * N + k] = U23[(m - b * j) * (N - b - j * b) + k - b * (j + 1)];

		//print_vec(A, true);

		//Считаем подматрицу 
		for (int m = b * (j + 1); m < N; ++m)
			for (int k = 0; k < b; ++k)
				for (int p = b * (j + 1); p < N; ++p)
					A[m * N + p] -= L32[(m - b * (j + 1)) * b + k] * U23[k * (N - b - j * b) + p - b * (j + 1)];


		//print_vec(A, true);

	}
	t += omp_get_wtime();
	//print_vec(A, true);
	//std::cout << (A[1047576]) << "\n";
	return t;
}


//(НЕБЛОЧНЫЙ - ПАРАЛЛЕЛЬНЫЙ) LU - разложение матрицы A (n * n)
double LU_parallel(vec& A)
{
	double t = -omp_get_wtime();
	for (int i = 0; i < N - 1; ++i) {
#pragma omp parallel
		{
#pragma omp for
			for (int j = i + 1; j < N; ++j)
				A[N * j + i] /= A[i * N + i];

#pragma omp for
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
double LU(vec& A)
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
	std::cout << (A[1047576]) << "\n";
	//print_vec(A, true);
	return t;
}
