#pragma once
#include"help.h"

// размер матрицы
const int N = 2048*1;
// размер блока
const int b = 64;


//void norma(const vec& arr1, const vec& arr2)
//{
//	double maximum = 0;
//	double new_maximum = 0;
//	long m = 0;
//	for (size_t i = 0; i < N; ++i)
//		for (size_t j = 0; j < N; ++j) {
//			new_maximum = fmax(maximum, fabs(arr1[i * N + j] - arr2[i * N + j]));
//			if (new_maximum > maximum) {
//				maximum = new_maximum;
//				m = i * N + j;
//			}
//		}
//	std::cout << "ќшибка: " << maximum << "  i = " << m << std::endl;
//}


matrix create_matrix(size_t n, size_t m, double min = -10.0, double max = 10.0)
{
	//srand(1);
	matrix A;
	A.resize(n, vec(m));
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < m; j++)
			A[i][j] = 1 + rand() % 100;//min + (max - min) / RAND_MAX * rand();
	return A;
}


void create_matrix(double* arr, size_t n, int min = 1, int max = 5) {
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			arr[i * n + j] = min + rand() % (max - min + 1);
	//arr[i * n + j] = min + (max - min) / RAND_MAX * rand();
}



std::pair<int, int> index_matrix(size_t index, size_t n, size_t m)
{
	int j = 1;
	if (index <= n * m)
		while (index > m) {
			index -= m;
			j += 1;
		}
	else
		std::cout << "Ќомер элемента вышел за диапазон" << std::endl;
	return std::pair<int, int>(j, index);
}

vec matrix_to_vector(const matrix& A)
{
	vec vect;
	for (size_t i = 0; i < A.size(); i++)
		for (size_t j = 0; j < A[0].size(); j++)
			vect.push_back(A[i][j]);
	return vect;
}


vec create_vector(size_t n, int min, int max)
{
	vec b(n);
	for (size_t i = 0; i < n; i++)
		b[i] = min + rand() % (max - min);
	print_vec(b);
	return b;
}

//произведение матриц
void multiplyAB(const matrix& A, const matrix& B)
{
	size_t n = A.size();
	matrix C;
	C.resize(N, vec(N));

	unsigned int start_time = clock(); // начальное врем€
	for (int i = 0; i < n; i++)
		// считать строку i матрицы ј в быструю пам€ть
		for (int j = 0; j < n; j++)
			// считать R[i][j] в быструю пам€ть
			// считать столбец j матрицы B в быструю пам€ть
			for (int k = 0; k < n; k++)
				C[i][j] += A[i][k] * B[k][j];
	unsigned int end_time = clock(); // конечное врем€
	std::cout << (double)(end_time - start_time) / CLOCKS_PER_SEC << std::endl;
	print_vec(C);
}

void multiplyA(matrix& A)
{
	size_t n = A.size();
	matrix check_A(n, vec(n));
	check_A[0][0] = A[0][0];
	for (int i = 1; i < n; i++) {
		check_A[i][0] = A[i][0] * A[0][0];
		check_A[0][i] = A[0][i];
	}
	print_vec(check_A);
}
