#pragma once
#include<vector>
#include<iostream>
#include<string>
#include<fstream>
#include<iomanip>
#include<cmath>



std::vector<std::vector<double>> typedef matrix;
std::vector<double> typedef vec;


void print_vec(double* A, int N, int M)
{
	for (size_t i = 0; i < N * M; i++)
		std::cout << std::setprecision(5) << A[i] << "\t";
	std::cout << std::endl;
}

void print_vec(vec& A)
{
	int n = A.size();
	for (size_t i = 0; i < n; i++)
		std::cout << std::setprecision(5) << A[i] << "\t";
	std::cout << std::endl << std::endl;
}


void print_vec(vec A, int N, int M, bool flag)
{
	if (flag)
		for (size_t i = 0; i < N; i++) {
			for (size_t j = 0; j < M; j++)
				std::cout << std::setprecision(5) << A[i * M + j] << "\t\t";
			std::cout << std::endl;
		}
	else
		for (size_t i = 0; i < N * M; i++) {
			std::cout << A[i] << "\t";
		}
	std::cout << std::endl;
	std::cout << std::endl;
}

void print_vec(vec& A, bool flag)
{
	int n = int(sqrt(A.size()));
	if (flag)
		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < n; j++)
				std::cout << std::setprecision(5) << A[i * n + j] << "\t\t";
			std::cout << std::endl;
		}
	else
		for (size_t i = 0; i < n * n; i++) {
			std::cout << A[i] << "\t";
		}
	std::cout << std::endl;
	std::cout << std::endl;
}

void print_vec(const matrix& res)
{
	for (size_t i = 0; i < res.size(); i++) {
		for (size_t j = 0; j < res[i].size(); j++)
			std::cout << std::setprecision(5) << res[i][j] << "\t";
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;
}

void matrix_to_file(const matrix& res, std::string str)
{
	std::ofstream out(str);
	for (size_t i = 0; i < res.size(); i++) {
		for (size_t j = 0; j < res[i].size(); j++)
			if (j == res[i].size() - 1)
				out << res[i][j];
			else
				out << res[i][j] << "\t";
		out << std::endl;
	}
	out.close();
}


void file_to_matrix(std::string str, matrix& A)
{
	std::ifstream in(str);
	if (in.is_open()) {
		double value = 0;
		char symbol;
		vec vect;
		while (true) {
			vect.clear();
			while (true) {
				in >> value;
				vect.push_back(value);
				in.get(symbol);
				if (symbol == '\n')
					break;
			}
			if (in.eof())
				break;
			A.push_back(vect);
		};
	}
	in.close();
}

matrix operator-(matrix A, matrix B)
{
	for (size_t i = 0; i < A.size(); i++)
		for (size_t j = 0; j < A.size(); j++)
			A[i][j] = A[i][j] - B[i][j];
	return A;
}

matrix operator+(matrix A, matrix B)
{
	for (size_t i = 0; i < A.size(); i++)
		for (size_t j = 0; j < A.size(); j++)
			A[i][j] = A[i][j] + B[i][j];
	return A;
}
