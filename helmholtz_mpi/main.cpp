#include<iostream>
#include<iomanip>
#include<functional>
#include<vector>
#include<cmath>
#include"mpi.h"

#define PI 3.14159265358979323846

using namespace std;

typedef vector<double> vec;

//параметры
size_t N = 5000;
double k = N;
const double eps = 0.000001;
double h = 1.0 / (N - 1);
double hh = h * h;
double coef = 4.0 + k * k * hh;


void print_vec(const vec& matr, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j)
			std::cout << fixed << setprecision(8) << matr[i * n + j] << "\t";
		std::cout << std::endl;
	}
	cout << endl << endl;

}

void print_vec(const vec& matr, size_t n, size_t m) {
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j)
			std::cout << fixed << setprecision(8) << matr[i * n + j] << "\t";
		std::cout << std::endl;
	}
}



double exact(double x, double y)
{
	return ((1 - x) * x * sin(PI * y));
}

double fun(double x, double y)
{
	return (2 * sin(PI * y) + k * k * (1 - x) * x * sin(PI * y) +
		PI * PI * (1 - x) * x * sin(PI * y));
}


double norma(const vec& y)
{
	double norm = 0.0;
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			norm += (exact(j * h, i * h) - y[i * N + j]) * (exact(j * h, i * h) - y[i * N + j]);
	return sqrt(norm);
}

template<typename T>
void Jacobi_Iteration(T y, const T y0, size_t M, size_t start)
{
	for (size_t i = 1; i < M - 1; ++i)
		for (size_t j = 1; j < N - 1; ++j)
			y[i * N + j] = (hh * fun(j * h, (i + start) * h) +
				y0[i * N + j - 1] + y0[i * N + j + 1] + y0[i * N + j - N] + y0[i * N + j + N]) / coef;
}

template<typename T>
void Seidel_Iteration(T y, const T y0, size_t M, size_t color, size_t start)
{
	for (size_t i = 1; i < M - 1; ++i)
		for (size_t j = 1 + (start + i + 1 + color) % 2; j < N - 1; j += 2)
			y[i * N + j] = (hh * fun(j * h, (i + start) * h) +
				y0[i * N + j - 1] + y0[i * N + j + 1] + y0[i * N + j - N] + y0[i * N + j + N]) / coef;
}

//method: "Jacobi", "Seidel"
void Send_plus_Recv(string method, vec& y, vec& y0, size_t M, size_t& count, size_t rank, size_t commSize, size_t start)
{
	count = 0;
	double err = 0.0;
	double next;
	double prev;

	do
	{
		if (method == "Jacobi")
			Jacobi_Iteration(y.data(), y0.data(), M, start);
		else
		{
			Seidel_Iteration(y.data(), y0.data(), M, 0, start);

			next = rank + 1;
			prev = rank - 1;
			//двойная синхронизация
			if (rank == commSize - 1)
				next = MPI_PROC_NULL;

			if (rank == 0)
				prev = MPI_PROC_NULL;

			MPI_Send(y.data() + N * (M - 2), N, MPI_DOUBLE, next, 0, MPI_COMM_WORLD);
			MPI_Recv(y.data() + N * (M - 1), N, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Recv(y.data(), N, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(y.data() + N, N, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD);


			Seidel_Iteration(y.data(), y.data(), M, 1, start);
		}

		//локальная ошибка
		double norm_rank = 0.0;
		for (size_t i = N; i < N * M - N; ++i)
			norm_rank += (y[i] - y0[i]) * (y[i] - y0[i]);


		//объединяет значения из всех процессов и распределяет результат обратно во все процессы
		MPI_Allreduce(&norm_rank, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		err = sqrt(err);

		// отправляем новые данные другим процессам
		next = rank + 1;
		prev = rank - 1;

		if (rank == commSize - 1)
			next = MPI_PROC_NULL;
		if (rank == 0)
			prev = MPI_PROC_NULL;

		MPI_Send(y.data() + N * (M - 2), N, MPI_DOUBLE, next, 0, MPI_COMM_WORLD);
		MPI_Recv(y.data() + N * (M - 1), N, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(y.data(), N, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(y.data() + N, N, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD);

		y0.swap(y);
		++count;

	} while (err > eps);
}


//method: "Jacobi", "Seidel"
void SendRecv(string method, vec& y, vec& y0, size_t M, size_t& count, size_t rank, size_t commSize, size_t start)
{
	count = 0;
	double err = 0.0;

	do
	{
		if (method == "Jacobi")
			Jacobi_Iteration(y.data(), y0.data(), M, start);
		else
		{
			Seidel_Iteration(y.data(), y0.data(), M, 0, start);

			if (rank != 0)
				MPI_Sendrecv(y.data() + N, N, MPI_DOUBLE, rank - 1, 0, y.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			if (rank != commSize - 1)
				MPI_Sendrecv(y.data() + N * (M - 2), N, MPI_DOUBLE, rank + 1, 0, y.data() + N * (M - 1), N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			Seidel_Iteration(y.data(), y.data(), M, 1, start);
		}

		//локальная ошибка
		double norm_rank = 0.0;
		for (size_t i = N; i < N * M - N; ++i)
			norm_rank += (y[i] - y0[i]) * (y[i] - y0[i]);


		//объединяет значения из всех процессов и распределяет результат обратно во все процессы
		MPI_Allreduce(&norm_rank, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		err = sqrt(err);

		// отправляем новые данные другим процессам
		if (rank != 0)
			MPI_Sendrecv(y.data() + N, N, MPI_DOUBLE, rank - 1, 0, y.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (rank != commSize - 1)
			MPI_Sendrecv(y.data() + N * (M - 2), N, MPI_DOUBLE, rank + 1, 0, y.data() + N * (M - 1), N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		y0.swap(y);
		++count;  

	} while (err > eps);
}


//method: "Jacobi", "Seidel"
void ISend_plus_IRecv(string method, vec& y, vec& y0, size_t M, size_t& count, size_t rank, size_t commSize, size_t start)
{
	count = 0;
	double err = 0.0;

	// инициализация запросов
	MPI_Request* prev_e = new MPI_Request[2], * prev_o = new MPI_Request[2];
	MPI_Request* next_e = new MPI_Request[2], * next_o = new MPI_Request[2];

	if (rank != 0) {
		MPI_Send_init(y0.data() + N, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, prev_e);
		MPI_Recv_init(y0.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, prev_e + 1);

		MPI_Send_init(y.data() + N, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, prev_o);
		MPI_Recv_init(y.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, prev_o + 1);
	}
	if (rank != commSize - 1) {
		MPI_Send_init(y0.data() + N * (M - 2), N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, next_e);
		MPI_Recv_init(y0.data() + N * (M - 1), N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, next_e + 1);

		MPI_Send_init(y.data() + N * (M - 2), N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, next_o);
		MPI_Recv_init(y.data() + N * (M - 1), N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, next_o + 1);
	}


	do {
		if (rank != 0)
			MPI_Startall(2, (count % 2 == 0) ? prev_e : prev_o);

		if (rank != commSize - 1)
			MPI_Startall(2, (count % 2 == 0) ? next_e : next_o);

		if (method == "Jacobi")
			Jacobi_Iteration(y.data() + N, y0.data() + N, M - 1, start + 1);
		else
			Seidel_Iteration(y.data() + N, y0.data() + N, M - 1, 0, start + 1);

		if (rank != 0)
			MPI_Waitall(2, (count % 2 == 0) ? prev_e : prev_o, MPI_STATUSES_IGNORE);

		if (rank != commSize - 1)
			MPI_Waitall(2, (count % 2 == 0) ? next_e : next_o, MPI_STATUSES_IGNORE);

		if (method == "Jacobi")
		{
			for (size_t j = 1; j < N - 1; ++j)
				y[N + j] = (hh * fun(j * h, (1 + start) * h) +
					y0[N + j - 1] + y0[N + j + 1] + y0[N + j - N] + y0[N + j + N]) / coef;

			for (size_t j = 1; j < N - 1; ++j)
				y[(M - 2) * N + j] = (hh * fun(j * h, ((M - 2) + start) * h) +
					y0[(M - 2) * N + j - 1] + y0[(M - 2) * N + j + 1] + y0[(M - 2) * N + j - N] + y0[(M - 2) * N + j + N]) / coef;
		}
		else {

			for (size_t j = 1 + (start) % 2; j < N - 1; j += 2)
				y[N + j] = (hh * fun(j * h, (1 + start) * h) +
					y0[N + j - 1] + y0[N + j + 1] + y0[N + j - N] + y0[N + j + N]) / coef;
			for (size_t j = 1 + (start + M - 1) % 2; j < N - 1; j += 2)
				y[(M - 2) * N + j] = (hh * fun(j * h, (M - 2 + start) * h) +
					y0[(M - 2) * N + j - 1] + y0[(M - 2) * N + j + 1] + y0[(M - 2) * N + j - N] + y0[(M - 2) * N + j + N]) / coef;

			if (rank != 0)
				MPI_Startall(2, (count % 2 == 1) ? prev_e : prev_o);

			if (rank != commSize - 1)
				MPI_Startall(2, (count % 2 == 1) ? next_e : next_o);

			Seidel_Iteration(y.data() + N, y.data() + N, M - 1, 1, start + 1);

			if (rank != 0)
				MPI_Waitall(2, (count % 2 == 1) ? prev_e : prev_o, MPI_STATUSES_IGNORE);

			if (rank != commSize - 1)
				MPI_Waitall(2, (count % 2 == 1) ? next_e : next_o, MPI_STATUSES_IGNORE);

			for (size_t j = 1 + (start + 1) % 2; j < N - 1; j += 2)
				y[N + j] = (hh * fun(j * h, (1 + start) * h) +
					y[N + j - 1] + y[N + j + 1] + y[N + j - N] + y[N + j + N]) / coef;

			for (size_t j = 1 + (start + M) % 2; j < N - 1; j += 2)
				y[(M - 2) * N + j] = (hh * fun(j * h, ((M - 2) + start) * h) +
					y[(M - 2) * N + j - 1] + y[(M - 2) * N + j + 1] + y[(M - 2) * N + j - N] + y[(M - 2) * N + j + N]) / coef;
		}

		//локальная ошибка
		double norm_rank = 0.0;
		for (size_t i = N; i < N * M - N; ++i)
			norm_rank += (y[i] - y0[i]) * (y[i] - y0[i]);

		//объединяет значения из всех процессов и распределяет результат обратно во все процессы
		MPI_Allreduce(&norm_rank, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		err = sqrt(err);

		// TODO
		y0.swap(y);
		++count;

	} while (err > eps);
}

//type_message: 1 - Send_plus_Recv, 2 -  Send_recv, 3 - ISecnd_IRecv
void Solver(int type_message, string method)
{
	int rank, commSize;

	MPI_Comm_size(MPI_COMM_WORLD, &commSize);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	size_t count;

	size_t M = (N - 2) / commSize + 2;

	if (rank == commSize - 1)
		M = N - ((N - 2) / commSize) * (commSize - 1);


	vec y0, y;
	y0.resize(N * M);
	y.resize(N * M);

	//с какой строки стартуем (для каждого потока свое)
	size_t start = rank * ((N - 2) / commSize);

	double t1 = MPI_Wtime();
	switch (type_message)
	{
	case 1:
		Send_plus_Recv(method, y, y0, M, count, rank, commSize, start);
		break;
	case 2:
		SendRecv(method, y, y0, M, count, rank, commSize, start);
		break;
	case 3:
		ISend_plus_IRecv(method, y, y0, M, count, rank, commSize, start);
		break;
	}
	double t2 = MPI_Wtime();
	double time = t2 - t1;

	// собираем решение
	vec result;
	result.resize(N * N);
	vector<int> recvcounts, displs;
	recvcounts.resize(commSize);
	displs.resize(commSize);

	for (size_t i = 0; i < commSize - 1; ++i)
		recvcounts[i] = N * ((N - 2) / commSize);

	recvcounts[commSize - 1] = N * (N - (M - 2) * (commSize - 1) - 2);

	for (size_t i = 0; i < commSize; ++i)
		displs[i] = i * N * ((N - 2) / commSize);

	//Собирает данные переменных из всех членов группы в один элемент
	MPI_Gatherv(y0.data() + N, N * (M - 2), MPI_DOUBLE, result.data() + N, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);


	//вывод
	if (rank == 0) {
		switch (type_message)
		{
		case 1:
			if (method == "Jacobi")
				cout << "Send+Recv_Jacobi   commSize = " << commSize << ",   N = " << N << ",   eps = " << eps << ",   time = " << time << ",   count = " << count << ",   error = " << norma(result) << "\n";
			else
				cout << "Send+Recv_Seidel   commSize = " << commSize << ",   N = " << N << ",   eps = " << eps << ",   time = " << time << ",   count = " << count << ",   error = " << norma(result) << "\n";
			break;
		case 2:
			if (method == "Jacobi")
				cout << "SendRecv_Jacobi   commSize = " << commSize << ",   N = " << N << ",   eps = " << eps << ",   time = " << time << ",   count = " << count << ",   error = " << norma(result) << "\n";
			else
				cout << "SendRecv_Seidel  commSize = " << commSize << ",   N = " << N << ",   eps = " << eps << ",   time = " << time << ",   count = " << count << ",   error = " << norma(result) << "\n";
			break;
		case 3:
			if (method == "Jacobi")
				cout << "ISend+IRecv_Jacobi   commSize = " << commSize << ",   N = " << N << ",   eps = " << eps << ",   time = " << time << ",   count = " << count << ",   error = " << norma(result) << "\n";
			else
				cout << "ISend+IRecv_Seidel   commSize = " << commSize << ",   N = " << N << ",   eps = " << eps << ",   time = " << time << ",   count = " << count << ",   error = " << norma(result) << "\n";
			break;
		}
	}

}


int main(int argc, char** argv)
{
	setlocale(LC_ALL, "Russian");


	MPI_Init(&argc, &argv);

	Solver(1, "Jacobi");
	Solver(1, "Seidel");
	Solver(2, "Jacobi");
	Solver(2, "Seidel");
	Solver(3, "Jacobi");
	Solver(3, "Seidel");

	MPI_Finalize();


}

