#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <mpi.h>
#include <cmath>

using namespace std;

const int N = 100;  // кол-во временных слоев
int M = 0; // кол-во тел
double T = 20; // интервал
double tau = T / (N - 1); // длина шага
double G = 6.67 * 1e-11; // гравитационная постоянная
const double eps = 1e-8; //точность

struct point {
	point(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {};
	double x;
	double y;
	double z;
};

struct body {
	point dist;
	point speed;
};

typedef vector<double> vec_mass; //вектор масс
typedef vector<body> vec_body; //вектор структур
typedef vector<int> vec_int; //вектор int

point operator+(const point& p1, const point& p2) {
	return { p1.x + p2.x, p1.y + p2.y , p1.z + p2.z };
}
point operator-(const point& p1, const point& p2) {
	return { p1.x - p2.x, p1.y - p2.y , p1.z - p2.z };
}
point operator*(const point& p, double value) {
	return { p.x * value, p.y * value , p.z * value };
}
body operator*(const body& b, double value) {
	return { b.dist * value, b.speed * value };
}
double norm(const point& p1, const point& p2) {
	point temp = p1 - p2;
	return sqrt(pow(temp.x, 2) + pow(temp.y, 2) + pow(temp.z, 2));
}
double degree3(double x) {
	return x * x * x;
}

void Initialization(const string& name, vec_mass& mass, vec_body& body) {
	ifstream file(name);
	file >> M;
	mass.resize(M);
	body.resize(M);

	for (int i = 0; i < M; ++i) {
		file >> mass[i];
		file >> body[i].dist.x >> body[i].dist.y >> body[i].dist.z;
		file >> body[i].speed.x >> body[i].speed.y >> body[i].speed.z;
	}
	file.close();
}

void Clear() {
	for (int i = 0; i < M; ++i) {
		ofstream out;
		out.open("traj___" + to_string(i + 1) + ".txt");
		out.clear();
		out.close();
	}
}

void Result(const vec_body& body, int begin, int end) {
	for (int i = begin; i < end; ++i) {
		ofstream out;
		out.open("traj___" + to_string(i + 1) + ".txt", ios::app);
		out << body[i].dist.x << " " << body[i].dist.y << " " << body[i].dist.z << "\n";
		out.close();
	}
}

body K1(size_t num_of_body, vec_body& body, vec_mass& mass) {
	point k1_dist = body[num_of_body].speed;
	point k1_speed;
	for (size_t k = 0; k < M; ++k)
		k1_speed = k1_speed + (body[num_of_body].dist - body[k].dist) * ((-G) * mass[k] / max(pow(norm(body[num_of_body].dist, body[k].dist), 3), pow(eps, 3)));
	return { k1_dist, k1_speed };
}

body K2(size_t num_of_body, vec_body& k1, vec_body& body, vec_mass& mass) {
	point k2_dist = body[num_of_body].speed + k1[num_of_body].speed * 0.5;
	point k2_speed;
	for (int k = 0; k < M; ++k) {
		k2_speed = k2_speed + ((body[num_of_body].dist + k1[num_of_body].dist * 0.5) - (body[k].dist + k1[k].dist * 0.5)) *
			((-G) * mass[k] / max(pow(norm(body[num_of_body].dist + k1[num_of_body].dist * 0.5, body[k].dist + k1[k].dist * 0.5), 3), pow(eps, 3)));
	}
	return { k2_dist, k2_speed };
}

void Runge_Kutta2(vec_mass& mass, vec_body& bodies, const vec_int& displaces,
	const vec_int& sizes, MPI_Datatype* MPI_BODY, int rank, int commSize) {

	vec_body k1(M);
	vec_body k2(M);
	vec_body temp(bodies);

	auto start = MPI_Wtime();
	int begin = displaces[rank]; // начало обработки каждым потоком
	int end = displaces[rank] + sizes[rank]; // конец обработки каждым потоком
	//Clear();
	//Result(bodies, begin, end);
	for (size_t i = 1; i < N; ++i) {
		for (int j = begin; j < end; ++j)
			k1[j] = K1(j, temp, mass) * tau;

		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, k1.data(), sizes.data(), displaces.data(), *MPI_BODY, MPI_COMM_WORLD);
		for (size_t j = begin; j < end; ++j) {
			k2[j] = K2(j, k1, temp, mass) * tau;
			bodies[j].dist = temp[j].dist + k2[j].dist;
			bodies[j].speed = temp[j].speed + k2[j].speed;
		}
		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), sizes.data(), displaces.data(), *MPI_BODY, MPI_COMM_WORLD);
		//Result(bodies, begin, end);
		bodies.swap(temp);
	}
	if (rank == 0)
		cout << "commSize = " << commSize << ", N = " << N << ", M = " << M << ", T = " << T << ", tau = " << tau << ", time = " << (MPI_Wtime() - start) / N << endl;
}

int main() {

	MPI_Init(NULL, NULL);

	int commSize, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// создаем производный тип структуры
	MPI_Datatype MPI_BODY1;
	const int count1 = 2; // кол-во блоков данных в типе
	vec_int len1 = { 3, 3 }; // массив длин блоков
	vector<MPI_Aint> displ1 = { offsetof(body, dist),
		offsetof(body, speed) }; // массив смещений для блоков
	vector<MPI_Datatype> types1 = { MPI_DOUBLE, MPI_DOUBLE }; // массив типов данных для блоков
	MPI_Type_create_struct(count1, len1.data(), displ1.data(), types1.data(), &MPI_BODY1); // инициализируем структуру
	MPI_Type_commit(&MPI_BODY1); // регистрируем тип

	MPI_Aint l_b;
	MPI_Aint extent;
	MPI_Type_get_extent(MPI_BODY1, &l_b, &extent);

	MPI_Datatype MPI_BODY2;
	const int count2 = 1; // кол-во блоков данных в типе
	vec_int len2 = { 3 }; // массив длин блоков
	vector<MPI_Aint> displ2 = { offsetof(body, dist) }; // массив смещений для блоков
	vector<MPI_Datatype> types2 = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE }; // массив типов данных для блоков
	MPI_Type_create_struct(count2, len2.data(), displ2.data(), types2.data(), &MPI_BODY2); // инициализируем структуру
	MPI_Type_create_resized(MPI_BODY1, l_b, extent, &MPI_BODY2);
	MPI_Type_commit(&MPI_BODY2); // регистрируем тип

	vec_mass mass;
	vec_body bodies;

	// на нулевом ранге читаем информацию о телах
	if (rank == 0)
		//Initialization("4body.txt", mass, bodies);
	//Initialization("init1000.txt", mass, bodies);
//Initialization("init5000.txt", mass, bodies);
Initialization("init10000.txt", mass, bodies);
	//Initialization("init20000.txt", mass, bodies);

	// нулевой процесс отправляет всем число тел
	MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// не нулевые процессы создают для себя векторы
	if (rank != 0) {
		mass.resize(M);
		bodies.resize(M);
	}

	// нулевой процесс отправляет данные всем
	MPI_Bcast(mass.data(), M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(bodies.data(), M, MPI_BODY1, 0, MPI_COMM_WORLD);
	// все процессы знают обо всех данных

	vec_int displaces(commSize);
	vec_int sizes(commSize);

	if (rank == 0) {
		for (int i = 0; i < commSize - 1; ++i) {
			sizes[i] = M / commSize; // сколько тел считает каждый процесс
			displaces[i + 1] = displaces[i] + sizes[i]; // смещения
		}
		sizes[commSize - 1] = M / commSize + M % commSize; //последнему достается остатки тел
	}

	// отправляем с нулевого процесса данные смещений и количество тел для каждого процесса
	MPI_Bcast(displaces.data(), commSize, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(sizes.data(), commSize, MPI_INT, 0, MPI_COMM_WORLD);

	// метод Рунге-Кутта второго порядка
	Runge_Kutta2(mass, bodies, displaces, sizes, &MPI_BODY2, rank, commSize);

	MPI_Finalize();
}