#include"help.h"
#include"matrix.h"
#include"lu_vec.h"
//#include"lu_matrix.h"
//#include"lu_points.h"

int main()
{
	setlocale(LC_ALL, "Russian");
	//srand(time(NULL));
	std::cout << "Размерность матрицы:\t" << N << " * " << N << std::endl;
	std::cout << "Размерность блока:\t" << b << std::endl;

	//создание матрицы А (рандомные числа от ... до ...)
	matrix A = create_matrix(N, N);
	vec A_vec = matrix_to_vector(A);
	vec A1_vec = A_vec;
	vec A3_vec = A_vec;
	vec A5_vec = A_vec;

	//print_vec(A_vec, true);


	//LU - разложение (НЕБЛОЧНОЕ)
	/*double time_1 = LU(A1_vec);
	std::cout << "Время НЕБЛОЧНОГО - ПОСЛЕДОВАТЕЛЬНОГО алгоритма:\t" << time_1 << std::endl << std::endl;*/

	for (int thread : {1, 2, 4, 6, 8, 10, 12, 14, 16, 18}) {
		vec A2_vec = A_vec;
		omp_set_num_threads(thread);
		double time_2 = LU_parallel(A2_vec);
		std::cout << "Количество ядер: " << thread << std::endl;
		std::cout << "Время НЕБЛОЧНОГО - ПАРАЛЛЕЛЬНОГО алгоритма:\t" << time_2 << std::endl << std::endl;
	}



	// LU - разложение (БЛОЧНОЕ) версия 1
	/*double time_3 = LU_BLOCK_v1(A3_vec);
	std::cout << "Время БЛОЧНОГО - ПОСЛЕДОВАТЕЛЬНОГО алгоритма:\t" << time_3 << std::endl << std::endl;*/

	for (int thread : {1, 2, 4, 6, 8, 10, 12, 14, 16, 18}) {
		vec A4_vec = A_vec;
		omp_set_num_threads(thread);
		double time_4 = LU_BLOCK_v1_parallel(A4_vec);
		std::cout << "Количество ядер: " << thread << std::endl;
		std::cout << "Время БЛОЧНОГО - ПАРАЛЛЕЛЬНОГО алгоритма:\t" << time_4 << std::endl << std::endl;
	}

	std::cout << std::endl << std::endl << std::endl;

	// LU - разложение (БЛОЧНОЕ) версия 2
	/*double time_5 = LU_BLOCK_v2(A5_vec);
	std::cout << "Время БЛОЧНОГО - ПОСЛЕДОВАТЕЛЬНОГО алгоритма:\t" << time_5 << std::endl << std::endl;*/

	for (int thread : {1, 2, 4, 6, 8, 10, 12, 14, 16, 18}) {
		vec A6_vec = A_vec;
		omp_set_num_threads(thread);
		double time_6 = LU_BLOCK_v2_parallel(A6_vec);
		std::cout << "Количество ядер: " << thread << std::endl;
		std::cout << "Время БЛОЧНОГО - ПАРАЛЛЕЛЬНОГО алгоритма:\t" << time_6 << std::endl << std::endl;
	}

}

