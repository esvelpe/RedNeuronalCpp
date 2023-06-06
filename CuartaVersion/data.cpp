#include <iostream>
#include <random>
#include <vector>
#include "Matrix.h"

// Función para generar una matriz con números aleatorios siguiendo una distribución normal
Matrix generarMatrizDistribucionNormal(int filas, int columnas, double media, double desviacion)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(media, desviacion);

    Matrix matriz(columnas, filas, 0.0);

    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            matriz.getMatrix()->at(i).at(j) = dist(gen);
        }
    }

    return matriz;
}

// Función para generar una matriz con números aleatorios siguiendo una distribución binomial
Matrix generarMatrizDistribucionBinomial(int filas, int columnas, int n, double p)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::binomial_distribution<int> dist(n, p);

    Matrix matriz(columnas, filas, 0.0);

    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            matriz.getMatrix()->at(i).at(j) = dist(gen);
        }
    }

    return matriz;
}