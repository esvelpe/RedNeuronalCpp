#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "Matrix.h"

using namespace std;

Matrix::Matrix(int num_col, int num_row, double init)
{
    this->matrix = new vector<vector<double>>;
    matrix->resize(num_row, vector<double>(num_col, 0.0));
    this->n_cols = num_col;
    this->n_rows = num_row;
}

Matrix::Matrix(vector<vector<double>> &matriz)
{
    this->matrix = new vector<vector<double>>;
    this->matrix->resize(matriz.size(), vector<double>(matriz.at(0).size()));
    for (int i = 0; i < matriz.size(); i++)
    {
        this->matrix->at(i) = vector<double>(matriz.at(i));
    }
    this->n_cols = this->matrix->at(0).size();
    this->n_rows = this->matrix->size();
}

Matrix::~Matrix()
{
    delete matrix;
}

Matrix Matrix::operator*(const Matrix &matriz)
{
    Matrix matriz_operacion(matriz.n_cols, this->n_rows, 0.0);
    if (this->getCols() == matriz.n_rows)
    {
        int n = this->getCols();
        for (int i = 0; i < matriz_operacion.getRows(); i++)
        {
            cout << "entra al primer for" << endl;
            for (int j = 0; j < matriz_operacion.getCols(); j++)
            {
                cout << "entra al segundo for" << endl;
                for (int k = 0; k < n; k++)
                {
                    cout << "entra al tercer for" << endl;
                    matriz_operacion.matrix->at(i).at(j) += this->matrix->at(i).at(k) * matriz.matrix->at(k).at(j);
                }
                cout << "resultado" << matriz_operacion.matrix->at(i).at(j) << endl;
            }
        }
        cout << "Realiza la operacion" << endl;
        return matriz_operacion;
    }
    else
    {
        cout << "Las matrices no son compatibles" << endl;
        exit(1);
    }
}

Matrix Matrix::operator+(const Matrix &matriz)
{
    Matrix matriz_operacion(this->getCols(), this->getRows(), 0.0);
    if ((this->getCols() == matriz.getCols()) && (this->getRows() == matriz.getRows()))
    {
        for (int i = 0; i < this->getRows(); i++)
        {
            for (int j = 0; j < this->getCols(); j++)
            {
                matriz_operacion.matrix->at(i).at(j) = this->matrix->at(i).at(j) + matriz.matrix->at(i).at(j);
            }
        }
        return matriz_operacion;
    }
    else
    {
        cout << "No se puede realizar la operación por las dimensiones" << endl;
        exit(1);
    }
}

void Matrix::operator=(const Matrix &matriz)
{
    for (int i = 0; i < this->getRows(); i++)
    {
        for (int j = 0; j < this->getCols(); j++)
        {
            this->matrix->at(i).at(j) = matriz.matrix->at(i).at(j);
        }
    }
    this->n_cols = matriz.n_cols;
    this->n_rows = matriz.n_rows;
}

void Matrix::T()
{
    vector<vector<double>> *matriz_aux = new vector<vector<double>>;
    matriz_aux->resize(this->getRows(), vector<double>(this->getCols(), 0.0));
    for (int i = 0; i < this->matrix->size(); i++)
    {
        matriz_aux->at(i) = vector<double>(this->matrix->at(i));
    }

    for (int i = 0; i < getRows(); i++)
    {
        for (int j = 0; j < getCols(); j++)
        {
            this->matrix->at(i).at(j) = matriz_aux->at(j).at(i);
        }
    }
    swap(this->n_cols, this->n_rows);

    delete matriz_aux;
}

vector<vector<double>> *Matrix::getMatrix() const
{
    return this->matrix;
}

int Matrix::getCols() const
{
    return this->n_cols;
}

int Matrix::getRows() const
{
    return this->n_rows;
}

void Matrix::printMatrix() const
{
    cout << "[" << setw(2);
    for (int i = 0; i < getRows(); i++)
    {
        for (int j = 0; j < getCols(); j++)
        {
            cout << setw(2) << this->matrix->at(i).at(j);
        }
        cout << endl;
    }
    cout << "]";
}
