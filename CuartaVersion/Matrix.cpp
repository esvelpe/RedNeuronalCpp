#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "Matrix.h"

using namespace std;

Matrix::Matrix()
{
    this->n_cols = 0;
    this->n_rows = 0;
    this->matrix = new vector<vector<double>>;
}

Matrix::Matrix(const Matrix &matriz)
{
    this->matrix = new vector<vector<double>>;
    matrix->resize(matriz.getRows(), vector<double>(matriz.getCols(), 0.0));
    for (int i = 0; i < matriz.getRows(); i++)
    {
        for (int j = 0; j < matriz.getCols(); j++)
        {
            this->getMatrix()->at(i).at(j) = matriz.getMatrix()->at(i).at(j);
        }
    }

    this->setRows(matriz.getRows());
    this->setCols(matriz.getCols());
}

Matrix::Matrix(int num_col, int num_row, double init)
{
    this->matrix = new vector<vector<double>>;
    matrix->resize(num_row, vector<double>(num_col, init));
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
    if (this->matrix != NULL)
    {
        delete matrix;
        matrix = NULL;
    }
}

Matrix Matrix::operator*(const Matrix &matriz)
{
    Matrix matriz_operacion(matriz.getCols(), this->getRows(), 0.0);
    if (this->getCols() == matriz.getRows())
    {
        int n = this->getCols();
        for (int i = 0; i < matriz_operacion.getRows(); i++)
        {
            for (int j = 0; j < matriz_operacion.getCols(); j++)
            {
                for (int k = 0; k < n; k++)
                {
                    matriz_operacion.getMatrix()->at(i).at(j) += double(this->getMatrix()->at(i).at(k) * matriz.getMatrix()->at(k).at(j));
                }
            }
        }
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
                matriz_operacion.getMatrix()->at(i).at(j) = double(this->getMatrix()->at(i).at(j) + matriz.getMatrix()->at(i).at(j));
            }
        }
        return matriz_operacion;
    }
    else if ((this->getCols() == matriz.getRows()) && (matriz.getCols() == 1))
    {
        for (int i = 0; i < this->getRows(); i++)
        {
            for (int j = 0; j < this->getCols(); j++)
            {
                matriz_operacion.getMatrix()->at(i).at(j) = double(this->getMatrix()->at(i).at(j) + matriz.getMatrix()->at(j).at(0));
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

Matrix Matrix::operator-(const Matrix &matriz)
{
    Matrix matriz_operacion(this->getCols(), this->getRows(), 0.0);
    if ((this->getCols() == matriz.getCols()) && (this->getRows() == matriz.getRows()))
    {
        for (int i = 0; i < this->getRows(); i++)
        {
            for (int j = 0; j < this->getCols(); j++)
            {
                matriz_operacion.getMatrix()->at(i).at(j) = double(this->getMatrix()->at(i).at(j) - matriz.getMatrix()->at(i).at(j));
            }
        }
        return matriz_operacion;
    }
    else if ((this->getCols() == matriz.getRows()) && (matriz.getCols() == 1))
    {
        for (int i = 0; i < this->getRows(); i++)
        {
            for (int j = 0; j < this->getCols(); j++)
            {
                matriz_operacion.getMatrix()->at(i).at(j) = double(this->getMatrix()->at(i).at(j) - matriz.getMatrix()->at(j).at(0));
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

Matrix Matrix::operator%(const Matrix &matriz)
{
    Matrix matriz_operacion(this->getCols(), this->getRows(), 0.0);
    for (int i = 0; i < this->getRows(); i++)
    {
        for (int j = 0; j < this->getCols(); j++)
        {
            matriz_operacion.getMatrix()->at(i).at(j) = double(this->getMatrix()->at(i).at(j) * matriz.getMatrix()->at(i).at(j));
        }
    }
    return matriz_operacion;
}

void Matrix::operator=(const Matrix &matriz)
{
    if (matriz.getMatrix() != NULL)
    {
        delete this->matrix;
        this->matrix = new vector<vector<double>>;
        this->matrix->resize(matriz.getRows(), vector<double>(matriz.getCols(), 0.0));
        for (int i = 0; i < matriz.getRows(); i++)
        {
            for (int j = 0; j < matriz.getCols(); j++)
            {
                this->getMatrix()->at(i).at(j) = matriz.getMatrix()->at(i).at(j);
            }
        }
        this->n_cols = matriz.getCols();
        this->n_rows = matriz.getRows();
    }
}

Matrix Matrix::T()
{
    Matrix matrix_transpuesta(this->getRows(), this->getCols(), 0.0);
    for (int i = 0; i < matrix_transpuesta.getRows(); i++)
    {
        for (int j = 0; j < matrix_transpuesta.getCols(); j++)
        {
            matrix_transpuesta.getMatrix()->at(i).at(j) = this->getMatrix()->at(j).at(i);
            // this->getMatrix()->at(i).at(j) = matriz_aux->at(j).at(i);
        }
    }
    return matrix_transpuesta;
}

vector<vector<double>> *Matrix::getMatrix() const
{
    return this->matrix;
}

int Matrix::getCols() const
{
    return this->n_cols;
}

void Matrix::setCols(int cols)
{
    this->n_cols = cols;
}

int Matrix::getRows() const
{
    return this->n_rows;
}

void Matrix::setRows(int rows)
{
    this->n_rows = rows;
}

void Matrix::printMatrix() const
{
    cout << endl
         << "[" << setw(4);
    for (int i = 0; i < getRows(); i++)
    {
        for (int j = 0; j < getCols(); j++)
        {
            cout << setw(12) << this->matrix->at(i).at(j);
        }
        cout << endl;
    }
    cout << "]" << endl
         << endl;
}