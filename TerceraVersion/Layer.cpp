#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <random>
#include "Layer.h"
#include <string>

using namespace std;

vector<vector<double>> sigmoid(vector<vector<double>> *x, bool derivative = false)
{
    int rows = x->size();
    int cols = x->at(0).size();
    vector<vector<double>> output(rows, vector<double>(cols, 0.0));
    if (derivative)
    {

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                output.at(i).at(j) = exp(-x->at(i).at(j)) / (pow(exp(-x->at(i).at(j)) + 1, 2));
            }
        }
        return output;
    }
    else
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                output.at(i).at(j) = 1 / (1 + exp(-x->at(i).at(j)));
            }
        }
        return output;
    }
}

vector<vector<double>> relu(vector<vector<double>> *x, bool derivative = false)
{
    int size_row = x->size();
    int size_col = x->at(0).size();
    vector<vector<double>> output(size_row, vector<double>(size_col, 0.0));
    if (derivative)
    {
        for (int i = 0; i < size_row; i++)
        {
            for (int j = 0; j < size_col; j++)
            {
                if (x->at(i).at(j) <= 0.0)
                {
                    output.at(i).at(j) = 0.0;
                }
                else
                {
                    output.at(i).at(j) = 1.0;
                }
            }
        }
        return output;
    }
    else
    {
        for (int i = 0; i < size_row; i++)
        {
            for (int j = 0; j < size_col; j++)
            {
                if (x->at(i).at(j) <= 0.0)
                {
                    output.at(i).at(j) = 0.0;
                }
                else
                {
                    output.at(i).at(j) = x->at(i).at(j);
                }
            }
        }
        return output;
    }
}

default_random_engine generator;
normal_distribution<double> distribution(0.0, 2.0);

Layer::Layer(int unidades, string activation_function, int input_shape) : Matrix(unidades, input_shape, 0.0)
{
    setActivation(activation_function);
    setWeights();
    setBias();
    cout << "construye layer con uni=" << unidades << " y input shape: " << input_shape << endl;
}

Layer::Layer(const Layer &layer) : Matrix(layer.getCols(), layer.getRows(), 0.0)
{
    for (int i = 0; i < layer.getRows(); i++)
    {
        for (int j = 0; j < layer.getCols(); j++)
        {
            this->getMatrix()->at(i).at(j) = layer.getMatrix()->at(i).at(j);
        }
    }

    this->bias = new vector<vector<double>>;
    this->bias->resize(layer.getCols(), vector<double>(1, 0.0));
    for (int i = 0; i < layer.getCols(); i++)
    {
        this->bias->at(i).at(0) = layer.bias->at(i).at(0);
    }

    this->setCols(layer.getCols());
    this->setRows(layer.getRows());
}

Layer::~Layer()
{

    if ((this->bias != NULL))
    {
        cout << "Entra a destructor de Layer con uni=" << this->getCols() << " y input_shape=" << this->getRows() << endl;
        delete this->bias;
        this->bias = NULL;
    }
    cout << "Sale de destructor de Layer" << endl;
}

void Layer::operator=(const Layer &matriz)
{
    cout << "entra al operador asignación" << endl;
    if (matriz.getMatrix() != NULL)
    {
        delete this->getMatrix();
        this->matrix = new vector<vector<double>>;
        this->getMatrix()->resize(matriz.getRows(), vector<double>(matriz.getCols(), 0.0));
        for (int i = 0; i < matriz.getRows(); i++)
        {
            for (int j = 0; j < matriz.getCols(); j++)
            {
                this->getMatrix()->at(i).at(j) = matriz.getMatrix()->at(i).at(j);
            }
        }
    }

    if (matriz.getBias() != NULL)
    {
        delete bias;
        this->bias = new vector<vector<double>>;
        this->bias->resize(matriz.getCols(), vector<double>(1, 0.0));
        for (int i = 0; i < matriz.getCols(); i++)
        {
            this->bias->at(i).at(0) = matriz.bias->at(i).at(0);
        }
    }

    this->setCols(matriz.getCols());
    this->setRows(matriz.getRows());
}

Matrix Layer::forward(Matrix &input)
{
    cout << "Entra a la función" << endl;
    Matrix *matrix_weights = new Matrix(*this->matrix);
    cout << "Crea matriz de pesos" << endl;
    Matrix *matrix_bias = new Matrix(*this->bias);
    cout << "Crea bias" << endl;
    Matrix *raw_output = new Matrix(*(input * (*matrix_weights) + *matrix_bias).getMatrix());
    cout << "Crea matriz de raw output" << endl;
    vector<vector<double>> *z_output = new vector<vector<double>>(activation_function(raw_output->getMatrix(), false));
    cout << "Crea vector de de output activada" << endl;
    Matrix output(*z_output);
    cout << "Crea matriz de output" << endl;
    delete matrix_weights;
    delete matrix_bias;
    delete raw_output;
    delete z_output;
    return output;
}

void Layer::setActivation(string activation_function)
{
    if (activation_function == "sigmoid")
    {
        this->activation_function = sigmoid;
    }
    else if (activation_function == "relu")
    {
        this->activation_function = relu;
    }
}

void Layer::setBias()
{
    this->bias = new vector<vector<double>>;
    bias->resize(getCols(), vector<double>(1, 0.0)); // getCols()=numero de neuronas
    for (int i = 0; i < getCols(); i++)
    {
        bias->at(i).at(0) = distribution(generator);
    }
}

void Layer::setWeights()
{
    for (int i = 0; i < getRows(); i++)
    {
        for (int j = 0; j < getCols(); j++)
        {
            getMatrix()->at(i).at(j) = distribution(generator);
        }
    }
}

vector<vector<double>> *Layer::getBias() const
{
    return this->bias;
}

vector<vector<double>> (*Layer::getFunction())(vector<vector<double>> *, bool)
{
    return this->activation_function;
}

void Layer::printBias() const
{
    cout << endl
         << "[" << setw(2);
    for (int i = 0; i < this->getCols(); i++)
    {
        cout << setw(2) << this->getBias()->at(i).at(0);
        cout << endl;
    }
    cout << "]" << endl
         << endl;
}
