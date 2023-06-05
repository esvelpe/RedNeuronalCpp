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

Matrix sigmoid(Matrix &x, bool derivative = false)
{
    int rows = x.getMatrix()->size();
    int cols = x.getMatrix()->at(0).size();
    vector<vector<double>> *out = new vector<vector<double>>(rows, vector<double>(cols, 0.0));
    if (derivative)
    {

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                out->at(i).at(j) = exp(-x.getMatrix()->at(i).at(j)) / (pow(exp(-x.getMatrix()->at(i).at(j)) + 1, 2));
            }
        }
        Matrix output(*out);
        delete out;
        return output;
    }
    else
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                out->at(i).at(j) = 1 / (1 + exp(-x.getMatrix()->at(i).at(j)));
            }
        }
        Matrix output(*out);
        delete out;
        return output;
    }
}

Matrix relu(Matrix &x, bool derivative = false)
{
    int size_row = x.getMatrix()->size();
    int size_col = x.getMatrix()->at(0).size();
    vector<vector<double>> *out = new vector<vector<double>>;
    out->resize(size_row, vector<double>(size_col, 0.0));
    if (derivative)
    {
        for (int i = 0; i < size_row; i++)
        {
            for (int j = 0; j < size_col; j++)
            {
                if (x.getMatrix()->at(i).at(j) <= 0.0)
                {
                    out->at(i).at(j) = 0.0;
                }
                else
                {
                    out->at(i).at(j) = 1.0;
                }
            }
        }
        Matrix output(*out);
        delete out;
        return output;
    }
    else
    {
        for (int i = 0; i < size_row; i++)
        {
            for (int j = 0; j < size_col; j++)
            {
                if (x.getMatrix()->at(i).at(j) <= 0.0)
                {
                    out->at(i).at(j) = 0.0;
                }
                else
                {
                    out->at(i).at(j) = x.getMatrix()->at(i).at(j);
                }
            }
        }
        Matrix output(*out);
        delete out;
        return output;
    }
}

default_random_engine generator;
normal_distribution<double> distribution(0.0, 2.0);

Layer::Layer(int unidades, string activation_function, int input_shape)
{
    setCols(unidades);
    setRows(input_shape);
    initActivation(activation_function);
    initWeights();
    initBias();
}

Layer::Layer(const Layer &layer)
{
    this->weights = new Matrix(layer.getCols(), layer.getRows(), 0.0);
    for (int i = 0; i < layer.getRows(); i++)
    {
        for (int j = 0; j < layer.getCols(); j++)
        {
            this->weights->getMatrix()->at(i).at(j) = layer.getWeights()->getMatrix()->at(i).at(j);
        }
    }

    this->bias = new Matrix(1, layer.getCols(), 0.0);
    for (int i = 0; i < layer.getCols(); i++)
    {
        this->bias->getMatrix()->at(i).at(0) = layer.bias->getMatrix()->at(i).at(0);
    }

    this->activation_function = layer.activation_function;

    this->setCols(layer.getCols());
    this->setRows(layer.getRows());
}

Layer::~Layer()
{

    if ((this->bias != NULL) && (this->weights != NULL))
    {
        delete this->bias;
        this->bias = NULL;
        delete this->weights;
        this->weights = NULL;
    }
}

void Layer::operator=(const Layer &matriz)
{
    if (matriz.getWeights() != NULL)
    {
        delete this->getWeights();
        this->weights = new Matrix(matriz.getCols(), matriz.getRows(), 0.0);
        for (int i = 0; i < matriz.getRows(); i++)
        {
            for (int j = 0; j < matriz.getCols(); j++)
            {
                this->getWeights()->getMatrix()->at(i).at(j) = matriz.getWeights()->getMatrix()->at(i).at(j);
            }
        }
    }

    if (matriz.getBias() != NULL)
    {
        delete bias;
        this->bias = new Matrix(1, matriz.getCols(), 0.0);
        for (int i = 0; i < matriz.getCols(); i++)
        {
            this->getBias()->getMatrix()->at(i).at(0) = matriz.getBias()->getMatrix()->at(i).at(0);
        }
    }

    this->activation_function = matriz.activation_function;

    this->setCols(matriz.getCols());
    this->setRows(matriz.getRows());
}

void Layer::initActivation(string activation_function)
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

void Layer::initBias()
{
    this->bias = new Matrix(1, getCols(), 0.0);
    for (int i = 0; i < getCols(); i++)
    {
        this->bias->getMatrix()->at(i).at(0) = distribution(generator);
    }
}

void Layer::initWeights()
{
    this->weights = new Matrix(getCols(), getRows(), 0.0);
    for (int i = 0; i < getRows(); i++)
    {
        for (int j = 0; j < getCols(); j++)
        {
            this->getWeights()->getMatrix()->at(i).at(j) = distribution(generator);
        }
    }
}

void Layer::setBias(Matrix &bias)
{
    // int n=bias.getMatrix()->size();
    // for (int i=0;i<n;i++){}
    *this->bias = bias;
}

void Layer::setWeights(Matrix &weights)
{
    *this->weights = weights;
}

Matrix *Layer::getBias() const
{
    return this->bias;
}

Matrix *Layer::getWeights() const
{
    return this->weights;
}

int Layer::getCols() const
{
    return this->cols;
}

int Layer::getRows() const
{
    return this->rows;
}

Matrix (*Layer::getFunction())(Matrix &, bool)
{
    return this->activation_function;
}

void Layer::setRows(int rows)
{
    this->rows = rows;
}

void Layer::setCols(int cols)
{
    this->cols = cols;
}

Matrix Layer::forward(Matrix &input)
{
    Matrix m1((input * (*this->getWeights())) + (*this->getBias()));
    return this->activation_function(m1, false);
}
