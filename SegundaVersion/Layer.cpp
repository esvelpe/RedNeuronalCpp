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

default_random_engine generator;
normal_distribution<double> distribution(0.0, 2.0);

double sigmoid(double x, bool derivative = false)
{
    if (derivative)
    {
        return exp(-x) / (pow(exp(-x) + 1, 2));
    }
    else
    {
        return 1 / (1 + exp(-x));
    }
}

double relu(double x, bool derivative = false)
{
    if (derivative)
    {
        if (x <= 0.0)
        {
            return 0.0;
        }
        else
        {
            return 1.0;
        }
    }
    else
    {
        if (x <= 0.0)
        {
            return 0.0;
        }
        else
        {
            return x;
        }
    }
}

Layer::Layer(int unidades, string activation_function, int input_shape)
{
    this->numeroNeuronas = unidades;
    this->input_shape = input_shape;
    setActivation(activation_function);
    setWeights(unidades, input_shape);
    setBias(unidades);
}

Layer::~Layer()
{
    delete bias;
    delete weights;
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

void Layer::setBias(int unidades)
{
    this->bias = new vector<double>;
    bias->resize(unidades, 0.0);
    for (int i = 0; i < unidades; i++)
    {
        bias->at(i) = distribution(generator);
    }
}

void Layer::setWeights(int unidades, int prevLayerUni)
{
    this->weights = new vector<vector<double>>;
    weights->resize(unidades, vector<double>(prevLayerUni, 0.0));
    for (int i = 0; i < unidades; i++)
    {
        for (int j = 0; j < prevLayerUni; j++)
        {
            weights->at(i).at(j) = distribution(generator);
        }
    }
}

vector<vector<double>> *Layer::getWeights() const
{
    return this->weights;
}

vector<double> *Layer::getBias() const
{
    return this->bias;
}

int Layer::getNumeroNeuronas() const
{
    return this->numeroNeuronas;
}

int Layer::getInputShape() const
{
    return this->input_shape;
}

void Layer::printWeights() const
{
    cout << "[" << setw(2);
    for (int i = 0; i < this->numeroNeuronas; i++)
    {
        for (int j = 0; j < this->input_shape; j++)
        {
            cout << setw(10) << this->weights->at(i).at(j);
        }
        cout << endl;
    }
    cout << "]";
}

void Layer::printBias() const
{
    cout << "[" << setw(2);
    for (int i = 0; i < this->numeroNeuronas; i++)
    {
        cout << setw(2) << this->bias->at(i);
        cout << endl;
    }
    cout << "]";
}
