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

vector<vector<double>> sigmoid(vector<double> &x, int size, bool derivative = false)
{
    vector<vector<double>> output(size, vector<double>(1, 0.0));
    if (derivative)
    {

        for (int i = 0; i < size; i++)
        {
            output.at(i).at(0) = exp(-x.at(i)) / (pow(exp(-x.at(i)) + 1, 2));
        }
        return output;
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            output.at(i).at(0) = 1 / (1 + exp(-x.at(i)));
        }
        return output;
    }
}

vector<vector<double>> relu(vector<double> &x, int size, bool derivative = false)
{
    vector<vector<double>> output(size, vector<double>(1, 0.0));
    if (derivative)
    {
        for (int i = 0; i < size; i++)
        {
            if (x.at(i) <= 0.0)
            {
                output.at(i).at(0) = 0.0;
            }
            else
            {
                output.at(i).at(0) = 1.0;
            }
        }
        return output;
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            if (x.at(i) <= 0.0)
            {
                output.at(i).at(0) = 0.0;
            }
            else
            {
                output.at(i).at(0) = x.at(i);
            }
        }
        return output;
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
    this->bias = new vector<vector<double>>;
    bias->resize(unidades, vector<double>(1, 0.0));
    for (int i = 0; i < unidades; i++)
    {
        bias->at(i).at(0) = distribution(generator);
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

vector<vector<double>> *Layer::getBias() const
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

vector<vector<double>> (*Layer::getFunction())(vector<double> &, int, bool)
{
    return this->activation_function;
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
        cout << setw(2) << this->bias->at(i).at(0);
        cout << endl;
    }
    cout << "]";
}
