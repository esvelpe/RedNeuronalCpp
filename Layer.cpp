#include <iostream>
#include <vector>
#include <cmath>
#include "Layer.h"

using namespace std;

const string FUNCTION_NAMES[4] = {"lineal", "sigmoid", "tanh", "relu"};
// Definición de la función de activación (por ejemplo, sigmoide)
double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

// Definición de la derivada de la función de activación (para el backpropagation)
double sigmoidDerivative(double x)
{
    double sigmoidX = sigmoid(x);
    return sigmoidX * (1.0 - sigmoidX);
}

// Definición de una capa de la red neuronal
Layer::Layer(int inputSize, int outputSize)
{
    this->activationFunctions[0] = {sigmoid};
    weights = new vector<vector<double>>;
    weights->resize(inputSize, vector<double>(outputSize, 0.0));
    biases = new vector<double>;
    biases->resize(outputSize, 0.0);
    deltaWeights = new vector<vector<double>>;
    deltaWeights->resize(inputSize, vector<double>(outputSize, 0.0));
    deltaBiases = new vector<double>;
    deltaBiases->resize(outputSize, 0.0);
}
Layer::~Layer()
{
    delete weights;
    delete biases;
    delete deltaWeights;
    delete deltaBiases;
}

vector<double> Layer::forward(const vector<double> &input)
{
    vector<double> output(biases->size(), 0.0);

    for (int i = 0; i < biases->size(); ++i)
    {
        for (int j = 0; j < weights->size(); ++j)
        {
            output.at(i) += input.at(j) * weights->at(j).at(i);
        }
        output.at(i) += biases->at(i);
        output.at(i) = sigmoid(output.at(i));
    }

    return output;
}

void Layer::backward(const vector<double> &input, const vector<double> &output, const vector<double> &target, double learningRate)
{
    vector<double> error(biases->size(), 0.0);

    for (int i = 0; i < biases->size(); ++i)
    {
        error.at(i) = (output.at(i) - target.at(i)) * sigmoidDerivative(output.at(i));
        deltaBiases->at(i) = -learningRate * error.at(i);

        for (int j = 0; j < weights->size(); ++j)
        {
            deltaWeights->at(j).at(i) = -learningRate * error.at(i) * input.at(j);
        }
    }
}

void Layer::updateWeights()
{
    for (int i = 0; i < weights->size(); ++i)
    {
        for (int j = 0; j < biases->size(); ++j)
        {
            weights->at(i).at(j) += deltaWeights->at(i).at(j);
        }
    }
    for (int i = 0; i < biases->size(); ++i)
    {
        biases->at(i) += deltaBiases->at(i);
    }
}

double Layer::getOutputSize() const
{
    return biases->size();
}