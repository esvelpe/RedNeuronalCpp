#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "NN.h"

using namespace std;

vector<double> mse_dev(vector<double> &y, vector<double> &y_hat)
{
    int n = y.size();
    vector<double> diff;
    diff.resize(n, 0.0);
    for (int i = 0; i < n; i++)
    {
        diff.at(i) = y.at(i) - y_hat.at(i);
    }
    return diff;
}
double mse(vector<double> &y, vector<double> &y_hat)
{
    int n = y.size();
    vector<double> *diff = new vector<double>;
    diff->resize(n, 0.0);
    for (int i = 0; i < n; i++)
    {
        diff->at(i) = y.at(i) - y_hat.at(i);
    }
    double suma = accumulate(diff->begin(), diff->end(), 0.0, [](double acc, double x)
                             { return acc + pow(x, 2); });
    return suma / (double)n;
}

NN::NN()
{
    this->numCapas = 0;
    this->layers = new vector<Layer>;
    this->weights = new vector<vector<double>>;
    this->biases = new vector<double>;
}

NN::~NN()
{
    delete layers;
}

void NN::add(Layer &layer)
{
    this->layers->push_back(layer);
    this->numCapas++;
}

void NN::compile()
{
    for (int i = 0; i < this->numCapas; i++)
    {
        vector<vector<double>> *weights_copy = new vector<vector<double>>;
        int *actualSize = new int;
        int *prevSize = new int;
        *actualSize = layers->at(i).getNumeroNeuronas();
        *prevSize = layers->at(i).getInputShape();
        weights_copy->resize(*actualSize, vector<double>(*prevSize, 0.0));

        // this->weights->push_back(layers->at(i).getWeights())
    }
}