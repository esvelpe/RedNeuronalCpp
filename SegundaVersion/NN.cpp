#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "NN.h"
#include "Matrix.h"

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
    this->weights = new vector<Matrix>;
    this->biases = new vector<Matrix>;
    this->functions = new vector<vector<vector<double>> (*)(vector<double> &, int, bool)>;
    this->functions->resize(numCapas);
}

NN::~NN()
{
    delete layers;
    delete weights;
    delete biases;
    delete functions;
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
        // sizes
        int *actualSize = new int;
        int *prevSize = new int;
        *actualSize = layers->at(i).getNumeroNeuronas();
        *prevSize = layers->at(i).getInputShape();

        // bias
        vector<vector<double>> *bias_copy = new vector<vector<double>>;
        bias_copy->resize(*actualSize, vector<double>(1, 0.0));
        Matrix *matrix_bias_copy = new Matrix(*bias_copy);
        Matrix *actual_bias = new Matrix(*layers->at(i).getBias());
        *matrix_bias_copy = *actual_bias;
        this->biases->push_back(*matrix_bias_copy);

        // weights
        vector<vector<double>> *weights_copy = new vector<vector<double>>;
        weights_copy->resize(*actualSize, vector<double>(*prevSize, 0.0));
        Matrix *matrix_copy = new Matrix(*weights_copy);
        Matrix *actual_matrix = new Matrix(*layers->at(i).getWeights());
        *matrix_copy = *actual_matrix;
        this->weights->push_back(*matrix_copy);

        // activation functions
        this->functions->at(i) = layers->at(i).getFunction();

        // delete
        delete weights_copy;
        weights_copy = NULL;
        delete bias_copy;
        bias_copy = NULL;
        delete actualSize;
        actualSize = NULL;
        delete prevSize;
        prevSize = NULL;
        delete matrix_copy;
        matrix_copy = NULL;
        delete matrix_bias_copy;
        matrix_bias_copy = NULL;
        delete actual_matrix;
        actual_matrix = NULL;
        delete actual_bias;
        actual_bias = NULL;
    }
}

void NN::train(Matrix &input, Matrix &real, double learning_rate, int epochs, int num_it = 500)
{
    int cont_epoch = 0;
    while (cont_epoch < epochs)
    {
        for (int i = 0; i < num_it; i++)
        {
            for (int j = 0; j < this->numCapas; j++)
            {
                Matrix raw_output = input * (this->weights->at(j)) + this->biases->at(j);
                Matrix output(this->functions->at(j)(raw_output.getMatrix(), this->biases->at(j).getRows()));
            }
        }
    }
}