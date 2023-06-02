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

vector<vector<double>> sigmoid(vector<vector<double>> *x, int size_row, int size_col, bool derivative = false)
{
    vector<vector<double>> output(size_row, vector<double>(size_col, 0.0));
    if (derivative)
    {

        for (int i = 0; i < size_row; i++)
        {
            for (int j = 0; j < size_col; j++)
            {
                output.at(i).at(j) = exp(-x->at(i).at(j)) / (pow(exp(-x->at(i).at(j)) + 1, 2));
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
                output.at(i).at(j) = 1 / (1 + exp(-x->at(i).at(j)));
            }
        }
        return output;
    }
}

vector<vector<double>> relu(vector<vector<double>> *x, int size_row, int size_col, bool derivative = false)
{
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

Layer::Layer(int unidades, string activation_function, int input_shape)
{
    this->numeroNeuronas = unidades;
    this->input_shape = input_shape;
    setActivation(activation_function);
    setWeights(unidades, input_shape);
    setBias(unidades);
    cout << "construye layer con uni=" << unidades << " y input shape: " << input_shape << endl;
}

Layer::~Layer()
{

    if ((this->bias != NULL) && (this->weights != NULL))
    {
        cout << "Entra a destructor de Layer con uni=" << this->numeroNeuronas << " y input_shape=" << this->input_shape << endl;
        delete this->bias;
        this->bias = NULL;
        delete this->weights;
        this->weights = NULL;
    }
    cout << "Sale de destructor de Layer" << endl;
}

void Layer::operator=(const Layer &matriz)
{
    cout << "entra al operador asignación" << endl;
    if (matriz.weights != NULL)
    {
        delete weights;
        this->weights = new vector<vector<double>>;
        this->weights->resize(matriz.getInputShape(), vector<double>(matriz.getNumeroNeuronas(), 0.0));
        for (int i = 0; i < matriz.getInputShape(); i++)
        {
            for (int j = 0; j < matriz.getNumeroNeuronas(); j++)
            {
                this->weights->at(i).at(j) = matriz.weights->at(i).at(j);
            }
        }
    }

    if (matriz.bias != NULL)
    {
        delete bias;
        this->bias = new vector<vector<double>>;
        this->bias->resize(matriz.getNumeroNeuronas(), vector<double>(1, 0.0));
        for (int i = 0; i < matriz.getInputShape(); i++)
        {
            this->bias->at(i).at(0) = matriz.bias->at(i).at(0);
        }
    }

    this->numeroNeuronas = matriz.numeroNeuronas;
    this->input_shape = matriz.input_shape;
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
    weights->resize(prevLayerUni, vector<double>(unidades, 0.0));
    for (int i = 0; i < prevLayerUni; i++)
    {
        for (int j = 0; j < unidades; j++)
        {
            weights->at(i).at(j) = distribution(generator);
        }
    }
}

Matrix Layer::forward(Matrix &input)
{
    Matrix *matrix_weights = new Matrix(*this->weights);
    Matrix *matrix_bias = new Matrix(*this->bias);
    Matrix *raw_output = new Matrix(*(input * (*matrix_weights) + *matrix_bias).getMatrix());
    vector<vector<double>> *z_output = new vector<vector<double>>(activation_function(raw_output->getMatrix(), raw_output->getRows(), raw_output->getCols(), false));
    Matrix output(*z_output);
    delete matrix_weights;
    delete matrix_bias;
    delete raw_output;
    delete z_output;
    return output;
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

vector<vector<double>> (*Layer::getFunction())(vector<vector<double>> *, int, int, bool)
{
    return this->activation_function;
}

void Layer::printWeights() const
{
    cout << "[" << setw(2);
    for (int i = 0; i < this->input_shape; i++)
    {
        for (int j = 0; j < this->numeroNeuronas; j++)
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
