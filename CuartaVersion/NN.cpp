#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "NN.h"

using namespace std;
Matrix mean(vector<vector<double>> *y)
{
    int n_rows = y->at(0).size();
    Matrix mean(1, n_rows, 0.0);
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < y->size(); j++)
        {
            mean.getMatrix()->at(i).at(0) += y->at(j).at(i);
        }
        mean.getMatrix()->at(i).at(0) = mean.getMatrix()->at(i).at(0) / (double)n_rows;
    }
    return mean;
}

Matrix mse_dev(Matrix &y, Matrix &y_hat)
{
    return y - y_hat;
}
double mse(Matrix &y, Matrix &y_hat)
{
    int n = y.getMatrix()->size();
    Matrix diff(y - y_hat);
    double suma = 0.0;
    for (int i = 0; i < n; i++)
    {
        suma += pow(diff.getMatrix()->at(i).at(0), 2);
    }
    return suma / (double)n;
}

NN::NN()
{
    this->numCapas = 0;
    this->layers = new vector<Layer>;
    this->weights = new vector<Matrix>;
    this->d_weights = new vector<Matrix>;
    this->biases = new vector<Matrix>;
    this->dz = new vector<Matrix>;
    this->functions = new vector<Matrix (*)(Matrix &, bool)>;
    this->loss = new vector<double>;
    this->outputs = new vector<Matrix>;
}

NN::~NN()
{
    delete layers;
    layers = NULL;
    delete weights;
    weights = NULL;
    delete d_weights;
    d_weights = NULL;
    delete biases;
    biases = NULL;
    delete functions;
    functions = NULL;
    delete loss;
    loss = NULL;
    delete outputs;
    outputs = NULL;
    delete dz;
    dz = NULL;
}

void NN::add(Layer &layer)
{
    Layer layer_copy(layer);
    this->layers->push_back(layer_copy);
    this->numCapas++;
}

void NN::compile()
{
    // d_weights
    this->d_weights->resize(this->numCapas);

    // outputs
    this->outputs->resize(this->numCapas + 1);

    // dz
    this->dz->resize(this->numCapas);

    // activation functions
    this->functions->resize(this->numCapas);
    for (int i = 0; i < this->numCapas; i++)
    {
        Layer *l = new Layer(layers->at(i));

        // bias
        Matrix *m_bias = new Matrix(*l->getBias());
        this->biases->push_back(*m_bias);

        // weights
        Matrix *m_weights = new Matrix(*l->getWeights());
        this->weights->push_back(*m_weights);

        // functions
        this->functions->at(i) = layers->at(i).getFunction();

        // delete
        delete m_bias;
        m_bias = NULL;
        delete m_weights;
        m_weights = NULL;
    }
}

Matrix NN::forwardProp(Matrix &input)
{
    Matrix output(input);
    for (int j = 0; j < this->numCapas; j++)
    {
        Matrix m1((output * getWeights()->at(j)) + getBiases()->at(j));
        output = this->functions->at(j)(m1, false);
    }
    return output;
}

void NN::train(Matrix &input, Matrix &real, double learning_rate, int epochs, int num_it)
{
    this->loss->resize(epochs, 0.0);
    int cont_epoch = 0;
    this->outputs->at(0) = input;
    while (cont_epoch < epochs)
    {
        cout << "Training the network in epoch= " << cont_epoch << endl;
        for (int i = 0; i < num_it; i++)
        {
            Matrix *output = new Matrix(input);
            for (int j = 0; j < this->numCapas; j++)
            {
                Matrix m1((*output * this->weights->at(j)) + this->biases->at(j));
                this->outputs->at(j + 1) = this->functions->at(j)(m1, false);
                *output = this->outputs->at(j + 1);
                // output->printMatrix();
            }

            for (int j = this->numCapas - 1; j >= 0; j--)
            {

                if (j == (this->numCapas - 1))
                {
                    // this->outputs->at(j + 1).printMatrix();
                    // Matrix *mse_result = new Matrix(mse_dev(this->outputs->at(j + 1), real));
                    // Matrix uno(1, 6, double(-1.0));
                    // uno.printMatrix();
                    Matrix out_aux(this->outputs->at(j + 1));
                    Matrix mse_result(mse_dev(out_aux, real));
                    // cout << "MSE result" << endl;
                    // mse_result.printMatrix();
                    // Matrix mse_result(real - this->outputs->at(j + 1));
                    //  mse_result->printMatrix();
                    //  Matrix *act_resul = new Matrix(functions->at(j)(this->outputs->at(j + 1), true));
                    Matrix act_resul(functions->at(j)(out_aux, true));

                    // dz = new Matrix();
                    this->dz->at(j) = mse_result % act_resul;
                    this->d_weights->at(j) = (this->outputs->at(j).T()) * (this->dz->at(j));
                    // delete mse_result;
                    // mse_result = NULL;
                    // delete act_resul;
                    // act_resul = NULL;
                }
                else
                {
                    Matrix *aux_func = new Matrix(functions->at(j)(this->outputs->at(j + 1), true));

                    this->dz->at(j) = (this->dz->at(j + 1) * this->weights->at(j + 1).T()) % *aux_func;
                    this->d_weights->at(j) = (this->outputs->at(j).T()) * this->dz->at(j);

                    delete aux_func;
                    aux_func = NULL;
                }
            }
            delete output;
            output = NULL;
            for (int j = 0; j < this->numCapas; j++)
            {
                updateWeights(j, learning_rate);
                updateBias(j, learning_rate);
            }
        }
        cout << "|" << setw(5) << "===============" << setw(5) << "===============" << setw(5) << "==============="
             << "|" << endl;
        Matrix out(forwardProp(input));
        this->loss->at(cont_epoch) = mse(out, real);
        cout << "Loss in the actual epoch= " << this->loss->at(cont_epoch) << endl;
        cout << "|" << setw(5) << "===============" << setw(5) << "===============" << setw(5) << "==============="
             << "|" << endl;
        cont_epoch++;
    }
}

void NN::updateWeights(int indice, double learning)
{
    Matrix lr(getDWeights()->at(indice).getCols(), getDWeights()->at(indice).getRows(), learning);
    getWeights()->at(indice) = getWeights()->at(indice) - (getDWeights()->at(indice) % lr);
}

void NN::updateBias(int indice, double learning)
{
    Matrix lr(getDWeights()->at(indice).getCols(), getDWeights()->at(indice).getRows(), learning);
    getBiases()->at(indice) = getBiases()->at(indice) - mean((getDWeights()->at(indice) % lr).getMatrix());
}

vector<Matrix> *NN::getDWeights() const
{
    return this->d_weights;
}

vector<Matrix> *NN::getWeights() const
{
    return this->weights;
}

vector<Matrix> *NN::getBiases() const
{
    return this->biases;
}

vector<Layer> *NN::getLayers() const
{
    return this->layers;
}

vector<Matrix (*)(Matrix &, bool)> *NN::getFunctions() const
{
    return this->functions;
}

int NN::getNumCapas() const
{
    return this->numCapas;
}
