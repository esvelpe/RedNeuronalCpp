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
    int num_rows = y.getMatrix()->size();
    vector<vector<double>> *diff = new vector<vector<double>>;
    diff->resize(num_rows, vector<double>(1, 0.0));
    for (int i = 0; i < num_rows; i++)
    {
        diff->at(i).at(0) = y.getMatrix()->at(i).at(0) - y_hat.getMatrix()->at(i).at(0);
    }
    Matrix output(*diff);
    delete diff;
    return output;
}
double mse(vector<vector<double>> *y, vector<vector<double>> &y_hat)
{
    int n = y->size();
    vector<double> *diff = new vector<double>;
    diff->resize(n, 0.0);
    for (int i = 0; i < n; i++)
    {
        diff->at(i) = y->at(i).at(0) - y_hat.at(i).at(0);
    }
    double suma = accumulate(diff->begin(), diff->end(), 0.0, [](double acc, double x)
                             { return acc + pow(x, 2); });
    delete diff;
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

// Matrix NN::forward(Matrix &input)
// {
//     Matrix m1((input * (*this->getWeights())) + (*this->getBias()));
//     return this->activation_function(m1, false);
// }

void NN::train(Matrix &input, Matrix &real, double learning_rate, int epochs, int num_it = 500)
{
    this->loss->resize(epochs, 0.0);
    int cont_epoch = 0;
    this->outputs->at(0) = input;
    while (cont_epoch < epochs)
    {
        for (int i = 0; i < num_it; i++)
        {
            Matrix *output = new Matrix(input);
            for (int j = 0; j < this->numCapas; j++)
            {
                Matrix m1((*output * this->weights->at(j)) + this->biases->at(j));
                this->outputs->at(j + 1) = this->functions->at(j)(m1, false);
                *output = this->outputs->at(j + 1);
            }
            Matrix *dz;

            for (int j = this->numCapas - 1; j >= 0; j--)
            {

                if (j == (this->numCapas - 1))
                {
                    Matrix *mse_result = new Matrix(mse_dev(this->outputs->at(j + 1), real));
                    Matrix *act_resul = new Matrix(functions->at(j)(this->outputs->at(j + 1), true));

                    // dz = new Matrix();
                    this->dz->at(j) = *mse_result % *act_resul;
                    cout << "this->dz->at(j) = *mse_result  *act_resul;" << endl;
                    this->dz->at(j).printMatrix();

                    this->d_weights->at(j) = (this->outputs->at(j).T()) * (this->dz->at(j));
                    delete mse_result;
                    mse_result = NULL;
                    delete act_resul;
                    act_resul = NULL;
                }
                else
                {
                    Matrix *aux_func = new Matrix(functions->at(j)(this->outputs->at(j + 1), true));
                    cout << "this->outputs->at(j + 1)" << endl;
                    this->outputs->at(j + 1).printMatrix();
                    cout << "aux functions" << endl;
                    aux_func->printMatrix();

                    this->dz->at(j) = (this->dz->at(j + 1) * this->weights->at(j + 1).T()) % *aux_func;
                    cout << "***DZ****" << endl;
                    this->dz->at(j).printMatrix();
                    this->d_weights->at(j) = (this->outputs->at(j).T()) * this->dz->at(j);
                    cout << "this->outputs->at(j).T()" << endl;
                    this->outputs->at(j).T().printMatrix();
                    cout << "*********d_weights*********" << endl;
                    this->d_weights->at(j).printMatrix();
                    cout << "*********d_weights*********" << endl;

                    delete aux_func;
                    aux_func = NULL;
                }
            }
            delete dz;
            dz = NULL;
            delete output;
            output = NULL;
            Matrix *lr_aux;
            cout << "COMIENZA ACTUALIZACIÓN WEIGHTS EN ITERACION= " << i << endl;
            for (int j = 0; j < this->numCapas; j++)
            {
                lr_aux = new Matrix(this->d_weights->at(j).getCols(), this->d_weights->at(j).getRows(), learning_rate);
                lr_aux->printMatrix();
                cout << "WEIGHTS j= " << j << endl;
                this->weights->at(j).printMatrix();
                this->weights->at(j) = this->weights->at(j) - (this->d_weights->at(j) % *lr_aux);
                this->d_weights->at(j).printMatrix();
                (this->d_weights->at(j) % *lr_aux).printMatrix();
                this->weights->at(j).printMatrix();
                this->biases->at(j) = this->biases->at(j) - mean((this->d_weights->at(j) % *lr_aux).getMatrix());
                delete lr_aux;
                lr_aux = NULL;
            }
        }
        this->outputs->back().printMatrix();
        this->loss->at(cont_epoch) = mse(this->outputs->back().getMatrix(), *real.getMatrix());
        cout << "Loss=" << this->loss->at(cont_epoch) << endl;
        cont_epoch++;
    }
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
