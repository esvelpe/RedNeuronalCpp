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
    vector<vector<double>> *prom;
    prom->resize(n_rows, vector<double>(1, 0.0));
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < y->size(); j++)
        {
            prom->at(i).at(0) += y->at(j).at(i);
        }
        prom->at(i).at(0) = prom->at(i).at(0) / (double)n_rows;
    }
    Matrix mean(*prom);
    delete prom;
    return mean;
}

vector<vector<double>> mse_dev(vector<vector<double>> *y, vector<vector<double>> &y_hat)
{
    int num_rows = y->size();
    vector<vector<double>> diff;
    diff.resize(num_rows, vector<double>(1, 0.0));
    for (int i = 0; i < num_rows; i++)
    {
        diff.at(i).at(0) = y->at(i).at(0) - y_hat.at(i).at(0);
    }
    return diff;
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
    this->functions = new vector<vector<vector<double>> (*)(vector<vector<double>> *, int, int, bool)>;
    this->loss = new vector<double>;
    this->outputs = new vector<Matrix>;
}

NN::~NN()
{
    cout << "Entra al destructorde NN" << endl;
    delete layers;
    delete weights;
    delete d_weights;
    delete biases;
    delete functions;
    delete loss;
    delete outputs;
}

void NN::add(Layer &layer)
{
    Layer layer_copy = layer;
    cout << "Entra a la función add" << endl;
    this->layers->push_back(layer_copy);
    this->numCapas++;
    cout << "Sale de la función add" << endl;
}

void NN::compile()
{
    for (int i = 0; i < this->numCapas; i++)
    {
        cout << "Entro al for" << endl;
        // sizes
        int *num_rows = new int;
        int *num_cols = new int;
        *num_cols = layers->at(i).getNumeroNeuronas();
        *num_rows = layers->at(i).getInputShape();

        // bias
        vector<vector<double>> *bias_copy = new vector<vector<double>>;
        bias_copy->resize(*num_rows, vector<double>(1, 0.0));
        Matrix *matrix_bias_copy = new Matrix(*bias_copy);
        Matrix *actual_bias = new Matrix(*layers->at(i).getBias());
        *matrix_bias_copy = *actual_bias;
        this->biases->push_back(*matrix_bias_copy);

        // weights
        vector<vector<double>> *weights_copy = new vector<vector<double>>;
        weights_copy->resize(*num_rows, vector<double>(*num_cols, 0.0));
        Matrix *matrix_copy = new Matrix(*weights_copy);
        Matrix *actual_matrix = new Matrix(*layers->at(i).getWeights());
        *matrix_copy = *actual_matrix;
        this->weights->push_back(*matrix_copy);

        // d_weights
        this->d_weights->resize(this->numCapas);

        // outputs
        this->outputs->resize(this->numCapas + 1);

        // activation functions
        this->functions->resize(numCapas);
        this->functions->at(i) = layers->at(i).getFunction();

        // delete
        delete weights_copy;
        weights_copy = NULL;
        delete bias_copy;
        bias_copy = NULL;
        delete num_rows;
        num_rows = NULL;
        delete num_cols;
        num_cols = NULL;
        delete matrix_copy;
        matrix_copy = NULL;
        delete matrix_bias_copy;
        matrix_bias_copy = NULL;
        delete actual_matrix;
        actual_matrix = NULL;
        delete actual_bias;
        actual_bias = NULL;
    }
    cout << "Salgo del for" << endl;
}

void NN::train(Matrix &input, Matrix &real, double learning_rate, int epochs, int num_it = 500)
{
    this->loss->resize(epochs, 0.0);
    int cont_epoch = 0;
    while (cont_epoch < epochs)
    {

        for (int i = 0; i < num_it; i++)
        {
            Matrix *output = new Matrix(*input.getMatrix());
            this->outputs->at(0) = input;
            for (int j = 0; j < this->numCapas; j++)
            {
                // Matrix raw_output = (input * (this->weights->at(j))) + this->biases->at(j);
                // vector<vector<double>> z_output=this->functions->at(j)(raw_output.getMatrix(), this->biases->at(j).getRows(),false);
                // Matrix output(z_output);
                // input=output;

                this->outputs->at(j + 1) = this->layers->at(j).forward(*output);
                *output = this->outputs->at(j + 1);
            }

            for (int j = this->numCapas - 1; j >= 0; j--)
            {
                vector<vector<double>> *dz_aux = new vector<vector<double>>;
                dz_aux->resize(this->outputs->at(j + 1).getRows(), vector<double>(this->outputs->at(j + 1).getCols(), 0.0));
                Matrix *dz = new Matrix(*dz_aux);
                delete dz_aux;
                if (j == (this->numCapas - 1))
                {
                    vector<vector<double>> *mse_result = new vector<vector<double>>;
                    *mse_result = mse_dev(this->outputs->at(j + 1).getMatrix(), *real.getMatrix());
                    Matrix *mse_resul = new Matrix(*mse_result);
                    vector<vector<double>> *act_result = new vector<vector<double>>;
                    *act_result = functions->at(j)(this->outputs->at(j + 1).getMatrix(), this->outputs->at(j + 1).getRows(), this->outputs->at(j + 1).getCols(), true);
                    Matrix *act_resul = new Matrix(*act_result);

                    *dz = *mse_resul % *act_resul;

                    this->outputs->at(j).T();
                    this->d_weights->at(j) = (this->outputs->at(j)) * (*dz);
                    delete mse_result;
                    delete mse_resul;
                    delete act_result;
                    delete act_resul;
                }
                else
                {
                    vector<vector<double>> *vec_func = new vector<vector<double>>;
                    *vec_func = functions->at(j)(this->outputs->at(j + 1).getMatrix(), this->outputs->at(j + 1).getRows(), this->outputs->at(j + 1).getCols(), true);
                    Matrix *aux_func = new Matrix(*vec_func);

                    this->weights->at(j + 1).T();
                    *dz = (*dz * this->weights->at(j + 1)) % *aux_func;
                    this->weights->at(j + 1).T();
                    this->outputs->at(j).T();
                    this->d_weights->at(j) = this->outputs->at(j) * *dz;
                    this->outputs->at(j).T();

                    delete vec_func;
                    delete aux_func;
                }
            }
            delete output;
            for (int i = 0; i < numCapas - 1; i++)
            {
                Matrix *lr_aux = new Matrix(this->weights->at(i).getRows(), this->weights->at(i).getCols(), learning_rate);
                this->weights->at(i) = this->weights->at(i) - this->d_weights->at(i) % *lr_aux;
                this->biases->at(i) = this->weights->at(i) - mean(this->weights->at(i).getMatrix()) % *lr_aux;
                delete lr_aux;
            }
        }
        this->loss->at(cont_epoch) = mse(this->outputs->at(this->numCapas - 1).getMatrix(), *real.getMatrix());
        cout << "Loss=" << this->loss->at(cont_epoch) << endl;
    }
}

void NN::predict(Matrix &input)
{
    Matrix *output = new Matrix(*input.getMatrix());
    this->outputs->at(0) = input;
    for (int j = 0; j < this->numCapas; j++)
    {
        this->outputs->at(j + 1) = this->layers->at(j).forward(*output);
        *output = this->outputs->at(j + 1);
    }
    delete output;

    this->outputs->back().printMatrix();
}

void NN::printWeights() const
{
    for (int i = 0; i < this->numCapas; i++)
    {
        this->weights->at(i).printMatrix();
        cout << endl;
    }
}