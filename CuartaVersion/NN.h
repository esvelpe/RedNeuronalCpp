#include <vector>
#include <numeric>
#include <cmath>
#include "Layer.h"

using namespace std;

class NN
{
public:
    NN();
    ~NN();
    void add(Layer &);
    void compile();
    void train(Matrix &, Matrix &, double, int, int);
    Matrix forward(Matrix &);
    //  void predict(Matrix &);
    //  void printWeights() const;
    vector<Layer> *getLayers() const;
    vector<Matrix> *getBiases() const;
    vector<Matrix> *getWeights() const;
    vector<Matrix> *getDz() const;
    vector<Matrix (*)(Matrix &, bool)> *getFunctions() const;
    int getNumCapas() const;

private:
    vector<Layer> *layers;
    vector<Matrix> *dz;
    vector<Matrix> *weights;
    vector<Matrix> *d_weights;
    vector<Matrix> *biases;
    vector<Matrix> *outputs;
    vector<Matrix (*)(Matrix &, bool)> *functions;
    vector<double> *loss;
    int numCapas;
};