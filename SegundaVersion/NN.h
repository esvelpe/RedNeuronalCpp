#include <vector>
#include <numeric>
#include <cmath>
#include "Layer.h"
// #include "Matrix.h"

using namespace std;

class NN
{
public:
    NN();
    ~NN();
    void add(Layer &);
    void compile();
    void train(Matrix &, Matrix &, double, int, int);
    void predict(Matrix &);
    void printWeights() const;

private:
    vector<Layer> *layers;
    vector<Matrix> *weights;
    vector<Matrix> *d_weights;
    vector<Matrix> *biases;
    vector<Matrix> *outputs;
    vector<vector<vector<double>> (*)(vector<vector<double>> *, int, int, bool)> *functions;
    vector<double> *loss;
    int numCapas;
};