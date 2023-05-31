#include <vector>
#include <numeric>
#include <cmath>
#include "Layer.h"
#include "Matrix.h"

using namespace std;

class NN
{
public:
    NN();
    ~NN();
    void add(Layer &);
    void compile();
    void train(Matrix &, Matrix &, double, int, int);
    void predict();

private:
    vector<Layer> *layers;
    vector<Matrix> *weights;
    vector<Matrix> *biases;
    vector<vector<vector<double>> (*)(vector<double> &, int, bool)> *functions;
    int numCapas;
};