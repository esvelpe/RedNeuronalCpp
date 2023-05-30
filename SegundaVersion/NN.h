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
    void train();
    void predict();

private:
    vector<Layer> *layers;
    vector<vector<double>> *weights;
    vector<double> *biases;
    int numCapas;
};