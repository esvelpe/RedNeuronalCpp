#include <iostream>
#include <vector>
#include <cmath>
#include "Layer.h"

class NeuralNetwork
{
public:
    NeuralNetwork(int);
    void addLayer(int);
    vector<double> forwardPropagation(const vector<double> &);
    void train(const vector<vector<double>> &, const vector<vector<double>> &, double, int);

private:
    int inputSize;
    std::vector<Layer> layers;
};