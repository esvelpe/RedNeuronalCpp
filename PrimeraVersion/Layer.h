#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Layer
{
public:
    Layer(int, int);
    ~Layer();
    vector<double> forward(const vector<double> &);
    void backward(const vector<double> &, const vector<double> &, const vector<double> &, double);
    void updateWeights();
    double getOutputSize() const;

private:
    vector<vector<double>> *weights;
    vector<double> *biases;
    vector<vector<double>> *deltaWeights;
    vector<double> *deltaBiases;
    // double (*activationFunctions[4])(double);
};