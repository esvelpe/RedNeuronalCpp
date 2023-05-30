#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Layer
{
public:
    Layer();
    ~Layer();
    void setActivation(string);
    void setWeights();
    void setBias();

private:
    int numeroNeuronas;
    bool input;
    int input_shape[2];
    vector<vector<double>> *weights;
    vector<double> *bias;
    string activation;
};