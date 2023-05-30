#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Layer
{
public:
    Layer(int, string, int);
    ~Layer();
    void setActivation(string);
    void setWeights(int, int);
    void setBias(int);
    void printWeights() const;
    void printBias() const;
    vector<vector<double>> *getWeights() const;
    vector<double> *getBias() const;
    int getNumeroNeuronas() const;
    int getInputShape() const;

private:
    int numeroNeuronas;
    int input_shape;
    vector<vector<double>> *weights;
    vector<double> *bias;
    double (*activation_function)(double, bool);
};