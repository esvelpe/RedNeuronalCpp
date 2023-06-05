#include <iostream>
#include <vector>
#include <string>
#include "Matrix.h"

using namespace std;

class Layer : public Matrix
{
public:
    Layer(int, string, int);
    Layer(const Layer &);
    virtual ~Layer();
    void setActivation(string);
    void setWeights();
    void setBias();
    virtual void operator=(const Layer &);
    virtual void printBias() const;
    vector<vector<double>> *getBias() const;
    vector<vector<double>> (*getFunction())(vector<vector<double>> *, bool);
    Matrix forward(Matrix &);

private:
    vector<vector<double>> *bias;
    vector<vector<double>> (*activation_function)(vector<vector<double>> *, bool);
};