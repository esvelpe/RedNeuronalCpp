#include <iostream>
#include <vector>
#include <string>
#include "Matrix.h"

using namespace std;

class Layer
{
public:
    Layer(int, string, int);
    Layer(const Layer &);
    ~Layer();
    void initActivation(string);
    void initWeights();
    void initBias();
    void setWeights(Matrix &);
    void setBias(Matrix &);
    void operator=(const Layer &);
    Matrix *getWeights() const;
    Matrix *getBias() const;
    Matrix (*getFunction())(Matrix &, bool);
    int getRows() const;
    int getCols() const;
    void setRows(int);
    void setCols(int);
    Matrix forward(Matrix &);

private:
    int rows;
    int cols;
    Matrix *weights;
    Matrix *bias;
    Matrix (*activation_function)(Matrix &, bool);
};