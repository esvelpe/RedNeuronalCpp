#include <iostream>
#include <vector>
#include "NN.h"
#include "data.cpp"

using namespace std;

int main()
{
    Layer layer1(5, "relu", 5);
    Layer layer2(4, "relu", 5);
    Layer layer3(1, "sigmoid", 4);

    NN model;
    model.add(layer1);
    model.add(layer2);
    model.add(layer3);

    model.compile();

    Matrix train = generarMatrizDistribucionNormal(1000, 5, 0, 1);
    Matrix target = generarMatrizDistribucionBinomial(1000, 1, 1, 0.2);

    model.train(train, target, 0.001, 5, 1000);

    cout << endl;

    Matrix input = generarMatrizDistribucionNormal(1, 5, 0, 1);

    cout << "Se ingresará el siguiente vector como input: " << endl;
    input.printMatrix();
    cout << "El modelo predice el siguiente valor: " << endl;
    (model.forwardProp(input)).printMatrix();

    return 0;
}