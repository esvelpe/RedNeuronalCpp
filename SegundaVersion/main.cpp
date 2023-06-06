#include <iostream>
#include <vector>
#include "NN.h"

using namespace std;

int main()
{
    vector<vector<double>> matriz1 = {{1.0, 2.0, 1.0}, {2.0, 1.0, 2.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {1.0, 1.0, 2.0}};
    Matrix matrix1(matriz1);
    matrix1.printMatrix();
    vector<vector<double>> matriz2 = {{1.0}, {0.0}, {0.0}, {0.0}, {1.0}, {0.0}};
    Matrix matrix2(matriz2);
    matrix2.printMatrix();
    NN model;
    Layer layer1(3, "relu", 6);
    layer1.printWeights();
    Layer layer2(4, "relu", 3);
    layer2.printWeights();
    Layer layer3(1, "sigmoid", 4);
    layer3.printWeights();
    model.add(layer1);
    model.add(layer2);
    model.add(layer3);
    cout << "sigue" << endl;
    // model.compile();
    //  model.printWeights();
    //     model.train(matrix1, matrix2, 0.01, 3, 200);
    //      Matrix matriz3 = matrix1 * matrix2;
    //      Matrix matriz4 = matriz3 + matrix2;

    // matrix1.printMatrix();

    // cout << endl;

    // matrix2.printMatrix();

    // cout << endl;

    // matriz3.printMatrix();

    // cout << endl;

    // matriz4.printMatrix();

    // cout << endl;

    // matrix1.T();
    // matrix1.printMatrix();
    // //  matrix1.printMatrix();

    // Layer layer1(5, "relu", 10);

    // layer1.printBias();

    // cout << endl;

    // layer1.printWeights();

    // cout << endl;

    return 0;
}