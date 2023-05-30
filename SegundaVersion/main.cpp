#include <iostream>
#include <vector>
#include "Matrix.h"
#include "Layer.h"

using namespace std;

int main()
{
    vector<vector<double>> matriz1 = {{1.0, 2.0, 1.0}, {2.0, 1.0, 2.0}, {1.0, 1.0, 1.0}};
    Matrix matrix1(matriz1);
    vector<vector<double>> matriz2 = {{1.0}, {2.0}, {5.0}};
    Matrix matrix2(matriz2);
    Matrix matriz3 = matrix1 * matrix2;
    Matrix matriz4 = matriz3 + matrix2;

    matrix1.printMatrix();

    cout << endl;

    matrix2.printMatrix();

    cout << endl;

    matriz3.printMatrix();

    cout << endl;

    matriz4.printMatrix();

    cout << endl;

    matrix1.T();
    matrix1.printMatrix();
    //  matrix1.printMatrix();

    Layer layer1(5, "relu", 10);

    layer1.printBias();

    cout << endl;

    layer1.printWeights();

    cout << endl;

    return 0;
}