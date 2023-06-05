#include <iostream>
#include <vector>
#include "Layer.h"

using namespace std;

int main()
{
    vector<vector<double>> *matrix1 = new vector<vector<double>>({{-1.0, 2.0, 1.0}, {-2.0, 1.0, 2.0}, {1.0, -1.0, 1.0}});
    vector<vector<double>> *matrix2 = new vector<vector<double>>({{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {1.0, 1.0, 2.0}});
    vector<vector<double>> *matrix3 = new vector<vector<double>>({{1.0}, {2.0}, {1.0}});
    Matrix M1(*matrix1);
    Matrix M2(*matrix2);
    Matrix M3(*matrix3);
    delete matrix1;
    delete matrix2;
    delete matrix3;
    //  M1.printMatrix();
    //  M2.printMatrix();
    //  M3.printMatrix();

    // Matrix M4 = M1 * M2;
    // M4.printMatrix();

    // Matrix M5 = M2 * M3;
    // M5.printMatrix();

    // Matrix M6 = M1 + M2;
    // M6.printMatrix();

    // Matrix M7 = M2 + M3;
    // M7.printMatrix();

    // Matrix M8 = M1 - M2;
    // M8.printMatrix();

    // Matrix M9 = M2 - M3;
    // M9.printMatrix();

    // Matrix M10 = M1 % M2;
    // M10.printMatrix();

    // Matrix M11;
    // M11 = M2;
    // M11.printMatrix();

    // M11 = M3;
    // M11.printMatrix();

    // M2.T();
    // M2.printMatrix();

    Layer layer1(5, "relu", 3);
    layer1.printMatrix();
    layer1.printBias();
    // vector<vector<double>> sol(layer1.getFunction()(matrix1, false));
    // Matrix M_func(sol);
    // M_func.printMatrix();

    Layer layer2(layer1);
    // layer2.printMatrix();
    // layer2.printBias();
    Matrix output = layer2.forward(M1);
    output.printMatrix();

    return 0;
}