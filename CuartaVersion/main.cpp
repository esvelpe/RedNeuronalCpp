#include <iostream>
#include <vector>
#include "NN.h"

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
    // layer1.getWeights()->printMatrix();
    // layer1.getBias()->printMatrix();
    Layer layer2(4, "relu", 5);
    // layer2.getWeights()->printMatrix();
    // layer2.getBias()->printMatrix();
    Layer layer3(1, "sigmoid", 4);

    // cout << "USO DE LA FUNCIÓN DE ACTIVACIÓN" << endl;
    // Matrix sol(layer1.getFunction()(M1, false));
    // cout << "Sigue imprimir" << endl;
    // sol.printMatrix();
    // cout << endl;

    // cout << "COPIA DE LAYER" << endl;

    // Layer layer2(layer1);
    // layer2.getWeights()->printMatrix();
    // layer2.getBias()->printMatrix();
    // Matrix output(layer1.forward(M1));
    // output.printMatrix();
    NN model;
    model.add(layer1);
    model.add(layer2);
    model.add(layer3);
    // cout << "tamaño layers" << endl;
    // // model.getLayers()->at(0).getWeights()->printMatrix();
    model.compile();
    // cout << "tamaño weights= " << model.getWeights()->size() << endl;
    // cout << "tamaño biases= " << model.getBiases()->size() << endl;
    // model.getWeights()->at(1).printMatrix();
    // model.getBiases()->at(0).printMatrix();
    // cout << "tamaño arreglo de funciones" << model.getFunctions()->size() << endl;

    vector<vector<double>> train = {{1.0, 2.0, 1.0}, {2.0, 1.0, 2.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {1.0, 1.0, 2.0}};
    Matrix m_train(train);
    vector<vector<double>> target = {{1.0}, {0.0}, {0.0}, {0.0}, {1.0}, {0.0}};
    Matrix m_target(target);

    model.train(m_train, m_target, 0.1, 2, 100);

    return 0;
}