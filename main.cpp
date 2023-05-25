#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

using namespace std;
int main()
{
    NeuralNetwork neuralNetwork(2);
    neuralNetwork.addLayer(3);
    neuralNetwork.addLayer(1);

    // Datos de entrenamiento de ejemplo
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    // Hiperparámetros
    double learningRate = 0.1;
    int numIterations = 10000;

    // Entrenamiento
    neuralNetwork.train(inputs, targets, learningRate, numIterations);

    // Prueba con datos de entrada de ejemplo después del entrenamiento
    for (const auto &input : inputs)
    {
        std::vector<double> output = neuralNetwork.forwardPropagation(input);
        std::cout << "Input: " << input[0] << ", " << input[1] << " - Output: " << output[0] << std::endl;
    }

    return 0;
}