#include <iostream>
#include <vector>
#include <cmath>
#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork(int inputSize) : inputSize(inputSize) {}

void NeuralNetwork::addLayer(int outputSize)
{
    if (layers.empty())
    {
        layers.push_back(Layer(inputSize, outputSize));
    }
    else
    {
        int prevLayerOutputSize = layers.back().getOutputSize();
        layers.push_back(Layer(prevLayerOutputSize, outputSize));
    }
}

vector<double> NeuralNetwork::forwardPropagation(const vector<double> &input)
{
    vector<double> output = input;

    for (Layer &layer : layers)
    {
        output = layer.forward(output);
    }

    return output;
}

// void NeuralNetwork::train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, double learningRate, int numIterations)
// {
//     for (int iteration = 0; iteration < numIterations; ++iteration)
//     {
//         for (int i = 0; i < inputs.size(); ++i)
//         {
//             vector<double> input = inputs.at(i);
//             vector<double> target = targets.at(i);

//             // Forward propagation
//             vector<double> output = forwardPropagation(input);

//             // Backpropagation
//             for (int j = layers.size() - 1; j >= 0; --j)
//             {
//                 Layer &layer = layers[j];
//                 vector<double> prevLayerOutput = (j == 0) ? input : layers[j - 1].forward(prevLayerOutput);
//                 layer.backward(prevLayerOutput, output, target, learningRate);
//                 output = prevLayerOutput;
//             }

//             // Update weights
//             for (Layer &layer : layers)
//             {
//                 layer.updateWeights();
//             }
//         }
//     }
// }

void NeuralNetwork::train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets, double learningRate, int numIterations)
{
    for (int iteration = 0; iteration < numIterations; ++iteration)
    {
        for (int i = 0; i < inputs.size(); ++i)
        {
            vector<double> input = inputs.at(i);
            vector<double> target = targets.at(i);

            // Forward propagation
            vector<double> output = forwardPropagation(input);

            // Backpropagation
            for (int j = layers.size() - 1; j >= 0; --j)
            {
                Layer &layer = layers[j];
                if (j == 0)
                {
                    // No hacer nada
                }
                else
                {
                    vector<double> prevLayerOutput = layers[j - 1].forward(prevLayerOutput);
                    layer.backward(prevLayerOutput, output, target, learningRate);
                    output = prevLayerOutput;
                }
            }

            // Update weights
            for (Layer &layer : layers)
            {
                layer.updateWeights();
            }
        }
    }
}