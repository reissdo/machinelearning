#include "layer.h"
#include <random>
#include <cassert>
#include <iostream>
#include <string>

Layer::Layer(int inputSize_, int outputSize_, ActivationType activationType_) : activationType(activationType_),
                                                                                weights(outputSize_, inputSize_),
                                                                                bias(outputSize_, 1)
{
}

void Layer::setPreviousLayer(Layer *layer_)
{
    previousLayer = layer_;
}

void Layer::setSubsequentLayer(Layer *layer_)
{
    subsequentLayer = layer_;
}

void Layer::initWeights()
{
    // TODO: add xavier/glorot weight init
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < weights.rows; i++)
    {
        for (int j = 0; j < weights.cols; j++)
        {
            weights.data[i][j] = dist(rng);
        }
    }

    for (int i = 0; i < bias.rows; i++)
    {
        float value = dist(rng);
        for (int j = 0; j < bias.cols; j++)
        {
            bias.data[i][j] = value;
        }
    }
}

void Layer::setInput(Matrix *input_)
{
    input = input_;
}

Matrix *Layer::getActivation()
{
    return activation;
}

Matrix *Layer::getWeights()
{
    return &weights;
}

Matrix *Layer::getError()
{
    return error;
}

void Layer::allocateMatrices(int batchSize, bool training)
{
    weightedInput = new Matrix(weights.rows, batchSize);
    activation = new Matrix(weights.rows, batchSize);

    if (training)
    {
        error = new Matrix(weights.rows, batchSize);
        gradweights = new Matrix(weights.rows, weights.cols);
        gradbias = new Matrix(bias.rows, bias.cols);
    }
}

void Layer::freeMatrices()
{
    delete weightedInput;
    delete activation;
    delete error;
}

void Layer::forward()
{
    if (previousLayer == nullptr)
    {
        assert(input != nullptr);
        matrixMultiply(&weights, input, weightedInput);
    }
    else
    {
        matrixMultiply(&weights, previousLayer->activation, weightedInput);
    }

    matrixVectorAdd(weightedInput, &bias, weightedInput);

    switch (activationType)
    {
    case ActivationType::SIGMOID:
        matrixSigmoid(weightedInput, activation);
        break;

    case ActivationType::RELU:
        matrixReLu(weightedInput, activation);
        break;

    case ActivationType::SOFTMAX:
        matrixSoftMax(weightedInput, activation);
        break;
    }
}

void Layer::print()
{
    std::cout << "weights: " << std::endl;
    weights.print();

    std::cout << "bias: " << std::endl;
    bias.print();
}

void Layer::information()
{
    std::string activationString;
    switch (activationType)
    {
    case ActivationType::SIGMOID:
        activationString = "sigmoid";
        break;

    case ActivationType::RELU:
        activationString = "relu";
        break;

    case ActivationType::SOFTMAX:
        activationString = "softmax";
        break;
    }

    std::cout << "Input Size: " << weights.cols << " Output Size: " << weights.rows << " Activation: " << activationString << std::endl;
}