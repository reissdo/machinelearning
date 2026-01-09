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
            weights.data[i * weights.cols + j] = dist(rng);
        }
    }

    for (int i = 0; i < bias.rows; i++)
    {
        float value = dist(rng);
        for (int j = 0; j < bias.cols; j++)
        {
            bias.data[i * bias.cols + j] = value;
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

Matrix *Layer::getGradient()
{
    return gradient;
}

void Layer::allocateMatrices(int batchSize, bool training)
{
    weightedInput = new Matrix(weights.rows, batchSize);
    activation = new Matrix(weights.rows, batchSize);

    if (training)
    {
        gradient = new Matrix(weights.rows, batchSize);
        gradweights = new Matrix(weights.rows, weights.cols);
        gradbias = new Matrix(bias.rows, bias.cols);
    }
}

void Layer::freeMatrices()
{
    delete weightedInput;
    delete activation;
    delete gradient;
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

void Layer::calculateGradients()
{
    Matrix temp1(gradient->rows, gradient->cols);
    Matrix temp2(gradient->rows, gradient->cols);

    /*
        calculate gradient dL/dz
    */

    // layer is output layer
    if (subsequentLayer == nullptr)
    {
        assert(groundtruth != nullptr);
        switch (activationType)
        {
        case ActivationType::SIGMOID:
            std::cout << "not supported yet!" << std::endl;
            // TODO: add LogLossDerivative
            break;

        case ActivationType::RELU:
            std::cout << "not supported yet!" << std::endl;
            // TODO: add MSEDerivative
            break;

        case ActivationType::SOFTMAX:
            matrixSoftMaxCCECombinedDerivative(activation, groundtruth, gradient);
            break;
        default:
            std::cout << "wrong activationtype in layer detected" << std::endl;
        }
    }
    else // layer is hidden layer
    {
        matrixTranspose(subsequentLayer->getWeights(), gradient);
        matrixMultiply(gradient, subsequentLayer->getGradient(), &temp1);

        switch (activationType)
        {
        case ActivationType::SIGMOID:
            matrixSigmoidDerivative(activation, gradient);
            break;

        case ActivationType::RELU:
            matrixReLuDerivative(weightedInput, gradient);
            break;

        default:
            std::cout << "wrong activationtype in layer detected" << std::endl;
        }

        matrixHadamard(&temp1, gradient, gradient);
    }

    /*
        calculate gradweights dL/dW and gradbias dL/db
    */

    // bias
    matrixRowMean(gradient, gradbias);

    // weights
    if (previousLayer != nullptr) // hidden layer
    {
        matrixTranspose(previousLayer->getActivation(), &temp1);
    }
    else // input layer
    {
        assert(input != nullptr);
        matrixTranspose(input, &temp1);
    }
    matrixMultiply(gradient, &temp1, &temp2);
    matrixScalarMultiply(&temp2, static_cast<float>(gradient->cols), gradweights);
}

void Layer::step(float learningRate)
{
    matrixScalarMultiply(gradweights, learningRate, gradweights);
    matrixSubstract(&weights, gradweights, &weights);

    matrixScalarMultiply(gradbias, learningRate, gradbias);
    matrixSubstract(&bias, gradbias, &bias);
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