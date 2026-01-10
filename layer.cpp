#include "layer.h"
#include <random>
#include <cassert>
#include <iostream>
#include <string>

Layer::Layer(uint inputSize_, uint outputSize_, ActivationType activationType_) : activationType(activationType_),
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

    for (uint i = 0; i < weights.rows; i++)
    {
        for (uint j = 0; j < weights.cols; j++)
        {
            weights.data[i * weights.cols + j] = dist(rng);
        }
    }

    for (uint i = 0; i < bias.rows; i++)
    {
        float value = dist(rng);
        for (uint j = 0; j < bias.cols; j++)
        {
            bias.data[i * bias.cols + j] = value;
        }
    }
}

void Layer::setInput(Matrix *input_)
{
    input = input_;
}

void Layer::setGroundtruth(Matrix *groundtruth_)
{
    groundtruth = groundtruth_;
}

Matrix *Layer::getPredictionActivation()
{
    return predictActivation;
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

Matrix *Layer::getWeightedInput()
{
    return weightedInput;
}

void Layer::allocateMatricesTraining(uint batchSize)
{

    weightedInput = new Matrix(weights.rows, batchSize);
    activation = new Matrix(weights.rows, batchSize);

    gradient = new Matrix(weights.rows, batchSize);
    gradweights = new Matrix(weights.rows, weights.cols);
    gradbias = new Matrix(bias.rows, bias.cols);

    if (subsequentLayer != nullptr)
    {
        assert(subsequentLayer->weights.cols == weights.rows);

        tempSsWeightsT = new Matrix(subsequentLayer->weights.cols, subsequentLayer->weights.rows);
        tempdZdA = new Matrix(weights.rows, batchSize);
    }
    tempPrActivationT = new Matrix(batchSize, weights.cols);
}

void Layer::allocateMatricesPrediction(uint batchSize)
{

    predictionWeightedInput = new Matrix(weights.rows, batchSize);
    predictActivation = new Matrix(weights.rows, batchSize);
}

void Layer::freeMatricesTraining()
{
    delete weightedInput;
    delete activation;
    delete gradient;

    delete gradweights;
    delete gradbias;

    delete tempSsWeightsT;
    delete tempdZdA;
    delete tempPrActivationT;
}

void Layer::freeMatricesPrediction()
{
    delete predictionWeightedInput;
    delete predictActivation;
}

ActivationType Layer::getActivationType()
{
    return activationType;
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

void Layer::predict()
{
    if (previousLayer == nullptr)
    {
        assert(input != nullptr);
        matrixMultiply(&weights, input, predictionWeightedInput);
    }
    else
    {
        matrixMultiply(&weights, previousLayer->predictActivation, predictionWeightedInput);
    }

    matrixVectorAdd(predictionWeightedInput, &bias, predictionWeightedInput);

    switch (activationType)
    {
    case ActivationType::SIGMOID:
        matrixSigmoid(predictionWeightedInput, predictActivation);
        break;

    case ActivationType::RELU:
        matrixReLu(predictionWeightedInput, predictActivation);
        break;

    case ActivationType::SOFTMAX:
        matrixSoftMax(predictionWeightedInput, predictActivation);
        break;
    }
}

void Layer::calculateGradients()
{
    /*
        calculate gradient dL/dz
    */

    Matrix temp1(gradient->rows, gradient->cols);

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

        matrixTranspose(subsequentLayer->getWeights(), tempSsWeightsT);
        matrixMultiply(tempSsWeightsT, subsequentLayer->getGradient(), tempdZdA);

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
        matrixHadamard(tempdZdA, gradient, gradient);
    }

    /*
        calculate gradweights dL/dW and gradbias dL/db
    */

    // bias
    matrixRowMean(gradient, gradbias);

    // weights
    if (previousLayer != nullptr) // hidden layer
    {
        matrixTranspose(previousLayer->getActivation(), tempPrActivationT);
    }
    else // input layer
    {
        matrixTranspose(input, tempPrActivationT);
    }

    matrixMultiply(gradient, tempPrActivationT, gradweights);
    float scalar = 1.0f / static_cast<float>(gradient->cols);
    matrixScalarMultiply(gradweights, scalar, gradweights);
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