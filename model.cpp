#include "model.h"
#include <iostream>

Model::Model()
{
}

void Model::addLayer(Layer *layer)
{
    if (layers.empty())
    {
        layer->setPreviousLayer(nullptr);
    }
    else
    {
        layer->setPreviousLayer(layers.back());
    }
    layer->initWeights();
    layers.push_back(layer);
}

void Model::allocateLayersTraining(uint size)
{
    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->allocateMatricesTraining(size);
    }
}

void Model::allocateLayersPrediction(uint size)
{
    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->allocateMatricesPrediction(size);
    }
}

void Model::freeLayersTraining()
{
    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->freeMatricesTraining();
    }
}

void Model::freeLayersPrediction()
{
    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->freeMatricesPrediction();
    }
}

void Model::predict(Matrix *data, Matrix *prediction)
{
    allocateLayersPrediction(data->cols);
    layers.front()->setInput(data);

    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->predict();
    }

    *prediction = *layers.back()->getPredictionActivation();
    freeLayersPrediction();
}

void Model::forward(Matrix *data, Matrix *groundtruth, float *loss)
{
    layers.front()->setInput(data);

    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->forward();
    }

    // lossfunction
    switch (layers.back()->getActivationType())
    {
    case ActivationType::RELU:
        matrixMSE(layers.back()->getActivation(), groundtruth, loss);
        break;

    case ActivationType::SIGMOID:
        matrixLogLoss(layers.back()->getActivation(), groundtruth, loss);
        break;

    case ActivationType::SOFTMAX:
        matrixCategoricalCrossEntropy(layers.back()->getActivation(), groundtruth, loss);
        break;
    }
}

void Model::calculateGradients(Matrix *input, Matrix *groundtruth)
{
    layers.back()->setGroundtruth(groundtruth);
    layers.front()->setInput(input);

    for (uint i = layers.size() - 1; i > 0; i--)
    {
        layers[i]->calculateGradients();
    }
}

void Model::step(float learningRate)
{
    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->step(learningRate);
    }
}

void Model::print()
{
    for (uint i = 0; i < layers.size(); i++)
    {
        std::cout << "layer " << i << std::endl;
        layers[i]->print();
    }
}

void Model::information()
{
    for (uint i = 0; i < layers.size(); i++)
    {
        std::cout << "Layer: " << i << " >> ";
        layers[i]->information();
    }
}

float Model::calculateCost(Matrix *layerOutput, Matrix *groundtruth)
{
    float cost = 0.0f;
    return cost;
}

void Model::initTraining(int batchSize)
{
    for (uint i = 0; i < layers.size() - 1; i++)
    {
        layers[i]->setSubsequentLayer(layers[i + 1]);
    }
    layers.back()->setSubsequentLayer(nullptr);

    for (uint i = 0; i < layers.size(); i++)
    {
        layers[i]->allocateMatricesTraining(batchSize);
    }
}

void Model::prepData(Matrix *data, Matrix *trainData, Matrix *trainLabels, Matrix *testData, Matrix *testLabels, float splitRatio)
{
}