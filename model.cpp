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

void Model::addCostFunction(CostFunction costFunction_)
{
    costFunction = costFunction_;
}

void Model::allocateLayers(int size, bool training)
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->allocateMatrices(size, training);
    }
}

void Model::freeLayers()
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->freeMatrices();
    }
}

void Model::predict(Matrix *data, Matrix *prediction)
{
    allocateLayers(data->cols, false);
    layers.front()->setInput(data);

    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->forward();
    }

    *prediction = *layers.back()->getActivation();
    freeLayers();
}

void Model::print()
{
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << "layer " << i << std::endl;
        layers[i]->print();
    }
}

void Model::information()
{
    for (int i = 0; i < layers.size(); i++)
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

void Model::train(Matrix *data, Matrix *groundtruth, float learningRate, int batchSize)
{
    // allocateLayers(batchSize);
    // layers.front()->setInput(data);

    // freeLayers();
}

void Model::initTraining()
{
    for (int i = 0; i < layers.size() - 1; i++)
    {
        layers[i]->setSubsequentLayer(layers[i + 1]);
    }
    layers.back()->setSubsequentLayer(nullptr);
}