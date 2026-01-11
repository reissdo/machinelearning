#include "model.h"
#include <iostream>
#include <iomanip>

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

void Model::allocateLayersTraining(int size)
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->allocateMatricesTraining(size);
    }
}

void Model::allocateLayersPrediction(int size)
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->allocateMatricesPrediction(size);
    }
}

void Model::freeLayersTraining()
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->freeMatricesTraining();
    }
}

void Model::freeLayersPrediction()
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->freeMatricesPrediction();
    }
}

void Model::predict(Matrix *data, Matrix *prediction)
{
    allocateLayersPrediction(data->cols);
    layers.front()->setInput(data);

    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->predict();
    }

    *prediction = *layers.back()->getPredictionActivation();
    freeLayersPrediction();
}

void Model::forward(Matrix *data, Matrix *groundtruth, float *loss)
{
    layers.front()->setInput(data);

    for (int i = 0; i < layers.size(); i++)
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

    for (int i = layers.size() - 1; i >= 0; i--)
    {
        layers[i]->calculateGradients();
    }
}

void Model::step(float learningRate)
{
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->step(learningRate);
    }
}

void Model::print()
{
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << "layer " << i << std::endl;
        layers[i]->print();
    }
}

void Model::printProgress(int epoch, int batch, int batchesPerEpoch, float loss)
{
    if (batch % 2 != 0 && batch < batchesPerEpoch - 1)
        return;

    float progress = static_cast<float>(batch + 1) / static_cast<float>(batchesPerEpoch);
    int barWidth = 20;
    int pos = static_cast<int>(barWidth * progress);

    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";

    std::cout << "\r" << BOLD << "Epoch [" << epoch << "] " << RESET;

    for (int i = 0; i < barWidth; ++i)
    {
        if (i <= pos)
            std::cout << "\u2588";
        else
            std::cout << "\u2592";
    }
    std::cout << " ";

    std::cout << std::fixed << std::setprecision(1) << std::setw(5) << (progress * 100.0) << "% "
              << "| Loss: " << std::setprecision(4) << loss << RESET
              << "          " << std::flush;

    if (batch + 1 == batchesPerEpoch)
    {
        std::cout << std::endl;
        std::cout << std::defaultfloat << std::setprecision(6);
    }
}

void Model::information()
{
    std::cout << "\n";
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << "Layer: " << i << " >> ";
        layers[i]->information();
    }
    std::cout << std::endl;
}

float Model::calculateCost(Matrix *layerOutput, Matrix *groundtruth)
{
    float cost = 0.0f;
    return cost;
}

void Model::initTraining(int batchSize)
{
    for (int i = 0; i < layers.size() - 1; i++)
    {
        layers[i]->setSubsequentLayer(layers[i + 1]);
    }
    layers.back()->setSubsequentLayer(nullptr);

    for (int i = 0; i < layers.size(); i++)
    {
        layers[i]->allocateMatricesTraining(batchSize);
    }
}