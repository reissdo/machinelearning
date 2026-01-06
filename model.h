#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "matrix.h"

#include <vector>

enum class CostFunction
{
    LOGLOSS,
    MSE,
    CCROSSENTROPY
};

class Model
{
public:
    Model();
    void addLayer(Layer *layer);
    void addCostFunction(CostFunction costFunction);

    void initTraining();
    void train(Matrix *data, Matrix *groundtruth, float learningRate, int batchSize);
    void predict(Matrix *data, Matrix *prediction);

    void print();
    void information();

private:
    std::vector<Layer *> layers;

    CostFunction costFunction;
    float calculateCost(Matrix *layerOutput, Matrix *groundtruth);
    void allocateLayers(int size, bool training);
    void freeLayers();
};

#endif