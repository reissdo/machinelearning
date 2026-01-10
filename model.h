#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "matrix.h"

#include <vector>

class Model
{
public:
    Model();
    void addLayer(Layer *layer);
    void prepData(Matrix *data, Matrix *trainData, Matrix *trainLabels, Matrix *testData, Matrix *testLabels, float splitRatio);

    void initTraining(int batchSize);
    void forward(Matrix *data, Matrix *groundtruth, float *loss);
    void predict(Matrix *data, Matrix *prediction);
    void calculateGradients(Matrix *input, Matrix *groundtruth);
    void step(float learningRate);

    void print();
    void information();

private:
    std::vector<Layer *> layers;

    float calculateCost(Matrix *layerOutput, Matrix *groundtruth);
    void allocateLayersTraining(int size);
    void allocateLayersPrediction(int size);
    void freeLayersTraining();
    void freeLayersPrediction();
};

#endif