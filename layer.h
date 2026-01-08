#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

enum class ActivationType
{
    SIGMOID,
    RELU,
    SOFTMAX
};

class Layer
{
public:
    Layer(int inputSize_, int outputSize_, ActivationType activationType_);
    void setPreviousLayer(Layer *layer_);
    void setSubsequentLayer(Layer *layer_);
    void initWeights();
    void setInput(Matrix *input_);

    Matrix *getActivation();
    Matrix *getWeights();
    Matrix *getGradient();

    void allocateMatrices(int batchSize, bool training);
    void freeMatrices();

    void forward();
    void calculateGradients();
    void step(float learningRate);

    void print();
    void information();

private:
    Matrix *input = nullptr; // only used if layer is input layer
    Matrix *groundtruth = nullptr; // only used if layer is output layer

    Matrix weights;
    Matrix bias;

    Matrix *gradweights;
    Matrix *gradbias;

    Matrix *gradient = nullptr;
    Matrix *weightedInput = nullptr;
    Matrix *activation = nullptr;

    Layer *previousLayer;
    Layer *subsequentLayer;

    ActivationType activationType;
};

#endif