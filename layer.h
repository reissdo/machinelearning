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
    Matrix *getError();

    void allocateMatrices(int batchSize, bool training);
    void freeMatrices();

    void forward();
    void print();
    void information();

private:
    Matrix *input = nullptr; // only used if layer is input layer

    Matrix weights;
    Matrix bias;

    Matrix *gradweights;
    Matrix *gradbias;

    Matrix *error = nullptr;
    Matrix *weightedInput = nullptr;
    Matrix *activation = nullptr;

    Layer *previousLayer;
    Layer *subsequentLayer;

    ActivationType activationType;
};

#endif