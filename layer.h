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
    Layer(uint inputSize_, uint outputSize_, ActivationType activationType_);
    void setPreviousLayer(Layer *layer_);
    void setSubsequentLayer(Layer *layer_);
    void initWeights();
    void setInput(Matrix *input_);
    void setGroundtruth(Matrix *groundtruth_);

    Matrix *getPredictionActivation();
    Matrix *getActivation();
    Matrix *getWeights();
    Matrix *getGradient();
    Matrix *getWeightedInput();

    void allocateMatricesTraining(uint batchSize);
    void allocateMatricesPrediction(uint batchSize);
    void freeMatricesTraining();
    void freeMatricesPrediction();

    ActivationType getActivationType();

    void forward();
    void predict();
    void calculateGradients();
    void step(float learningRate);

    void print();
    void information();

private:
    Matrix *input = nullptr;       // only used if layer is input layer
    Matrix *groundtruth = nullptr; // only used if layer is output layer

    Matrix weights;
    Matrix bias;

    // used during training
    Matrix *gradweights = nullptr;
    Matrix *gradbias = nullptr;
    Matrix *gradient = nullptr;
    Matrix *weightedInput = nullptr;
    Matrix *activation = nullptr;

    Matrix *tempSsWeightsT = nullptr;
    Matrix *tempdZdA = nullptr;
    Matrix *tempPrActivationT = nullptr;

    // used during prediction
    Matrix *predictionWeightedInput = nullptr;
    Matrix *predictActivation = nullptr;

    Layer *previousLayer;
    Layer *subsequentLayer;

    ActivationType activationType;
};

#endif