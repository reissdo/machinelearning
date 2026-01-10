#include "matrix.h"
#include "layer.h"
#include "model.h"
#include <iostream>

int main(void)
{
    /*
        constants
    */

    const int mnistDataSize = 784;
    const int mnistClasses = 10;
    const int epochs = 10;
    const int batchSize = 200;
    const float learningRate = 0.25;

    /*
        data preparation
    */

    Matrix *data = matrixLoad("../mnist_test.txt");
    if (data == nullptr)
    {
        std::cerr << "Error: could not load mnist dataset" << std::endl;
        return 1;
    }

    // TODO: in model.prepData(Matrix *data, float splitRatio, Matrix *trainingdata, Matrix *testData) auslagern
    Matrix *labelsT = new Matrix(data->rows, 1);
    Matrix *inputsT = new Matrix(data->rows, 784);

    Matrix *labels = new Matrix(1, data->rows);
    Matrix *inputs = new Matrix(784, data->rows);

    data->getCols(0, 1, labelsT);
    data->getCols(1, data->cols, inputsT);
    delete data;

    matrixTranspose(labelsT, labels);
    matrixTranspose(inputsT, inputs);
    delete labelsT;
    delete inputsT;

    Matrix *oneHotLabels = new Matrix(mnistClasses, labels->cols, 0.0f);
    matrixOneHot(labels, oneHotLabels, mnistClasses);

    labels->shape();
    oneHotLabels->shape();
    inputs->shape();

    /*
        model creation
    */

    Model model;

    model.addLayer(new Layer(mnistDataSize, 100, ActivationType::SIGMOID));
    model.addLayer(new Layer(100, 50, ActivationType::SIGMOID));
    model.addLayer(new Layer(50, 20, ActivationType::SIGMOID));
    model.addLayer(new Layer(20, mnistClasses, ActivationType::SOFTMAX));

    model.information();
    model.initTraining(batchSize);

    /*
        training
    */

    const int numBatches = inputs->cols / batchSize;
    std::cout << "number of batches per epoch: " << numBatches << std::endl;

    for (int e = 0; e < epochs; e++)
    {
        for (int b = 0; b < numBatches; b++)
        {
            Matrix batch(inputs->rows, batchSize);
            Matrix batchGroundTruthOneHot(mnistClasses, batchSize);
            Matrix batchGroundTruth(1, batchSize);
            float loss;

            inputs->getCols(b * batchSize, b * batchSize + batchSize, &batch);
            oneHotLabels->getCols(b * batchSize, b * batchSize + batchSize, &batchGroundTruthOneHot);
            labels->getCols(b * batchSize, b * batchSize + batchSize, &batchGroundTruth);

            // TODO: add prediction on test data
            
            model.forward(&batch, &batchGroundTruthOneHot, &loss);
            std::cout << "\repoch: " << e << " loss: " << loss << std::flush;

            model.calculateGradients(&batch, &batchGroundTruth);
            model.step(learningRate);
        }
        std::cout << std::endl;
    }

    Matrix number(mnistDataSize, 1);
    inputs->getCols(5, 6, &number);
    matrixPrintMNIST(&number);

    Matrix prediction(mnistClasses, 1);
    model.predict(&number, &prediction);
    prediction.print();

    return 0;
}