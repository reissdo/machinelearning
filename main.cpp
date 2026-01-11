#include "matrix.h"
#include "layer.h"
#include "model.h"
#include <iostream>

void prepData(Matrix *inputTrain, Matrix *inputTest, Matrix **trainData, Matrix **labelsTrain, Matrix **testData, Matrix **labelsTest)
{
    /*
        get labels
    */

    Matrix tempTrainLabels(inputTrain->rows, 1);
    Matrix tempTestLabels(inputTest->rows, 1);

    inputTrain->getCols(0, 1, &tempTrainLabels);
    inputTest->getCols(0, 1, &tempTestLabels);

    *labelsTrain = new Matrix(1, inputTrain->rows);
    matrixTranspose(&tempTrainLabels, *labelsTrain);
    *labelsTest = new Matrix(1, inputTest->rows);
    matrixTranspose(&tempTestLabels, *labelsTest);

    /*
        get data
    */

    Matrix tempTrainData(inputTrain->rows, inputTrain->cols - 1);
    Matrix tempTestData(inputTest->rows, inputTest->cols - 1);

    inputTrain->getCols(1, inputTrain->cols, &tempTrainData);
    inputTest->getCols(1, inputTest->cols, &tempTestData);

    *trainData = new Matrix(inputTrain->cols - 1, inputTrain->rows);
    matrixTranspose(&tempTrainData, *trainData);
    *testData = new Matrix(inputTest->cols - 1, inputTest->rows);
    matrixTranspose(&tempTestData, *testData);
}

int main(void)
{
    /*
        constants
    */

    const int mnistDataSize = 784;
    const int mnistClasses = 10;
    const int epochs = 3;
    const int batchSize = 100;
    const float learningRate = 0.1f;

    /*
        data preparation
    */

    Matrix *train = matrixLoad("../mnist_train.txt");
    Matrix *test = matrixLoad("../mnist_test.txt");

    Matrix *trainData = nullptr;
    Matrix *labelsTrain = nullptr;
    Matrix *testData = nullptr;
    Matrix *labelsTest = nullptr;

    if (train == nullptr || test == nullptr)
    {
        std::cerr << "Error: could not load mnist dataset" << std::endl;
        return 1;
    }

    prepData(train, test, &trainData, &labelsTrain, &testData, &labelsTest);

    Matrix *OHlabelsTrain = new Matrix(mnistClasses, labelsTrain->cols);
    Matrix *OHlabelsTest = new Matrix(mnistClasses, labelsTest->cols);

    matrixOneHot(labelsTrain, OHlabelsTrain, mnistClasses);
    matrixOneHot(labelsTest, OHlabelsTest, mnistClasses);

    std::cout << "Train data: " << trainData->shape() << " " << OHlabelsTrain->shape() << std::endl;
    std::cout << "Test data: " << testData->shape() << " " << OHlabelsTest->shape() << std::endl;

    /*
        model creation
    */

    Model model;

    model.addLayer(new Layer(mnistDataSize, 64, ActivationType::SIGMOID));
    model.addLayer(new Layer(64, 32, ActivationType::SIGMOID));
    model.addLayer(new Layer(32, mnistClasses, ActivationType::SOFTMAX));

    model.information();
    model.initTraining(batchSize);

    /*
        training
    */

    const int numBatches = trainData->cols / batchSize;

    for (int e = 0; e < epochs; e++)
    {
        for (int b = 0; b < numBatches; b++)
        {
            float loss;
            Matrix batch(trainData->rows, batchSize);
            Matrix batchGroundTruthOneHot(mnistClasses, batchSize);

            trainData->getCols(b * batchSize, b * batchSize + batchSize, &batch);
            OHlabelsTrain->getCols(b * batchSize, b * batchSize + batchSize, &batchGroundTruthOneHot);

            model.forward(&batch, &batchGroundTruthOneHot, &loss);
            model.printProgress(e, b, numBatches, loss);

            model.calculateGradients(&batch, &batchGroundTruthOneHot);
            model.step(learningRate);
        }
        std::cout << std::endl;
    }

    /*
        calculate accuracy on test data
    */

    /*
        display and predict a few numbers
    */

    for (int k = 0; k < 10; k++)
    {
        Matrix number(mnistDataSize, 1);
        testData->getCols(k, k + 1, &number);
        matrixPrintMNIST(&number);

        Matrix prediction(mnistClasses, 1);
        Matrix predT(1, mnistClasses);
        model.predict(&number, &prediction);
        matrixTranspose(&prediction, &predT);
        predT.print();

        Matrix argmaxNumber(2, 1);
        matrixArgMax(&prediction, &argmaxNumber);
        std::cout << "predicted number: " << argmaxNumber.data[0] << "\n"
                  << std::endl;
    }

    return 0;
}