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

    /*
        data preparation
    */

    Matrix *data = matrixLoad("../mnist_test.txt");
    if(data == nullptr)
    {
        std::cerr << "Error: could not load mnist dataset" << std::endl;
        return 1;
    }

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
    delete labels;

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
    model.addCostFunction(CostFunction::CCROSSENTROPY);

    model.information();

    /*
        training
    */

    return 0;
}