#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <vector>
#include <string>

typedef uint32_t uint;

struct Matrix
{
    uint cols;
    uint rows;
    std::vector<float> data; // data[row * cols + col] = data[i * cols + j] = data[i][j]

    Matrix();
    Matrix(uint rows_, uint cols_, float value_ = 0.0f);
    Matrix(uint rows_, uint cols_, std::vector<float> data);

    void getCols(uint startIndex, uint endIndex, Matrix *out);
    void getRows(uint startIndex, uint endIndex, Matrix *out);

    void print();
    std::string shape();
};

/*
    standard matrix operators
*/
void matrixAdd(Matrix *in1, Matrix *in2, Matrix *out);
void matrixSubstract(Matrix *in1, Matrix *in2, Matrix *out);
void matrixTranspose(Matrix *in, Matrix *out);
void matrixMultiply(Matrix *in1, Matrix *in2, Matrix *out);
void matrixHadamard(Matrix *in1, Matrix *in2, Matrix *out);
void matrixVectorAdd(Matrix *in, Matrix *vec, Matrix *out);
void matrixScalarMultiply(Matrix *in, float scalar, Matrix *out);
void matrixSum(Matrix *in, float *out);
void matrixRowMean(Matrix *in, Matrix *out);

/*
    activation functions
*/
void matrixSigmoid(Matrix *in, Matrix *out);
void matrixReLu(Matrix *in, Matrix *out);
void matrixSoftMax(Matrix *in, Matrix *out);

/*
    loss functions
*/
void matrixCategoricalCrossEntropy(Matrix *in, Matrix *groundtruth, float *loss);
void matrixMSE(Matrix *in, Matrix *groundtruth, float *loss);
void matrixLogLoss(Matrix *in, Matrix *groundtruth, float *loss);

/*
    cost functions
*/
// void matrixAccuracy();

/*
    machine learning specific matrix functions
*/
Matrix *matrixLoad(const char *filename);
void matrixArgMax(Matrix *in, Matrix *argmax);
void matrixOneHot(Matrix *in, Matrix *out, uint numClasses);
void matrixPrintMNIST(Matrix *in);

/*
    gradient functions
*/
void matrixSigmoidDerivative(Matrix *activation, Matrix *gradient);
void matrixReLuDerivative(Matrix *wIn, Matrix *gradient);
void matrixSoftMaxCCECombinedDerivative(Matrix *activation, Matrix *groundtruthIndex, Matrix *gradient);

#endif