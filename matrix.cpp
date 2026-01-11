#include "matrix.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

Matrix::Matrix() : rows(0), cols(0), data()
{
}

Matrix::Matrix(uint rows_, uint cols_, float value_) : rows(rows_), cols(cols_), data(rows_ * cols_, value_)
{
}

Matrix::Matrix(uint rows_, uint cols_, std::vector<float> data_) : rows(rows_), cols(cols_), data(std::move(data_))
{
}

void Matrix::getCols(uint startIndex, uint endIndex, Matrix *out)
{
    assert(startIndex >= 0 && endIndex <= cols && startIndex < endIndex);
    assert(out->rows == rows);
    assert(out->cols == (endIndex - startIndex));

    for (uint i = 0; i < out->rows; i++)
    {
        uint col = 0;
        for (uint j = startIndex; j < endIndex; j++)
        {
            out->data[i * out->cols + col] = data[i * cols + j];
            col++;
        }
    }
}

void Matrix::getRows(uint startIndex, uint endIndex, Matrix *out)
{
    assert(startIndex >= 0 && endIndex <= cols && startIndex < endIndex);
    assert(out->cols == cols && out->rows == (endIndex - startIndex));

    uint row = 0;
    for (uint i = startIndex; i < endIndex; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[row * out->cols + j] = data[i * cols + j];
        }
        row++;
    }
}

void Matrix::print()
{
    std::cout << "rows: " << rows << " cols: " << cols << std::endl;
    std::cout << "[";
    for (uint i = 0; i < rows; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << "[";

        for (uint j = 0; j < cols; j++)
        {
            std::cout << data[i * cols + j];
            if (j < cols - 1)
                std::cout << ",";
        }

        std::cout << "]";
        if (i < rows - 1)
            std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

std::string Matrix::shape()
{
    return std::string() + "[" + std::to_string(rows) + "," + std::to_string(cols) + "]";
}

void matrixAdd(Matrix *in1, Matrix *in2, Matrix *out)
{
    assert((in1->cols == in2->cols) && (in1->rows == in2->rows) && (in1->cols == out->cols) && (in1->rows == out->rows));

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = in1->data[i * in1->cols + j] + in2->data[i * in2->cols + j];
        }
    }
}

void matrixSubstract(Matrix *in1, Matrix *in2, Matrix *out)
{
    assert((in1->cols == in2->cols) && (in1->rows == in2->rows) && (in1->cols == out->cols) && (in1->rows == out->rows));

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = in1->data[i * in1->cols + j] - in2->data[i * in2->cols + j];
        }
    }
}

void matrixTranspose(Matrix *in, Matrix *out)
{
    assert((in->cols == out->rows) && (in->rows == out->cols));
    assert(in != out);

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = in->data[j * in->cols + i];
        }
    }
}

void matrixMultiply(Matrix *in1, Matrix *in2, Matrix *out)
{
    // out = in1 * in2
    assert((in1->cols == in2->rows) && (out->rows == in1->rows) && (out->cols == in2->cols));
    assert((in1 != out) && (in2 != out));

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            float dot = 0.0f;
            for (uint k = 0; k < in1->cols; k++)
            {
                dot += (in1->data[i * in1->cols + k] * in2->data[k * in2->cols + j]);
            }
            out->data[i * out->cols + j] = dot;
        }
    }
}

void matrixHadamard(Matrix *in1, Matrix *in2, Matrix *out)
{
    assert((in1->cols == in2->cols) && (in1->rows == in2->rows) && (in1->cols == out->cols) && (in1->rows == out->rows));

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = in1->data[i * in1->cols + j] * in2->data[i * in2->cols + j];
        }
    }
}

void matrixSigmoid(Matrix *in, Matrix *out)
{
    assert((in->rows == out->rows) && (in->cols == out->cols));

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = 1.0f / (1.0f + std::exp(-in->data[i * in->cols + j]));
        }
    }
}

void matrixReLu(Matrix *in, Matrix *out)
{
    assert((in->rows == out->rows) && (in->cols == out->cols));

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = in->data[i * in->cols + j] >= 0.0f ? in->data[i * in->cols + j] : 0.0f;
        }
    }
}

void matrixSoftMax(Matrix *in, Matrix *out)
{
    assert((in->rows == out->rows) && (in->cols == out->cols));

    for (uint i = 0; i < in->cols; i++)
    {
        float expSum = 0.0f;
        for (uint j = 0; j < in->rows; j++)
        {
            expSum += std::exp(in->data[j * in->cols + i]);
        }

        for (uint k = 0; k < in->rows; k++)
        {
            out->data[k * out->cols + i] = std::exp(in->data[k * in->cols + i]) / expSum;
        }
    }
}

void matrixCategoricalCrossEntropy(Matrix *in, Matrix *groundtruth, float *loss)
{
    // groundtruth has to be one hot encoded
    assert((in->rows == groundtruth->rows) && (in->cols == groundtruth->cols));
    *loss = 0.0f;

    for (uint j = 0; j < in->cols; j++)
    {
        float colLoss = 0.0f;
        for (uint i = 0; i < in->rows; i++)
        {
            colLoss += groundtruth->data[i * groundtruth->cols + j] * std::log(in->data[i * in->cols + j]);
        }
        colLoss *= -1.0f;
        *loss += colLoss;
    }

    *loss /= static_cast<float>(in->cols);
}

void matrixMSE(Matrix *in, Matrix *groundtruth, float *loss)
{
    assert((in->rows == groundtruth->rows) && (in->cols == groundtruth->cols));
    *loss = 0.0f;

    for (uint j = 0; j < in->cols; j++)
    {
        float colLoss = 0.0f;
        for (uint i = 0; i < in->rows; i++)
        {
            float singleLoss = groundtruth->data[i * groundtruth->cols + j] - in->data[i * in->cols + j];
            colLoss += (singleLoss * singleLoss);
        }
        *loss += colLoss;
    }

    *loss /= static_cast<float>(in->cols);
}

void matrixLogLoss(Matrix *in, Matrix *groundtruth, float *loss)
{
    // groundtruth->data = 0/1
    assert((in->rows == groundtruth->rows) && (in->cols == groundtruth->cols));
    *loss = 0.0f;

    for (uint j = 0; j < in->cols; j++)
    {
        float colLoss = 0.0f;
        for (uint i = 0; i < in->rows; i++)
        {
            float gT = groundtruth->data[i * groundtruth->cols + j];
            float pred = in->data[i * in->cols + j];

            colLoss += (gT * std::log(pred) + (1.0f - gT) * std::log(1.0f - pred));
        }

        *loss += colLoss;
    }

    *loss /= static_cast<float>(in->cols);
    *loss *= -1.0f;
}

void matrixVectorAdd(Matrix *in, Matrix *vec, Matrix *out)
{
    assert(vec->cols == 1 && vec->rows == in->rows);
    assert(in->cols == out->cols && in->rows == out->rows);

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = in->data[i * in->cols + j] + vec->data[i];
        }
    }
}

void matrixScalarMultiply(Matrix *in, float scalar, Matrix *out)
{
    assert(in->cols == out->cols && in->rows == out->rows);

    for (uint i = 0; i < out->rows; i++)
    {
        for (uint j = 0; j < out->cols; j++)
        {
            out->data[i * out->cols + j] = in->data[i * in->cols + j] * scalar;
        }
    }
}

void matrixSum(Matrix *in, float *out)
{
    *out = 0.0f;
    for (uint i = 0; i < in->rows; i++)
    {
        for (uint j = 0; j < in->cols; j++)
        {
            *out += in->data[i * in->cols + j];
        }
    }
}

void matrixArgMax(Matrix *in, Matrix *argmax)
{
    // in->data contains only positive entries
    assert(argmax->rows == 2 && argmax->cols == in->cols);

    for (uint j = 0; j < in->cols; j++)
    {
        float maxVal = 0.0f;
        uint maxIndexI = 0;
        uint maxIndexJ = 0;
        for (uint i = 0; i < in->rows; i++)
        {
            if (in->data[i * in->cols + j] > maxVal)
            {
                maxVal = in->data[i * in->cols + j];
                maxIndexI = i;
                maxIndexJ = j;
            }
        }

        argmax->data[j] = maxIndexI;
        argmax->data[argmax->cols + j] = maxIndexJ;
    }
}

Matrix *matrixLoad(const char *filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return nullptr;
    }

    std::cout << "loading dataset ..." << std::endl;

    std::vector<float> temp_data;
    std::string line;

    uint count = 0;
    uint cols = -1;
    uint rows = -1;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        float value;
        while (ss >> value)
        {
            count++;
            temp_data.push_back(value);
        }

        if (cols == -1)
        {
            cols = count;
        }
    }

    rows = count / cols;
    return new Matrix(rows, cols, temp_data);
}

void matrixPrintMNIST(Matrix *in)
{
    assert(in->cols == 1 && in->rows == 784);

    for (uint i = 0; i < 28; i++)
    {
        for (uint j = 0; j < 28; j++)
        {
            uint val = static_cast<uint>(in->data[j + i * 28]);
            uint grayIdx = 232 + (val * 23 / 255);
            std::cout << "\033[48;5;" << grayIdx << "m  \033[0m";
        }
        std::cout << "\n";
    }
}

void matrixOneHot(Matrix *in, Matrix *out, uint numClasses)
{
    // out->data has to be full of zeros
    assert(in->rows == 1 && in->cols == out->cols);
    assert(out->rows == numClasses);

    for (uint i = 0; i < in->cols; i++)
    {
        uint classIndex = in->data[i];
        out->data[classIndex * out->cols + i] = 1.0f;
    }
}

void matrixSigmoidDerivative(Matrix *activation, Matrix *gradient)
{
    // sig´(x) = exp(-x) / (1 + exp(-x))^2
    // sig´(x) = sig(x) * (1 - sig(x))
    // sig(x) -> activation
    assert((activation->rows == gradient->rows) && (activation->cols == gradient->cols));

    for (uint i = 0; i < gradient->rows; i++)
    {
        for (uint j = 0; j < gradient->cols; j++)
        {
            float sigmoid = activation->data[i * activation->cols + j];
            gradient->data[i * gradient->cols + j] = sigmoid * (1.0f - sigmoid);
        }
    }
}

void matrixReLuDerivative(Matrix *wIn, Matrix *gradient)
{
    assert((wIn->rows == gradient->rows) && (wIn->cols == gradient->cols));

    for (uint i = 0; i < gradient->rows; i++)
    {
        for (uint j = 0; j < gradient->cols; j++)
        {
            wIn->data[i * wIn->cols + j] >= 0.0f ? gradient->data[i * gradient->cols + j] = 1.0f : gradient->data[i * gradient->cols + j] = 0.0f;
        }
    }
}

void matrixSoftMaxCCECombinedDerivative(Matrix *activation, Matrix *groundtruth, Matrix *gradient)
{
    // CCE(z_i) = ln(exp(z_1) + ... + exp(z_I)) - z_k; k is the index for the true class
    // dCCE/dz_k = softmax(z_k) - 1
    // dCCE/dz_j = softmax(z_j)
    // softmax(z) -> activation
    // groundtruth is one hot encoded

    assert(activation->cols == gradient->cols && activation->rows == gradient->rows);
    assert(groundtruth->rows == activation->rows);

    for (uint j = 0; j < gradient->cols; j++)
    {
        for (uint i = 0; i < gradient->rows; i++)
        {
            gradient->data[i * gradient->cols + j] = activation->data[i * activation->cols + j] - groundtruth->data[i * groundtruth->cols + j];
        }
    }
}

void matrixRowMean(Matrix *in, Matrix *out)
{
    assert(out->cols == 1 && in->rows == out->rows);

    float cols = static_cast<float>(in->cols);
    for (uint i = 0; i < out->rows; i++)
    {
        float sum = 0.0f;
        for (uint j = 0; j < in->cols; j++)
        {
            sum += in->data[i * in->cols + j];
        }

        out->data[i] = sum / cols;
    }
}