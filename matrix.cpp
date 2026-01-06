#include "matrix.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

Matrix::Matrix(int rows_, int cols_, float value_) : rows(rows_), cols(cols_), data(rows_, std::vector<float>(cols_, value_))
{
}

Matrix::Matrix(std::vector<std::vector<float>> data_) : data(data_)
{
    assert(data_.size() > 0);

    rows = data_.size();
    cols = data[0].size();
}

void Matrix::getCols(int startIndex, int endIndex, Matrix *out)
{
    assert(startIndex >= 0 && endIndex <= cols && startIndex < endIndex);
    assert(out->rows == rows && out->cols == (endIndex - startIndex));

    for (int i = 0; i < out->rows; i++)
    {
        int col = 0;
        for (int j = startIndex; j < endIndex; j++)
        {
            out->data[i][col] = data[i][j];
            col++;
        }
    }
}

void Matrix::getRows(int startIndex, int endIndex, Matrix *out)
{
    assert(startIndex >= 0 && endIndex <= cols && startIndex < endIndex);
    assert(out->cols == cols && out->rows == (endIndex - startIndex));

    int row = 0;
    for (int i = startIndex; i < endIndex; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[row][j] = data[i][j];
        }
        row++;
    }
}

void Matrix::print()
{
    std::cout << "rows: " << rows << " cols: " << cols << std::endl;
    std::cout << "[";
    for (int i = 0; i < rows; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << "[";

        for (int j = 0; j < cols; j++)
        {
            std::cout << data[i][j];
            if (j < cols - 1)
                std::cout << ",";
        }

        std::cout << "]";
        if (i < rows - 1)
            std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

void Matrix::shape()
{
    std::cout << "[" << rows << "," << cols << "]" << std::endl;
}

void matrixAdd(Matrix *in1, Matrix *in2, Matrix *out)
{
    assert((in1->cols == in2->cols) && (in1->rows == in2->rows) && (in1->cols == out->cols) && (in1->rows == out->rows));

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = in1->data[i][j] + in2->data[i][j];
        }
    }
}

void matrixSubstract(Matrix *in1, Matrix *in2, Matrix *out)
{
    assert((in1->cols == in2->cols) && (in1->rows == in2->rows) && (in1->cols == out->cols) && (in1->rows == out->rows));

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = in1->data[i][j] - in2->data[i][j];
        }
    }
}

void matrixTranspose(Matrix *in, Matrix *out)
{
    assert((in->cols == out->rows) && (in->rows == out->cols));
    assert(in != out);

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = in->data[j][i];
        }
    }
}

void matrixMultiply(Matrix *in1, Matrix *in2, Matrix *out)
{
    // out = in1 * in2
    assert((in1->cols == in2->rows) && (out->rows == in1->rows) && (out->cols == in2->cols));
    assert((in1 != out) && (in2 != out));

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            float dot = 0.0f;
            for (int k = 0; k < in1->cols; k++)
            {
                dot += (in1->data[i][k] * in2->data[k][j]);
            }
            out->data[i][j] = dot;
        }
    }
}

void matrixHadamard(Matrix *in1, Matrix *in2, Matrix *out)
{
    assert((in1->cols == in2->cols) && (in1->rows == in2->rows) && (in1->cols == out->cols) && (in1->rows == out->rows));

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = in1->data[i][j] * in2->data[i][j];
        }
    }
}

void matrixSigmoid(Matrix *in, Matrix *out)
{
    assert((in->rows == out->rows) && (in->cols == out->cols));

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = 1.0f / (1.0f + std::exp(-in->data[i][j]));
        }
    }
}

void matrixReLu(Matrix *in, Matrix *out)
{
    assert((in->rows == out->rows) && (in->cols == out->cols));

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = in->data[i][j] > 0.0f ? in->data[i][j] : 0.0f;
        }
    }
}

void matrixSoftMax(Matrix *in, Matrix *out)
{
    assert((in->rows == out->rows) && (in->cols == out->cols));

    for (int i = 0; i < in->cols; i++)
    {
        float expSum = 0.0f;
        for (int j = 0; j < in->rows; j++)
        {
            expSum += std::exp(in->data[j][i]);
        }

        for (int k = 0; k < in->rows; k++)
        {
            out->data[k][i] = std::exp(in->data[k][i]) / expSum;
        }
    }
}

void matrixCategoricalCrossEntropy(Matrix *in, Matrix *groundtruth, float *cost)
{
    // groundtruth has to be one hot encoded
    assert((in->rows == groundtruth->rows) && (in->cols == groundtruth->cols));
    *cost = 0.0f;

    for (int j = 0; j < in->cols; j++)
    {
        float colCost = 0.0f;
        for (int i = 0; i < in->rows; i++)
        {
            colCost += groundtruth->data[i][j] * std::log(in->data[i][j]);
        }
        colCost *= -1.0f;
        *cost += colCost;
    }

    *cost /= static_cast<float>(in->cols);
}

void matrixCloneCols(Matrix *in, Matrix *out, int size)
{
    assert(size > 0 && size > in->cols && in->cols == 1);
    assert(out->cols == size && out->rows == in->rows);

    for (int i = 0; i < in->rows; i++)
    {
        float temp = in->data[i][0];
        for (int j = 0; j < size; j++)
        {
            out->data[i][j] = temp;
        }
    }
}

void matrixVectorAdd(Matrix *in, Matrix *vec, Matrix *out)
{
    assert(vec->cols == 1 && vec->rows == in->rows);
    assert(in->cols == out->cols && in->rows == out->rows);

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = in->data[i][j] + vec->data[i][0];
        }
    }
}

void matrixScalarMultiply(Matrix *in, float scalar, Matrix *out)
{
    assert(in->cols == out->cols && in->rows == out->rows);

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            out->data[i][j] = in->data[i][j] * scalar;
        }
    }
}

void matrixSum(Matrix *in, float *out)
{
    *out = 0.0f;
    for (int i = 0; i < in->rows; i++)
    {
        for (int j = 0; j < in->cols; j++)
        {
            *out += in->data[i][j];
        }
    }
}

void matrixArgMax(Matrix *in, Matrix *argmax)
{
    // in->data contains only positive entries
    assert(argmax->rows == 2 && argmax->cols == in->cols);

    for (int j = 0; j < in->cols; j++)
    {
        float maxVal = 0.0f;
        int maxIndexI = 0;
        int maxIndexJ = 0;
        for (int i = 0; i < in->rows; i++)
        {
            if (in->data[i][j] > maxVal)
            {
                maxVal = in->data[i][j];
                maxIndexI = i;
                maxIndexJ = j;
            }
        }

        argmax->data[0][j] = maxIndexI;
        argmax->data[1][j] = maxIndexJ;
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

    std::cout << "loading mnist dataset ..." << std::endl;

    std::vector<std::vector<float>> temp_data;
    std::string line;

    while (std::getline(file, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        float value;
        while (ss >> value)
        {
            row.push_back(value);
        }
        if (!row.empty())
        {
            temp_data.push_back(row);
        }
    }

    if (temp_data.empty())
        return nullptr;

    return new Matrix(temp_data);
}

void matrixOneHot(Matrix *in, Matrix *out, int numClasses)
{
    // out->data has to be full of zeros
    assert(in->rows == 1 && in->cols == out->cols);
    assert(out->rows == numClasses);

    for (int i = 0; i < in->cols; i++)
    {
        int classIndex = in->data[0][i];
        out->data[classIndex][i] = 1.0f;
    }
}

void matrixSigmoidDerivative(Matrix *in, Matrix *out)
{
    // sig´(x) = exp(-x) / (1 + exp(-x))^2
    // sig´(x) = sig(x) * (1 - sig(x))
    assert((in->rows == out->rows) && (in->cols == out->cols));

    for (int i = 0; i < out->rows; i++)
    {
        for (int j = 0; j < out->cols; j++)
        {
            // float expIn = std::exp(-in->data[i][j]);
            // out->data[i][j] = expIn / ((1 + expIn) * (1 + expIn));

            float sig = 1.0f / (1.0f + std::exp(-in->data[i][j]));
            out->data[i][j] = sig * (1.0f - sig);
        }
    }
}