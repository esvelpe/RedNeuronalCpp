#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

using namespace std;

class Matrix
{
public:
    Matrix();
    Matrix(const Matrix &);
    Matrix(int, int, double);
    Matrix(vector<vector<double>> &);
    ~Matrix();
    Matrix T();
    Matrix operator-();
    Matrix operator*(const Matrix &);
    Matrix operator+(const Matrix &);
    Matrix operator-(const Matrix &);
    Matrix operator%(const Matrix &);
    void operator=(const Matrix &);
    vector<vector<double>> *getMatrix() const;
    void printMatrix() const;
    int getCols() const;
    int getRows() const;
    void setCols(int);
    void setRows(int);

private:
    vector<vector<double>> *matrix;
    int n_cols;
    int n_rows;
};

#endif