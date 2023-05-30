#include <vector>

using namespace std;

class Matrix
{
public:
    Matrix(int, int, double);
    Matrix(vector<vector<double>> &);
    ~Matrix();
    void T();
    Matrix operator*(const Matrix &);
    Matrix operator+(const Matrix &);
    void operator=(const Matrix &);
    void printMatrix() const;
    int getCols() const;
    int getRows() const;

private:
    vector<vector<double>> *matrix;
    int size;
    int n_cols;
    int n_rows;
};