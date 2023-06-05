#include <vector>

using namespace std;

class Matrix
{
public:
    Matrix();
    Matrix(int, int, double);
    Matrix(vector<vector<double>> &);
    virtual ~Matrix();
    void T();
    virtual Matrix operator*(const Matrix &);
    virtual Matrix operator+(const Matrix &);
    virtual Matrix operator-(const Matrix &);
    virtual Matrix operator%(const Matrix &);
    virtual void operator=(const Matrix &);
    vector<vector<double>> *getMatrix() const;
    virtual void printMatrix() const;
    int getCols() const;
    int getRows() const;
    void setCols(int);
    void setRows(int);

protected:
    vector<vector<double>> *matrix;

private:
    int n_cols;
    int n_rows;
};