#ifndef CSV_TO_EIGEN_HPP
#define CSV_TO_EIGEN_HPP

#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

namespace Eigen
{
    // template<typename T, int Rows, int Cols>
    template<typename T>
    // Matrix<T, Dynamic, Dynamic> load_csv (const std::string & path, int Rows, int Cols) {
    void load_csv (const std::string & path, Ref<Matrix<T, Dynamic, Dynamic>> mat) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        // std::vector<T> values;
        int row = 0;
        int col = 0;
        while (std::getline(indata, line)) {
            col = 0;
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                // values.push_back(static_cast<T>(std::stod(cell)));
                mat(row, col) = static_cast<T>(std::stod(cell));
                col ++;
            }
            row ++;
        }
        indata.close();
        // return Map<Matrix<T, Dynamic, Dynamic>>(values.data(), Rows, Cols);
    }
}

#endif