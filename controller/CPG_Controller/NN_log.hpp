#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Dense>

int WriteLog(std::string filename, Eigen::Ref<Eigen::MatrixXf> mat)
{
    int r = mat.rows();
    int c = mat.cols();
    
    std::ofstream outfile;
    outfile.open(filename, std::ios::app); 
    if (outfile.is_open()) {
        std::stringstream ss;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                ss << mat(i, j) << " ";
            }
            ss << std::endl;
        }
        outfile << ss.str();
        outfile.close();  
        return 0;
    } else {
        return 1;
    }
}