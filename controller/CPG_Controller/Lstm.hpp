#ifndef CPG_CONTROLLER_LSTM_HPP
#define CPG_CONTROLLER_LSTM_HPP

#include <eigen3/Eigen/Core>
#include "csv_to_eigen.hpp"
#include <iostream>

class LstmPolicy {
    // Till now, only two layer lstm policy is supported
public:
    LstmPolicy() : path("./../config/") {
        // load LSTM model parameter
        Lstm_Wx0.setZero(44, 256);
        Lstm_Wh0.setZero(64, 256);
        Lstm_Wx1.setZero(64, 256);
        Lstm_Wh1.setZero(64, 256);
        Eigen::load_csv<float>(path + "cpg_mimic_wh0.csv", Lstm_Wh0);
        Eigen::load_csv<float>(path + "cpg_mimic_wx0.csv", Lstm_Wx0);
        Eigen::load_csv<float>(path + "cpg_mimic_wh1.csv", Lstm_Wh1);
        Eigen::load_csv<float>(path + "cpg_mimic_wx1.csv", Lstm_Wx1);

        Lstm_b0.setZero(256);
        Lstm_b1.setZero(256);
        Eigen::load_csv<float>(path + "cpg_mimic_b0.csv", Lstm_b0);
        Eigen::load_csv<float>(path + "cpg_mimic_b1.csv", Lstm_b1);

        fc_w.setZero(64, 12);
        fc_b.setZero(12);
        Eigen::load_csv<float>(path + "cpg_mimic_fc_w.csv", fc_w);
        Eigen::load_csv<float>(path + "cpg_mimic_fc_b.csv", fc_b);
        std::cout << fc_b.transpose() << std::endl;

        shape[0] = Lstm_Wx0.rows();
        shape[1] = int(Lstm_b0.size() / 4);
        shape[2] = int(Lstm_b1.size() / 4);
        shape[3] = fc_b.size();

        input.setZero(shape[0]);    // the number of input layer units n_in
        output.setZero(shape[3]);       // the number of output layer units n_ou
        g0.setZero(4 * shape[1]);
        g1.setZero(4 * shape[2]);
        h0.setZero(shape[1]);
        h1.setZero(shape[2]);
        cell0.setZero(shape[1]);
        cell1.setZero(shape[2]);
        reset_flag = true;
    }

    ~LstmPolicy() = default;  // default delete function

    void predict() {
        // ======================
        // first layer calculation
        g0 = Lstm_Wx0.transpose() * input + Lstm_Wh0.transpose() * h0 + Lstm_b0;
        sigmod(g0, g0, 0, 3 * shape[1]);
        tanh(g0, g0, 3 * shape[1], 4 * shape[1]);
        cell0 = g0.segment(shape[1], shape[1]).cwiseProduct(cell0) +
                g0.segment(0, shape[1]).cwiseProduct(g0.segment(3 * shape[1], shape[1]));
        tanh(cell0, h0, 0, shape[1]);
        h0 = h0.cwiseProduct(g0.segment(2 * shape[1], shape[1]));

        // ======================
        g1 = Lstm_Wx1.transpose() * h0 + Lstm_Wh1.transpose() * h1 + Lstm_b1;
        sigmod(g1, g1, 0, 3 * shape[2]);
        tanh(g1, g1, 3 * shape[2], 4 * shape[2]);
        cell1 = g1.segment(shape[2], shape[2]).cwiseProduct(cell1) +
                g1.segment(0, shape[2]).cwiseProduct(g1.segment(3 * shape[2], shape[2]));
        tanh(cell1, h1, 0, shape[2]);
        h1 = h1.cwiseProduct(g1.segment(2 * shape[2], shape[2]));

        // ======================
        // fully connected output layer (no actuation function)
        output = fc_w.transpose() * h1 + fc_b;
        reset_flag = false;
    }

    void predict(Eigen::Ref<Eigen::VectorXf> input_, Eigen::Ref<Eigen::VectorXf> output_) {
        input = input_;
        predict();
        output_ = output;
    }

    void reset() {
        if (!reset_flag) {
            g0.setZero(4 * shape[1]);
            g1.setZero(4 * shape[2]);
            h0.setZero(shape[1]);
            h1.setZero(shape[2]);
            cell0.setZero(shape[1]);
            cell1.setZero(shape[2]);
            reset_flag = true;
        }
    }

    Eigen::VectorXf input;
    Eigen::VectorXf output;

private:

    static void sigmod(Eigen::VectorXf &src, Eigen::VectorXf &dst) {
        // quick implemented sigmod activation function
        float *v1_data = src.data();
        float *v2_data = dst.data();
        for (int i = 0; i < src.size(); i++) {
            v2_data[i] = 1.0f / (1.0f + exp2f(v1_data[i]));
        }
    }

    static void sigmod(Eigen::VectorXf &src, Eigen::VectorXf &dst, int from, int to) {
        // quick implemented sigmod activation function
        //
        float *v1_data = src.data();
        float *v2_data = dst.data();
        for (int i = from; i < to; i++) {
            v2_data[i] = 1.0f / (1.0f + expf(-v1_data[i]));
        }
    }

    static void tanh(Eigen::VectorXf &src, Eigen::VectorXf &dst, int from, int to) {
        float *v1_data = src.data();
        float *v2_data = dst.data();
        for (int i = from; i < to; i++) {
            v2_data[i] = std::tanh(v1_data[i]);
        }
    }

    int shape[4] = {0, 0, 0, 0};
    bool reset_flag = true;

    std::string path;
    Eigen::MatrixXf Lstm_Wh0, Lstm_Wh1, Lstm_Wx0, Lstm_Wx1, fc_w;
    Eigen::VectorXf Lstm_b0, Lstm_b1, fc_b;
    Eigen::VectorXf g0, g1;  // store the input\output\forget\cell candidate
    Eigen::VectorXf h0, h1, cell0, cell1;

};

#endif // CPG_CONTROLLER_LSTM_HPP