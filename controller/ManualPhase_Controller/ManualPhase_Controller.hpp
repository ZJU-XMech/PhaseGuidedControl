#ifndef BLACKPANTHER_MP_CONTROLLER_HPP
#define BLACKPANTHER_MP_CONTROLLER_HPP

#include <RobotController.h>
#include "ManualPhase_UserParameters.h"
#include "NN_IO_t.hpp"
#include "command_phase_lcmt.hpp"
#include <lcm/lcm-cpp.hpp>
#include "Lstm.hpp"
#include "ManualPhase.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <random>
#include <vector>
//#include <time.h>
#include <mutex>

#define PI 3.1415926f
#define Height_Filter_Ratio 0.999f
#define ThresholdRatio 0.6f
#define LIGHT_GREEN "\033[1;32m"
//#define BeltGear 2.4025f
#define LIGHT_BLUE "\033[1;34m"

class MP_Controller : public RobotController {
public:
    MP_Controller() : RobotController(), _dt(0.001), _lcm(getLcmUrl(255)) {
        memset(&nn_data_lcm, 0, sizeof(NN_IO_t));
    }

    ~MP_Controller() override = default;  // delete function

    /*
     * Init the controller
     * load model parameter here
     */
    void initializeController() override;

    /*
     * Main loop control part
     */
    void runController() override;

    /*
     * Visual the contact information
     */
    void updateVisualization() override;

    /*
     * User Parameter process
     */
    ControlParameters *getUserControlParameters() override {
        return &userParameters;
    }

private:

    /*
     * According to the input(observation) to predict the action
     */
    void predict();

    /*
     * According to gamepad command to produce reference joint
     */
    void updateManualCommand();

    /*
     * set fake gamepad command from control parameters
     */
    void updateFakeCommand();

    void readGamepadCommand2();
    
    void readGamepadCommand1();

    void readRCCommand1();

    void readRCCommand2();

    // -------------------------------------------------------------
    // parameter defination
    MP_UserParameters userParameters;  // config parameter
    float _dt;  // time step of the algorithm
    bool is_ai_control = false;  // flag to determine whether switch to ai control
    bool leg_control_enable = false;
    float filter_para = 1.0f;    // filter parameter, according to the frequency parameter
    float _current_time = 0.0f;  // record current time
    float command_filtered[3] = {0.0f, 0.0f, 0.0f};
    float _phase[4] = {0.5f, 0.0f, 0.0f, 0.5f};
    float command_filter_para = 0.99f;  // smooth the command

    float gait_step_, side_step_, rot_step_ = 0.0f;
    float stiff = 0.0f;
    float damping = 0.0f;
    float StandUp_pr = 0.0f;

    float stand_height = 0.0f;
    float up_height = 0.0f;
    float max_up_height = 0.0f;
    float down_height = 0.0f;
    float freq = 0.0f;
    float stiff_max = 0.0f;
    float stiff_low = 10.0f;
    float damping_max = 0.0f;
    float VxMax = 0.0f;
    float VyMax = 0.0f;
    float OmegaMax = 0.0f;
    float lean_middle = 0.0f;
    float period = 0.0;
    float lam = 0.5;
    float height_variable = 0.0f;
    float BeltGear = 1.0f;

    // -------------------------------------------------------------
    // ai model related parameter
    LstmPolicy lstm_policy;
    // Eigen::MatrixXf w0, w1, w2;  // ai model parameter
    // Eigen::VectorXf b0, b1, b2;  // ai model parameter
    Eigen::VectorXf input, output, input_raw;  // input/output layer of the nn
    Eigen::VectorXf output_last;    // last time control value
    // Eigen::VectorXf input_last;
    Eigen::VectorXf hidden_layer1, hidden_layer2;  // hidden layer parameter
    Eigen::VectorXf actionMean_, actionStd_, obMean_, obStd_;  // normalized parameters
    Eigen::VectorXf joint_last_;  // record last time step joint state
    Eigen::Vector3f p0, pf, toe;  // for reference trajectory producer
    float joint_filter_freq;
    float joint_filter_alpha;

    NN_IO_t nn_data_lcm;  // save nn input and output data for debug
    command_phase_lcmt cp_lcm;  // save the user command and current phase
    lcm::LCM _lcm;        // lcm communication object
    std::mutex _mutex;    // lock

    // --------------------------------------------------------------
    Mat3<float> body_posture;  // store body posture
    Vec4<float> FakePhaseInStance;  // hoof contact ground phase
    Vec4<float> Contact;
    bool first_print = true;
    time_t start;
    time_t now;
    int iter = 0;
    int frame_skip = 10;  // jump the frame

    Eigen::MatrixXf NN_log;
    int NN_log_buffer = 100;
    int NN_log_index = 0;
    int obDim = 44;
    std::string NN_log_file;

    Eigen::VectorXf torque_max;
    float ratio = 0.8;

    Eigen::VectorXf joint_p_gains;
    Eigen::VectorXf joint_d_gains;

    Eigen::VectorXf stand_pos;
    int ai_control_trans_count = 0;

    float OrientationThreshold = 0.0;
    float LinVelThreshold = 0.0;
    float AngVelThreshold = 0.0;

    ManualPhase<float> mp = ManualPhase<float>();
    Eigen::Matrix<float, 8, 1> cpg_status;
    Eigen::Matrix<float, 4, 1> cpg_phase;

    Eigen::Matrix<float, 3, 1> smoothTargetVelocity;
    Eigen::Matrix<float, 3, 1> targetVelocity;
    int gait_idx = 1;
    int prev_gait_idx = 1;
    int hold_leg = -1;

    float second_level_kp = 0.0;

    int imu_filter_counter = 0;
    int imu_filter_frames = 0;
    Eigen::VectorXf imu_last;

    bool rt_prev = false;
    bool rb_prev = false;
    bool lb_prev = false;
    bool y_prev = false;
    bool x_prev = false;
    bool change_gait = false;

    bool start_prev = false;
    bool standingup_flag = false;
    bool sittingdown_flag = false;

    bool SF_prev = false;
    bool SH_prev = false;
    bool valid_gait = false;
};

#endif //BLACKPANTHER_MP_CONTROLLER_HPP
