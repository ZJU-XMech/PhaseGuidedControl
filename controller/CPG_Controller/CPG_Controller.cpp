#include "CPG_Controller.hpp"
#include "csv_to_eigen.hpp"
#include <Configuration.h>
#include "NN_log.hpp"
#include "InverseKinematics_v2.hpp"
#include "GaitGenerator.hpp"

#include <cmath>
#include <ctime>
#include <utility>

float CrawlAbad[4] = {-0.4, 0.4, -0.4, 0.4};
float CrawlHip[4] = {-1.0, -1.0, -1.0, -1.0};
float CrawlKnee[4] = {2.4, 2.4, 2.4, 2.4};

float StandAbad[4] = {0.0, -0.0, 0.0, -0.0};
float StandHip[4] = {-0.8, -0.8, -0.8, -0.8};
float StandKnee[4] = {1.6, 1.6, 1.6, 1.6};

// float contact_threshold[4] = {0.195f, 0.182f, 0.190f, 0.182f};
// float contact_threshold[4] = {12.5f*12.5f, 12.5f*12.5f, 12.5f*12.5f, 12.5f*12.5f};
float contact_threshold[4] = {5.0f*5.0f, 5.0f*5.0f, 5.0f*5.0f, 5.0f*5.0f};

void CPG_Controller::initializeController() {

    // --------------------------------------------------------------
    // load w parameter
    std::string header = "config/";
    header = "./../" + header;
    lstm_policy = LstmPolicy();
    lstm_policy.reset();

    NN_log.setZero(NN_log_buffer, obDim+12+1);
    time_t now_ = time(0);
    tm *ltm = localtime(&now_);
    std::stringstream logfile;
    std::string log_prefix = "../other/data/";
    if (!IS_SIM) {
        log_prefix = "./";
    }
    logfile << log_prefix
            << 1900 + ltm->tm_year << "_" 
            << 1 + ltm->tm_mon << "_" 
            << ltm->tm_mday << "_" 
            << ltm->tm_hour << "_" 
            << ltm->tm_min << "_" 
            << ltm->tm_sec << ".csv";
    NN_log_file = logfile.str();

    input.setZero(obDim);  // reshape and init the input buffer
    input_raw.setZero(obDim);  // reshape and init the input buffer
    output.setZero(12);  // reshape and init the output buffer
    output_last.setZero(12);  // reshape and init last time output buffer
    joint_last_.setZero(12);  // reshape and init last time step joint angle buffer

    imu_last.setZero(6);

    stand_pos.setZero(12);

    // init the normalization parameter
    actionMean_.resize(12);
    actionMean_ << 0.0f, -0.7f, 1.4f,  // FR
            0.0f, -0.7f, 1.4f,  // FL
            0.0f, -0.7f, 1.4f,  // HR
            0.0f, -0.7f, 1.4f;  // HL
    actionStd_.resize(12);
    actionStd_.setConstant(0.8f);

    obMean_.resize(obDim);
    obStd_.resize(obDim);
    obMean_ << 0.0f, 0.0f, 1.0f,                            // gravity axis
        Eigen::VectorXf::Constant(6, 0.0),        // body lin/ang vel
        actionMean_,
        Eigen::VectorXf::Constant(12, 0.0),
        Eigen::VectorXf::Constant(8, 0.0),        // phase
        0.0, 0.0, 0.0;
    
    obStd_ << Eigen::VectorXf::Constant(3, 1.0 / 0.3), // gravity axis
        Eigen::VectorXf::Constant(3, 1.0 / 1.0), // body linear velocities
        Eigen::VectorXf::Constant(3, 1.0 / 3.0), // body angular velocities
        Eigen::VectorXf::Constant(12, 1.0 / 0.8),  // joint angles
        1.0/12.0, 1.0/30.0, 1.0/30.0, 1.0/12.0, 1.0/30.0, 1.0/30.0, 1.0/12.0, 1.0/30.0, 1.0/30.0, 1.0/12.0, 1.0/30.0, 1.0/30.0,
        // 1.0/5.0, 1.0/30.0, 1.0/30.0, 1.0/5.0, 1.0/30.0, 1.0/30.0, 1.0/5.0, 1.0/30.0, 1.0/30.0, 1.0/5.0, 1.0/30.0, 1.0/30.0,
        Eigen::VectorXf::Constant(8, 1.0 / 1.0),   // phase
        1.0, 1.0 / 0.5, 1.0; 
    
    smoothTargetVelocity.setZero();
    targetVelocity.setZero();
    cpg_status.setZero();


    // -----------------------------------------------------------------
    // load parameters from config file
    stand_height = (float) userParameters.stand_height;
    up_height = (float) userParameters.up_height;
    max_up_height = up_height;
    down_height = (float) userParameters.down_height;
    freq = (float) userParameters.Freq;
    stiff_max = (float) userParameters.Stiffness;
    damping_max = (float) userParameters.DDamping;
    VxMax = (float) userParameters.Vx;
    VyMax = (float) userParameters.Vy;
    OmegaMax = (float) userParameters.Omega;
    lean_middle = (float) userParameters.Lean_middle;
    period = (float) userParameters.Period;
    lam = (float) userParameters.lam;
    height_variable = (float) userParameters.HeightVariable;

    torque_max.setZero(12);
    torque_max << 18.0, 18.0, 27.0,
                18.0, 18.0, 27.0,
                18.0, 18.0, 27.0,
                18.0, 18.0, 27.0;
    
    torque_max = torque_max * ratio;

    joint_p_gains.setConstant(12, stiff_max);
    joint_d_gains.setConstant(12, damping_max);

    joint_filter_freq = 20.0;
    joint_filter_alpha = 2.0 * M_PI * 0.01 * joint_filter_freq / (2.0 * M_PI * 0.01 * joint_filter_freq + 1.0);

    filter_para = 1.0f - freq * _dt * (float) frame_skip;  // real time step is dt * frame_skip>>
    _lcm.publish("NN_IO", &nn_data_lcm);
    BeltGear = (IS_SIM) ? 1.0f : 2.4025f;
     // BeltGear = (IS_SIM) ? 1.0f : 1.55f;
     // BeltGear = (IS_SIM) ? 1.0f : 2.0f;
    std::cout<<"gear"<<IS_SIM<<std::endl;
    std::cout << "remote status: " << REMOTE_CONTROL << std::endl;
}

// void CPG_Controller::predict() {
//     // according the observation produce the action
//     hidden_layer1 = tanh(w0.transpose() * input + b0);
//     hidden_layer2 = tanh(w1.transpose() * hidden_layer1 + b1);
//     output = w2.transpose() * hidden_layer2 + b2;
//     output = output.cwiseMax(-1.0);
//     output = output.cwiseMin(1.0);
// }

void CPG_Controller::readGamepadCommand1()
{
    if (_driverCommand->start)
    {
        if (!start_prev)
        {
            // push
            is_ai_control = !is_ai_control;
        }
        start_prev = true;
    }
    else
    {
        start_prev = false;
    }

    is_ai_control = (StandUp_pr < 0.9) ? false : is_ai_control;
    is_ai_control = (_driverCommand->back) ? false : is_ai_control;
    leg_control_enable = (_driverCommand->a) ? true : leg_control_enable;
    leg_control_enable = (_driverCommand->b) ? false : leg_control_enable;

    standingup_flag = _driverCommand->leftTriggerButton;
}

void CPG_Controller::readRCCommand1()
{
    switch (_remoteController->SA)
    {
        case 0:
            leg_control_enable = false;
            is_ai_control = false;
            break;
        case 1:
            leg_control_enable = true;
            is_ai_control = false;
            break;
        case 2:
            leg_control_enable = true;
            is_ai_control = StandUp_pr > 0.9;
            break;
        default:
            leg_control_enable = false;
            is_ai_control = false;
            break;
    }
    switch (_remoteController->SB)
    {
        case 0:
            standingup_flag = false;
            sittingdown_flag = true;
            break;
        case 1:
            break;
        case 2:
            standingup_flag = true;
            sittingdown_flag = false;
            break;
        default:
            break;
    }

    // std::cout << "SA: " << _remoteController->SA << std::endl;
    // std::cout << "SB: " << _remoteController->SB << std::endl;
}

void CPG_Controller::readGamepadCommand2()
{
    change_gait = false;

    // update command
    targetVelocity[0] = std::fabs(_driverCommand->leftStickAnalog[1]) * _driverCommand->leftStickAnalog[1];
    targetVelocity[1] = - std::fabs(_driverCommand->leftStickAnalog[0]) * _driverCommand->leftStickAnalog[0];
    targetVelocity[2] = - std::fabs(_driverCommand->rightStickAnalog[0]) * _driverCommand->rightStickAnalog[0];

    // targetVelocity[0] = _driverCommand->leftStickAnalog[1];
    // targetVelocity[1] = - _driverCommand->leftStickAnalog[0];
    // targetVelocity[2] = - _driverCommand->rightStickAnalog[0];

    // std::cout << "left: " << _driverCommand->leftStickAnalog[0]
    //             << " " << _driverCommand->leftStickAnalog[1] << std::endl;

    targetVelocity[0] = fabs(targetVelocity[0]) < 0.1 ? 0.0 : targetVelocity[0];
    targetVelocity[1] = fabs(targetVelocity[1]) < 0.1 ? 0.0 : targetVelocity[1] * 0.5;
    targetVelocity[2] = fabs(targetVelocity[2]) < 0.3 ? 0.0 : targetVelocity[2];

    // set leg
    if (_driverCommand->y && !y_prev)
    {
        hold_leg += 1;
        hold_leg = hold_leg > 3 ? 0 : hold_leg;

        change_gait = true;

        cpg.three_leg_mode(hold_leg);

        std::cout << "Enter three leg mode! Swing leg: " << hold_leg << std::endl;
    }
    y_prev = _driverCommand->y;

    if (_driverCommand->x && !x_prev)
    {
        hold_leg = -1;
        change_gait = true;
        
        cpg.three_leg_mode(hold_leg);

        std::cout << "Leave three leg mode!" << std::endl;
    }
    x_prev = _driverCommand->x;

    // set gait
    if (_driverCommand->rightTriggerButton > 0.5 && !rt_prev)
    {
        gait_idx = 3;
        change_gait = true;
        // std::cout << "RT trigger!" << std::endl;
    }
    rt_prev = _driverCommand->rightTriggerButton > 0.5;

    if (_driverCommand->rightBumper && !rb_prev)
    {
        gait_idx = 0;
        change_gait = true;
        // std::cout << "RB trigger!" << std::endl;
    }
    rb_prev = _driverCommand->rightBumper;

    if (_driverCommand->leftBumper && !lb_prev)
    {
        gait_idx = 2;
        change_gait = true;
        // std::cout << "LB trigger!" << std::endl;
    }
    lb_prev = _driverCommand->leftBumper;

    if (hold_leg >= 0 && gait_idx >= 0)
    {
        gait_idx = 0;
    }

    if (change_gait)
    {
        if (hold_leg >= 0)
        {
            // three leg mode
            // if (gait_idx != prev_gait_idx)
            // {
            //     cpg.change_gait(gait_idx);
            // }
            cpg.change_gait(gait_idx);
            prev_gait_idx = gait_idx;
        }
        else
        {
            // four leg mode
            if (gait_idx != prev_gait_idx)
            {
                std::cout << "changing gait! from " << prev_gait_idx << " to " << gait_idx << std::endl;
                prev_gait_idx = gait_idx;

                cpg.change_gait(gait_idx);
            }
            else
            {
                gait_idx = 1;
                prev_gait_idx = 1;

                cpg.change_gait(gait_idx);
            }
        }
    }
    else
    {
        if (cpg.get_gait_index() != gait_idx)
        {
            cpg.change_gait(gait_idx);
            prev_gait_idx = gait_idx;
        }
    }
    

    if (gait_idx == 3)
    {
        targetVelocity[2] *= 0.5;
        gait_idx = 3;

        if (targetVelocity[0] > 0.9)
        {
            targetVelocity[0] = 0.9 + (targetVelocity[0] - 0.9) * 6;
        }
        if (targetVelocity[0] < - 0.9)
        {
            targetVelocity[0] = - 0.9 + (targetVelocity[0] + 0.9) * 6;
        }
    }

    if (gait_idx == 0)
    {
        targetVelocity[1] *= 0.5;
        targetVelocity[2] *= 0.4;

        if (std::fabs(targetVelocity[0]) > 0.4)
        {
            targetVelocity[0] = std::fabs(targetVelocity[0]) / targetVelocity[0] * 0.4;
        }
    }

    if (targetVelocity.lpNorm<2>() < 0.05)
    {
        // prev_gait_idx = -1;
        cpg.change_gait(-1);
    }

    // if (_driverCommand->rightTriggerButton > 0.5)
    // {
    //     targetVelocity[2] *= 0.5;
    //     gait_idx = 3;

    //     if (targetVelocity[0] > 0.9)
    //     {
    //         targetVelocity[0] = 0.9 + (targetVelocity[0] - 0.9) * 6;
    //     }
    //     if (targetVelocity[0] < - 0.9)
    //     {
    //         targetVelocity[0] = - 0.9 + (targetVelocity[0] + 0.9) * 6;
    //     }
    // }

    // if (_driverCommand->rightBumper) 
    // {
    //     gait_idx = 0;
    //     // targetVelocity[0] *= 0.3;
    //     targetVelocity[1] *= 0.5;
    //     targetVelocity[2] *= 0.4;

    //     if (std::fabs(targetVelocity[0]) > 0.4)
    //     {
    //         targetVelocity[0] = std::fabs(targetVelocity[0]) / targetVelocity[0] * 0.4;
    //     }
    // }

    // if (_driverCommand->leftBumper) 
    // {
    //     gait_idx = 2;
    // }

    // if (targetVelocity.lpNorm<2>() < 0.05)
    // {
    //     gait_idx = -1;
    // }

    smoothTargetVelocity = smoothTargetVelocity * 0.95 + targetVelocity * 0.05;

    // other options
    // second_level_kp = std::fabs(_dr*iverCommand->rightStickAnalog[1]) > 0.3 ? _driverCommand->rightStickAnalog[1]*0.12 : 0.0;

    // set gait
    // cpg.change_gait(gait_idx);
}

void CPG_Controller::readRCCommand2()
{
    change_gait = false;
    
    targetVelocity[0] = std::fabs(_remoteController->LeftStickY) * _remoteController->LeftStickY;
    targetVelocity[1] = - std::fabs(_remoteController->LeftStickX) * _remoteController->LeftStickX;
    targetVelocity[2] = - std::fabs(_remoteController->RightStickX) * _remoteController->RightStickX;

    targetVelocity[0] = fabs(targetVelocity[0]) < 0.1 ? 0.0 : targetVelocity[0];
    targetVelocity[1] = fabs(targetVelocity[1]) < 0.1 ? 0.0 : targetVelocity[1] * 0.5;
    targetVelocity[2] = fabs(targetVelocity[2]) < 0.3 ? 0.0 : targetVelocity[2];    

    // set leg
    if (_remoteController->SF == 2)
    {
        if (!SF_prev)
        {
            // change to three leg
            hold_leg = hold_leg < 0 ? 1 : hold_leg;

            change_gait = true;

            cpg.three_leg_mode(hold_leg);

            std::cout << "Enter three leg mode! Swing leg: " << hold_leg << std::endl;
        }
    }
    else
    {
        if (SF_prev)
        {
            hold_leg = -1;
            change_gait = true;
            
            cpg.three_leg_mode(hold_leg);

            std::cout << "Leave three leg mode!" << std::endl;
        }
    }
    SF_prev = _remoteController->SF == 2;

    if (_remoteController->SH == 2)
    {
        if (!SH_prev)
        {
            // change leg
            hold_leg += 1;
            hold_leg = hold_leg > 3 ? 0 : hold_leg;

            change_gait = true;

            cpg.three_leg_mode(hold_leg);

            std::cout << "Enter three leg mode! Swing leg: " << hold_leg << std::endl;
        }
    }
    SH_prev = _remoteController->SH == 2;

    // set gait
    valid_gait = false;
    if (_remoteController->SD == 0 && _remoteController->SG == 1)
    {
        gait_idx = 1;
        // change_gait = true;
        valid_gait = true;
    }
    if (_remoteController->SD == 2 && _remoteController->SG == 1)
    {
        gait_idx = 0;
        // change_gait = true;
        valid_gait = true;
    }
    if (_remoteController->SD == 1 && _remoteController->SG == 0)
    {
        gait_idx = 2;
        // change_gait = true;
        valid_gait = true;
    }
    if (_remoteController->SD == 1 && _remoteController->SG == 2)
    {
        gait_idx = 3;
        // change_gait = true;
        valid_gait = true;
    }
    if (!valid_gait)
    {
        // gait_idx = cp;
    }

    if (hold_leg >= 0 && gait_idx >= 0)
    {
        gait_idx = 0;
    }

    // change gait
    if (hold_leg < 0)
    {
        // four leg
        if (gait_idx != prev_gait_idx)
        {
            std::cout << "changing gait! from " << prev_gait_idx << " to " << gait_idx << std::endl;
            prev_gait_idx = gait_idx;

            cpg.change_gait(gait_idx);
        }
        else
        {
            if (cpg.get_gait_index() != gait_idx)
            {
                cpg.change_gait(gait_idx);
                prev_gait_idx = gait_idx;
            }
        }
    }
    else
    {
        // three leg
        if (change_gait)
        {
            cpg.change_gait(gait_idx);
            prev_gait_idx = gait_idx;
        }
        else
        {
            if (cpg.get_gait_index() != gait_idx)
            {
                cpg.change_gait(gait_idx);
                prev_gait_idx = gait_idx;
            }
        }
    }
    

    if (gait_idx == 3)
    {
        targetVelocity[2] *= 0.5;
        gait_idx = 3;

        if (targetVelocity[0] > 0.9)
        {
            targetVelocity[0] = 0.9 + (targetVelocity[0] - 0.9) * 6;
        }
        if (targetVelocity[0] < - 0.9)
        {
            targetVelocity[0] = - 0.9 + (targetVelocity[0] + 0.9) * 6;
        }
    }

    if (gait_idx == 0)
    {
        targetVelocity[1] *= 0.5;
        targetVelocity[2] *= 0.4;

        if (std::fabs(targetVelocity[0]) > 0.4)
        {
            targetVelocity[0] = std::fabs(targetVelocity[0]) / targetVelocity[0] * 0.4;
        }
    }

    if (targetVelocity.lpNorm<2>() < 0.05)
    {
        cpg.change_gait(-1);
    }

    smoothTargetVelocity = smoothTargetVelocity * 0.95 + targetVelocity * 0.05;
    // EMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

    // if (change_gait)
    // {
    //     if (hold_leg >= 0)
    //     {
    //         // three leg mode
    //         cpg.change_gait(gait_idx);
    //         prev_gait_idx = gait_idx;
    //     }
    //     else
    //     {
    //         // four leg mode
    //         if (gait_idx != prev_gait_idx)
    //         {
    //             std::cout << "changing gait! from " << prev_gait_idx << " to " << gait_idx << std::endl;
    //             prev_gait_idx = gait_idx;

    //             cpg.change_gait(gait_idx);
    //         }
    //         else
    //         {
    //             gait_idx = 1;
    //             prev_gait_idx = 1;

    //             cpg.change_gait(gait_idx);
    //         }
    //     }
    // }
    // else
    // {
    //     if (cpg.get_gait_index() != gait_idx)
    //     {
    //         cpg.change_gait(gait_idx);
    //         prev_gait_idx = gait_idx;
    //     }
    // }
}

void CPG_Controller::runController() {
    // update user parameter
    OrientationThreshold = static_cast<float>(userParameters.OrientationThreshold);
    LinVelThreshold = static_cast<float>(userParameters.LinVelThreshold);
    AngVelThreshold = static_cast<float>(userParameters.AngVelThreshold);
    
    // ----------------------------------------------------------------
    // interface, control the robot mode

    // if (_driverCommand->start)
    // {
    //     if (!start_prev)
    //     {
    //         // push
    //         is_ai_control = !is_ai_control;
    //     }
    //     start_prev = true;
    // }
    // else
    // {
    //     start_prev = false;
    // }
    // is_ai_control = (_driverCommand->start) ? true : is_ai_control;
    // is_ai_control = (StandUp_pr < 0.9) ? false : is_ai_control;
    // is_ai_control = (_driverCommand->back) ? false : is_ai_control;
    // leg_control_enable = (_driverCommand->a) ? true : leg_control_enable;
    // leg_control_enable = (_driverCommand->b) ? false : leg_control_enable;
    if (REMOTE_CONTROL)
    {
        readRCCommand1();
    }
    else
    {
        readGamepadCommand1();
    }
    // std::cout<<leg_control_enable<<std::endl;
    _legController->setEnabled(leg_control_enable);

    if (not is_ai_control) {
        _current_time = 0.0f;
        // iter = 0;
        if (standingup_flag && leg_control_enable) {
            // std::cout<<StandUp_pr<<std::endl;
            StandUp_pr = StandUp_pr * Height_Filter_Ratio + 1.0f * (1.0f - Height_Filter_Ratio);
            stiff = stiff * Height_Filter_Ratio + stiff_max * (1.0f - Height_Filter_Ratio);
            damping = damping * Height_Filter_Ratio + damping_max * (1.0f - Height_Filter_Ratio);
        }
        if (sittingdown_flag && leg_control_enable) {
            // std::cout<<StandUp_pr<<std::endl;
            StandUp_pr = StandUp_pr * Height_Filter_Ratio + 0.0f * (1.0f - Height_Filter_Ratio);
            stiff = stiff * Height_Filter_Ratio + 0.0f * (1.0f - Height_Filter_Ratio);
            damping = damping * Height_Filter_Ratio + 0.0f * (1.0f - Height_Filter_Ratio);
        }

        for (int leg = 0; leg < 4; ++leg) {
            output[3 * leg + 0] = CrawlAbad[leg] * (1.0f - StandUp_pr) + StandAbad[leg] * StandUp_pr;
            output[3 * leg + 1] = CrawlHip[leg] * (1.0f - StandUp_pr) + StandHip[leg] * StandUp_pr;
            output[3 * leg + 2] = CrawlKnee[leg] * (1.0f - StandUp_pr) + StandKnee[leg] * StandUp_pr;
        }
        stand_pos = output;

        Eigen::VectorXf jpos;
        jpos.setZero(12);
        jpos << _legController->datas[0].q,
                _legController->datas[1].q,
                _legController->datas[2].q,
                _legController->datas[3].q;

        if (ai_control_trans_count > 0) {
            ai_control_trans_count--;
            output = output * (1.0 - ai_control_trans_count / 1000.0) + jpos * (ai_control_trans_count / 1000.0);
        }

        _mutex.lock();
        for (int leg = 0; leg < 4; leg++) {

            _legController->commands[leg].kpJoint << stiff, 0.0f, 0.0f, 0.0f, stiff, 0.0f, 0.0f, 0.0f, stiff;
            _legController->commands[leg].kdJoint << damping, 0.0f, 0.0f, 0.0f, damping, 0.0f, 0.0f, 0.0f, damping;
            _legController->commands[leg].qDes << output.segment<3>(leg * 3);
            
            output_last.tail<12>().segment(3 * leg, 3) << _legController->commands[leg].qDes[0],
                    _legController->commands[leg].qDes[1],
                    _legController->commands[leg].qDes[2];

            // contact_threshold[leg] = (float) (*contact_force_squared_norm)(leg) * ThresholdRatio;
        }
        _mutex.unlock();

        // update lcm data
        // body orientation
        input.setZero();
        body_posture = rpyToRotMat(_stateEstimate->rpy);
        input.head<3>() = body_posture.col(2);
        // body lin/ang vel
        input.segment<3>(3) = _stateEstimate->vBody;
        input.segment<3>(3) << 0.0, 0.0, 0.0;
        input.segment<3>(6) = _stateEstimate->omegaBody;
        // joint pos
        input.segment<12>(9) << _legController->datas[0].q,
                                _legController->datas[1].q,
                                _legController->datas[2].q,
                                _legController->datas[3].q;
        joint_last_ = input.segment<12>(21);
        memcpy(nn_data_lcm.input, input.data(), obDim * sizeof(float));
        memcpy(nn_data_lcm.output, output.data(), 12 * sizeof(float));
        _lcm.publish("NN_IO", &nn_data_lcm);

        // reset cpg
        cpg.reset();
    } else {
        if (StandUp_pr > 0.9) {
            if (iter % frame_skip == 0) {
                // clock_t nn_start, nn_finish;
                // nn_start = clock();

                // read gamepad command and update cpg
                if (REMOTE_CONTROL)
                {
                    readRCCommand2();
                }
                else
                {
                    readGamepadCommand2();
                }
                cpg_status = cpg.get_status();
                cpg.step();

                // log cpg
                // std::cout << "current gait: " << cpg.get_current_gait() << "; previous gait: " << cpg.get_previous_gait() << std::endl;

                // prepare NN control observation
                // body orientation
                body_posture = rpyToRotMat(_stateEstimate->rpy);
                input.head<3>() = body_posture.col(2);
                // body lin/ang vel
                input.segment<3>(3) = _stateEstimate->vBody;
                input.segment<3>(3) << 0.0, 0.0, 0.0;
                input.segment<3>(6) = _stateEstimate->omegaBody;
                // joint pos
                input.segment<12>(9) << _legController->datas[0].q,
                                        _legController->datas[1].q,
                                        _legController->datas[2].q,
                                        _legController->datas[3].q;
                // joint velocity
                input.segment<12>(21) << _legController->datas[0].qd,
                                        _legController->datas[1].qd,
                                        _legController->datas[2].qd,
                                        _legController->datas[3].qd;
                input.segment<12>(21) = input.segment<12>(21) * joint_filter_alpha + joint_last_ * (1.0 - joint_filter_alpha);
                joint_last_ = input.segment<12>(21);
                // cpg status
                input.segment<8>(33) = cpg_status;
                // input.segment<6>(33).setZero();

                // second level feedback
                // smoothTargetVelocity[2] += second_level_kp * (smoothTargetVelocity[2] - _stateEstimate->omegaBody[2]);

                // desired velocity
                input.tail<3>() = smoothTargetVelocity;

                // filter state estimator
                input[0] = (std::fabs(input[0]) < OrientationThreshold) ? 0.0 : input[0];
                input[1] = (std::fabs(input[1]) < OrientationThreshold) ? 0.0 : input[1];
                input[2] = std::sqrt(1.0 - input[0] * input[0] - input[1] * input[1]);
                input[3] = (std::fabs(input[3]) < LinVelThreshold) ? 0.0 : input[3];
                input[4] = (std::fabs(input[4]) < LinVelThreshold) ? 0.0 : input[4];
                input[5] = (std::fabs(input[5]) < LinVelThreshold) ? 0.0 : input[5];
                input[6] = (std::fabs(input[6]) < AngVelThreshold) ? 0.0 : input[6];
                input[7] = (std::fabs(input[7]) < AngVelThreshold) ? 0.0 : input[7];
                input[8] = (std::fabs(input[8]) < AngVelThreshold) ? 0.0 : input[8];

                if (imu_filter_counter == 0)
                {
                    imu_last.head<3>() = input.head<3>();
                    imu_last.tail<3>() = input.segment<3>(6);
                }
                else
                {
                    input.head<3>() = imu_last.head<3>();
                    input.segment<3>(6) = imu_last.tail<3>();
                }
                if (imu_filter_counter < imu_filter_frames)
                {
                    imu_filter_counter ++;
                }
                else
                {
                    imu_filter_counter = 0;
                }
                

                memcpy(nn_data_lcm.input, input.data(), obDim * sizeof(float));
                input_raw = input;
                input_raw.segment<3>(3) = _stateEstimate->vBody;
                input_raw.segment<12>(21) << _legController->datas[0].qd,
                                            _legController->datas[1].qd,
                                            _legController->datas[2].qd,
                                            _legController->datas[3].qd;
                NN_log.row(NN_log_index).head(obDim) = input_raw;

                // std::cout << "input: " << input_raw.transpose() << std::endl;

                // std::cout << "input done" << std::endl;
                // std::cout << input.transpose();
                // normalize, predict, denormalize
                input = (input - obMean_).cwiseProduct(obStd_);
                // predict();
                lstm_policy.input = input;
                lstm_policy.predict();
                output = lstm_policy.output;
                output = output.cwiseProduct(actionStd_) + actionMean_;

                if (cpg.get_hold_leg() >= 0)
                {
                    float effectfactor = (cpg_status.segment<2>(2*cpg.get_hold_leg()) - Eigen::Matrix<float, 2, 1>(0.0, -1.0)).norm();
                    effectfactor = effectfactor > 0.2 ? 1.0 : effectfactor / 0.2;
                
                    Eigen::Matrix<float, 3, 1> hold_state(0.0, -1.0, 2.0);
                    output.segment<3>(3*cpg.get_hold_leg()) = output.segment<3>(3*cpg.get_hold_leg()) * effectfactor + hold_state * (1.0 - effectfactor);
                }

                // std::cout << "predict done" << std::endl;
                // std::cout << output.transpose();

                // std::cout << "NN output: " << output.transpose() << std::endl;

                // quick copy the data from eigen vector to lcm data struct
                memcpy(nn_data_lcm.output, output.data(), 12 * sizeof(float));
                _lcm.publish("NN_IO", &nn_data_lcm);
                // output = output_last * filter_para + output * (1.0f - filter_para);
                // output_last = output;
                // output = output / 3.0 + output_last.tail<12>() / 3.0 + output_last.head<12>() / 3.0;
                // output = output_last * 0.7 + output * 0.3;
                // for (int abad = 0; abad < 4; ++abad)
                // {
                //     output[abad*3] = output_last[abad*3] * 0.7 + output[abad*3] * 0.3;
                // }
                output_last = output;

                NN_log.row(NN_log_index).segment<12>(obDim) = output;

                NN_log.row(NN_log_index).tail<1>() << _stateEstimate->position[2];

                // clip output
                Eigen::VectorXf jpos, jvel;
                jpos.setZero(12);
                jvel.setZero(12);
                jpos << _legController->datas[0].q,
                        _legController->datas[1].q,
                        _legController->datas[2].q,
                        _legController->datas[3].q;
                jvel << _legController->datas[0].qd,
                        _legController->datas[1].qd,
                        _legController->datas[2].qd,
                        _legController->datas[3].qd;
                
                auto output_max = jpos + (torque_max + jvel.cwiseProduct(joint_d_gains)).cwiseQuotient(joint_p_gains);
                auto output_min = jpos + (- torque_max + jvel.cwiseProduct(joint_d_gains)).cwiseQuotient(joint_p_gains);
                output = output.cwiseMax(output_min);
                output = output.cwiseMin(output_max);

                if (ai_control_trans_count < 1000) {
                    output = output * (ai_control_trans_count / 1000.0) + stand_pos * (1.0 - ai_control_trans_count / 1000.0);
                    ai_control_trans_count += frame_skip;
                    ai_control_trans_count = std::min(1000, ai_control_trans_count);
                }

                // record NN log
                // ++NN_log_index;
                // if (NN_log_index > NN_log_buffer - 1) {
                //     NN_log_index = 0;
                //     auto log_status = WriteLog(NN_log_file, NN_log);
                //     if (log_status == 0) {
                //         std::cout << "[SUCCESS] Save log at " << NN_log_file << std::endl;
                //     } else {
                //         std::cout << "[FAIL] Attempt to save log at " << NN_log_file << std::endl;
                //     }
                //     NN_log.setZero(NN_log_buffer, obDim+12+1);
                // }

                // nn_finish = clock();
                // std::cout << "NN time: " << (double)(nn_finish-nn_start)/CLOCKS_PER_SEC << "s" << std::endl;
            }
            // iter++;
            // std::cout<<"output"<<output.transpose()<<std::endl;
            _mutex.lock();
            for (int leg = 0; leg < 4; ++leg) {
                _legController->commands[leg].kpJoint << stiff_max, 0.0f, 0.0f, 0.0f, stiff_max, 0.0f, 0.0f, 0.0f, stiff_max/BeltGear;
                _legController->commands[leg].kdJoint << damping_max, 0.0f, 0.0f, 0.0f, damping_max, 0.0f, 0.0f, 0.0f, damping_max/BeltGear;
                _legController->commands[leg].qDes << output.segment<3>(leg * 3);
            }


            // for (int leg = 0; leg < 4; leg++) {
            //     // if (Contact[leg] > 0.0001f) {
            //     if(RealPhaseInStance[leg]<0.8f and RealPhaseInStance[leg]>0.2f){
            //         // if (false) {
            //         _legController->commands[leg].kpJoint << stiff, 0.0f, 0.0f, 0.0f, stiff, 0.0f, 0.0f, 0.0f, stiff/BeltGear;
            //         // change PD control stiffness to stabilize the robot's swing leg movement
            //     } else {
            //         _legController->commands[leg].kpJoint
            //                 << stiff_low, 0.0f, 0.0f, 0.0f, stiff_low, 0.0f, 0.0f, 0.0f, stiff_low/BeltGear;
            //         // _legController->commands[leg].kpJoint << stiff, 0.0f, 0.0f, 0.0f, stiff, 0.0f, 0.0f, 0.0f, stiff;
            //         // change PD control stiffness to stabilize the robot's swing leg movement
            //     }
            //     _legController->commands[leg].kdJoint << damping, 0.0f, 0.0f, 0.0f, damping, 0.0f, 0.0f, 0.0f, damping/BeltGear;
            //     _legController->commands[leg].qDes << output.segment(leg * 3, 3);
            // }

            // set fake contact information
            for (int i = 0; i < 4; ++i)
            {
                if (cpg_status[2*i+1] >= 0)
                {
                    // stance
                    FakePhaseInStance[i] = 0.5;
                }
                else
                {
                    // swing
                    FakePhaseInStance[i] = 0.0;
                }

                // if (gait_idx < 0)
                // {
                //     FakePhaseInStance[i] = 0.5;
                // }
            }
            _stateEstimator->setContactPhase(FakePhaseInStance);
            _mutex.unlock();
            _lcm.publish("cmd_and_phase", &cp_lcm);   
        }
    }

    if (iter % frame_skip == 0)
    {
        // record NN log
        ++NN_log_index;
        if (NN_log_index > NN_log_buffer - 1) {
            NN_log_index = 0;
            auto log_status = WriteLog(NN_log_file, NN_log);
            if (log_status == 0) {
                std::cout << "[SUCCESS] Save log at " << NN_log_file << std::endl;
            } else {
                std::cout << "[FAIL] Attempt to save log at " << NN_log_file << std::endl;
            }
            NN_log.setZero(NN_log_buffer, obDim+12+1);
        }
    }
    iter++;
}

void CPG_Controller::updateVisualization() {
    // since now, do nothing
    ;
}
