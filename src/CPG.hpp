#ifndef CPG_HPP
#define CPG_HPP

#include <set>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace raisim
{

    template<typename T>
    class CPG
    {
    public:
        CPG()
        {
            // 0: WALK; 1: TROT; 2: PACE; 3: GALLOP; -1: STAND
            OFFSET[0] = {0.0, M_PI, M_PI * 0.5, M_PI * 1.5};
            OFFSET[1] = {0.0, M_PI, M_PI, 0.0};
            OFFSET[2] = {0.0, M_PI, 0.0, M_PI};
            OFFSET[3] = {0.0, 0.0, M_PI, M_PI};

            OFFSET_THREE = {0.0, 2.0 / 3.0 * M_PI, 4.0 / 3.0 * M_PI, 0.0};

            q_ << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0;

            reset_gait_four_leg(gait_);
        }

        inline void update_r_mat_four_leg()
        {
            r_mat_.setZero();

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    T theta = offset_[j] - offset_[i];
                    r_mat_(2 * j, 2 * i) = std::cos(theta);
                    r_mat_(2 * j + 1, 2 * i + 1) = std::cos(theta);
                    r_mat_(2 * j, 2 * i + 1) = -std::sin(theta);
                    r_mat_(2 * j + 1, 2 * i) = std::sin(theta);
                }
            }
        }

        inline void update_r_mat_four_leg_transition(int init_gait, int target_gait)
        {
            r_mat_.setZero();
            update_phase();

            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    T target_theta = OFFSET[target_gait][j] - OFFSET[target_gait][i];
                    T init_theta = OFFSET[init_gait][j] - OFFSET[init_gait][i];
                    int control_variable = get_control_variable(init_gait, target_gait);
                    
                    // T theta = offset_[j] - offset_[i];
                    T theta = target_theta + (target_theta - init_theta) / (OFFSET[target_gait][control_variable] - OFFSET[init_gait][control_variable]) * (phase_[control_variable] - OFFSET[target_gait][control_variable]) * 0.5;

                    r_mat_(2 * j, 2 * i) = std::cos(theta);
                    r_mat_(2 * j + 1, 2 * i + 1) = std::cos(theta);
                    r_mat_(2 * j, 2 * i + 1) = -std::sin(theta);
                    r_mat_(2 * j + 1, 2 * i) = std::sin(theta);

                    // if (isnan(theta))
                    // {
                    //     std::cout << "theta is nan!!!" << std::endl
                    //                 << "i: " << i << " j: " << j << std::endl
                    //                 << "target_theta: " << target_theta << " init_theta: " << init_theta << std::endl
                    //                 << "control variable: " << control_variable << std::endl
                    //                 << "delta control" << OFFSET[target_gait][control_variable] - OFFSET[init_gait][control_variable] << std::endl
                    //                 << "target gait: " << target_gait << " init_gait: " << init_gait << std::endl;
                    // }
                }
            }
        }

        inline void update_r_mat_three_leg()
        {
            r_mat_.setZero();
            int ii = 0;
            int jj = 0;

            for (int i = 0; i < 4; ++i)
            {
                if (i == hold_leg_)
                {
                    continue;
                }
                
                ii = i > hold_leg_ ? i - 1 : i;

                for (int j = 0; j < 4; ++j)
                {
                    if (j == hold_leg_)
                    {
                        continue;
                    }
    
                    jj = j > hold_leg_ ? j - 1 : j;

                    T theta = offset_[jj] - offset_[ii];
                    r_mat_(2 * j, 2 * i) = std::cos(theta);
                    r_mat_(2 * j + 1, 2 * i + 1) = std::cos(theta);
                    r_mat_(2 * j, 2 * i + 1) = -std::sin(theta);
                    r_mat_(2 * j + 1, 2 * i) = std::sin(theta);
                }
            }
        }

        inline int get_control_variable(int init_gait, int target_gait)
        {
            int p = 0;
            for (int i = 0; i < 4; ++i)
            {
                if (fabs(OFFSET[target_gait][i] - OFFSET[init_gait][i]) > 0.1)
                {
                    p = i;
                    break;
                }
            }
            return p;
        }

        inline void update_phase()
        {
            for (int i = 0; i < 4; ++i)
            {
                phase_[i] = std::atan2(q_[2 * i + 1], q_[2 * i]);
            }
            for (int i = 1; i < 4; ++i)
            {
                phase_[i] -= phase_[0];
                if (phase_[i] < 0)
                {
                    phase_[i] += M_PI * 2.0;
                }
            }
            phase_[0] = 0.0;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> get_raw_phase()
        {
            phase_raw_.setZero();
            for (int i = 0; i < 4; ++i)
            {
                phase_raw_[i] = std::atan2(q_[2 * i + 1], q_[2 * i]);
            }
            return phase_raw_;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> get_omega()
        {
            omega_.setZero();
            for (int i = 0; i < 4; ++i)
            {
                omega_[i] = -M_PI / t_ * (1.0 / beta_ / (1.0 + std::exp(-b_ * q_(2 * i + 1))) + 1.0 / (1.0 - beta_) / (1.0 + std::exp(b_ * q_(2 * i + 1))));
                // if (omega_[i] < -3.15*4)
                // {
                //     std::cout << "omega: " << omega_[i] << "; i: " << i << "; t_: " << t_ << "; beta_: " << beta_ << "; q: " << q_(2*i+1) << "; b: " << b_ << "gait: " << gait_ << std::endl;
                // }

                if (gait_ < 0)
                {
                    omega_[i] = 0.0;
                }
            }
            return omega_;
        }

        inline void reset()
        {
            q_ << 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0;
            hold_leg_ = -1;
            reset_gait_four_leg(-1);
        }

        inline void reset_gait_four_leg(int gait)
        {
            // use this function only for reset
            if (gait >= 0)
            {
                gait_ = gait;
                previous_gait_ = gait;
                target_gait_ = gait;

                beta_ = BETA[gait_];
                delta_ = DELTA[gait_];
                t_ = TIME[gait_];
                offset_ = OFFSET[gait_];
                
                phase_ = offset_;

                for (int i = 0; i < 4; ++i)
                {
                    q_[2 * i] = std::cos(phase_[i]);
                    q_[2 * i + 1] = std::sin(phase_[i]);
                }

                update_r_mat_four_leg();
            }
            else
            {
                gait_ = gait;
                previous_gait_ = gait;
                target_gait_ = gait;

                beta_ = BETA[gait_];
                delta_ = DELTA[gait_];
                t_ = TIME[gait_];
                offset_ = OFFSET[gait_];

                // q_ << -1.0, 0.0, -0.5, 0.866, 0.5, 0.866, 1.0, 0.0;
                // q_dot_.setZero();
                // phase_raw_ << -M_PI, -M_PI / 3.0 * 2.0, M_PI / 3.0, 0.0;
                // phase_ = {0.0, M_PI / 3.0, M_PI / 3.0 * 2.0, M_PI};
                q_ << 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0;
                q_dot_.setZero();
                phase_raw_ << M_PI / 2.0, M_PI / 2.0, M_PI / 2.0, M_PI / 2.0;
                phase_ = {0.0, 0.0, 0.0, 0.0};
            }
        }

        inline void reset_gait_three_leg(int gait)
        {
            gait_ = gait;
            previous_gait_ = gait;
            target_gait_ = gait;

            beta_ = BETA_THREE;
            delta_ = DELTA_THREE;
            t_ = TIME_THREE;
            offset_ = OFFSET_THREE;

            if (gait >= 0)
            {   
                phase_ = offset_;

                int ii = 0;

                for (int i = 0; i < 4; ++i)
                {
                    if (i == hold_leg_)
                    {
                        q_[2 * i] = 0.0;
                        q_[2 * i + 1] = -1.0;
                        continue;
                    }
                    ii = i > hold_leg_ ? i - 1 : i;
                    q_[2 * i] = std::cos(phase_[ii]);
                    q_[2 * i + 1] = std::sin(phase_[ii]);
                }

                update_r_mat_three_leg();
            }
            else
            {
                q_ << 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0;
                q_[2 * hold_leg_ + 1] = -1.0;
                q_dot_.setZero();
                phase_raw_ << M_PI / 2.0, M_PI / 2.0, M_PI / 2.0, M_PI / 2.0;
                phase_raw_[hold_leg_] = - M_PI / 2.0;
                phase_ = {0.0, 0.0, 0.0, 0.0};
                phase_[hold_leg_] = -0.5;
            }
        }

        inline void change_gait(int gait)
        {
            if (hold_leg_ < 0)
            {
                if (previous_gait_ < 0 || gait_ < 0 || gait < 0)
                {
                    reset_gait_four_leg(gait);
                }
                else
                {
                    previous_gait_ = target_gait_;
                    target_gait_ = gait;
                    gait_ = target_gait_;

                    beta_ = BETA[gait_];
                    delta_ = DELTA[gait_];
                    t_ = TIME[gait_];
                    offset_ = OFFSET[gait_];

                    update_r_mat_four_leg();
                }
            }
            else
            {
                gait = gait > 0 ? 0 : gait;
                if (previous_gait_ < 0 || gait_ < 0 || gait < 0)
                {
                    reset_gait_three_leg(gait);
                }
                else
                {
                    previous_gait_ = target_gait_;
                    target_gait_ = gait;
                    gait_ = target_gait_;

                    beta_ = BETA_THREE;
                    delta_ = DELTA_THREE;
                    t_ = TIME_THREE;
                    offset_ = OFFSET_THREE;

                    update_r_mat_three_leg();
                }
            }
        }

        inline void three_leg_mode(int hold_leg)
        {
            hold_leg_ = hold_leg;

            // if (hold_leg_ >= 0)
            // {
            //     update_r_mat_three_leg(hold_leg_);
            // }
            // else
            // {
            //     update_r_mat_four_leg();
            // }
        }

        inline void step_()
        {
            for (int i = 0; i < 4; ++i)
            {
                r_square_[i] = q_[2 * i] * q_[2 * i] + q_[2 * i + 1] * q_[2 * i + 1];
            }

            f_mat_.setZero();
            for (int i = 0; i < 4; ++i)
            {
                f_mat_(2 * i, 2 * i) = alpha_ * (mu_ - r_square_[i]);
                f_mat_(2 * i + 1, 2 * i + 1) = gamma_ * (mu_ - r_square_[i]);
                T omega = M_PI / t_ * (1.0 / beta_ / (1.0 + std::exp(-b_ * q_(2 * i + 1))) + 1.0 / (1.0 - beta_) / (1.0 + std::exp(b_ * q_(2 * i + 1))));
                f_mat_(2 * i, 2 * i + 1) = omega;
                f_mat_(2 * i + 1, 2 * i) = -omega;
            }

            q_dot_ = f_mat_ * q_ + r_mat_ * q_ * delta_;

            if (hold_leg_ >= 0)
            {
                // for (int i = 0; i < 4; ++i)
                // {
                //     f_mat_(2 * i, 2 * i) = alpha_ * (mu_ - r_square_[i]);
                //     f_mat_(2 * i + 1, 2 * i + 1) = gamma_ * (mu_ - r_square_[i]);
                //     T omega = M_PI / TIME_THREE * (1.0 / BETA_THREE / (1.0 + std::exp(-b_ * q_(2 * i + 1))) + 1.0 / (1.0 - BETA_THREE) / (1.0 + std::exp(b_ * q_(2 * i + 1))));
                //     f_mat_(2 * i, 2 * i + 1) = omega;
                //     f_mat_(2 * i + 1, 2 * i) = -omega;
                // }
                // q_dot_ = f_mat_ * q_ + r_mat_ * q_ * delta_;

                auto vec_dis = Eigen::Matrix<T, 2, 1>(0.0 - q_[2 * hold_leg_], -1.0 - q_[2 * hold_leg_ + 1]);
                T dis = vec_dis.norm();
                dis = dis > 0.4 ? 1.0 : dis * dis / 0.16;

                q_dot_(2 * hold_leg_) *= dis;
                q_dot_(2 * hold_leg_ + 1) *= dis;
                q_dot_(2 * hold_leg_) += vec_dis(0);
                q_dot_(2 * hold_leg_ + 1) += vec_dis(1);
            }

            q_ = q_ + q_dot_ * dt_;
        }

        inline void step()
        {
            if (hold_leg_ < 0)
            {
                if (gait_ != previous_gait_)
                {
                    update_r_mat_four_leg_transition(previous_gait_, target_gait_);

                    int control_variable = get_control_variable(previous_gait_, target_gait_);
                    // if (fabs(phase_[control_variable] - OFFSET[target_gait_][control_variable]) < 0.1 * phase_[control_variable])
                    if (fabs(phase_[control_variable] - OFFSET[target_gait_][control_variable]) < 0.1 * M_PI)
                    {
                        previous_gait_ = gait_;
                        update_r_mat_four_leg();
                    }
                }
            }

            if (gait_ < 0)
            {
                ;
            }
            else
            {
                step_();
            }
        }

        bool get_transition_status()
        {
            if (gait_ != previous_gait_)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> get_status()
        {
            return q_;
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> get_velocity()
        {
            return q_dot_;
        }

        T get_stance_time()
        {
            if (hold_leg_ < 0)
            {
                return t_ * beta_;
            }
            else
            {
                return TIME_THREE * BETA_THREE;
            }
        }

        int get_gait_index()
        {
            return gait_;
        }

        int get_hold_leg()
        {
            return hold_leg_;
        }

    // private:
        T mu_ = 1.0;
        T alpha_ = 50.0;
        T gamma_ = 50.0;
        T b_ = 50.0;
        T dt_ = 0.01;

        std::array<T, 4> BETA = {0.75, 0.5, 0.5, 0.4};
        std::array<T, 4> DELTA = {1.0, 1.0, 1.0, 1.0};
        std::array<T, 4> TIME = {0.6, 0.5, 0.5, 0.3};
        std::array<std::array<T, 4>, 4> OFFSET;

        T BETA_THREE = 2.0 / 3.0;
        T DELTA_THREE = 1.0;
        T TIME_THREE = 0.45;
        std::array<T, 4> OFFSET_THREE;

        int gait_ = -1;
        int previous_gait_ = -1;
        int target_gait_ = -1;

        T beta_, delta_, t_;
        std::array<T, 4> offset_;
        std::array<T, 4> r_square_;
        std::array<T, 4> phase_; // 0 to 2 * pi

        Eigen::Matrix<T, 8, 1> q_;
        Eigen::Matrix<T, 8, 1> q_dot_;
        Eigen::Matrix<T, 8, 8> r_mat_;
        Eigen::Matrix<T, 8, 8> f_mat_;
        Eigen::Matrix<T, 4, 1> phase_raw_; // -pi to +pi
        Eigen::Matrix<T, 4, 1> omega_;
        
        int hold_leg_ = -1;
    };

} // namespace raisim

#endif