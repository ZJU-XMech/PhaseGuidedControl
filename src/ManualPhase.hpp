#ifndef MANUALPHASE_HPP
#define MANUALPHASE_HPP

#include <set>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace raisim
{
    constexpr int n_gait = 9;

    template<typename T>
    class ManualPhase
    {
    public:
        ManualPhase()
        {
            for (int i = 0; i < n_gait; ++i)
            {
                TIME[i] = UNIT[i] * unit_time_;
                STANCE_TIME[i] = DUTY_UNIT[i] * unit_time_ / (UNIT[i] / 2.0);
            }

            reset();
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

        Eigen::Matrix<T, 4, 1> get_omega()
        {
            return omega_;
        }

        int get_hold_leg()
        {
            // not implemented in manual phase
            return -1;
        }

        bool get_transition_status()
        {
            // not implemented in manual phase
            return false;
        }

        inline void three_leg_mode(int leg)
        {
            // not implemented in manual phase
            ;
        }

        inline void change_gait(int gait_idx)
        {
            target_gait_idx_ = gait_idx;
        }

        inline void change_gait_()
        {
            if (target_gait_idx_ != gait_idx_ && std::fabs(phase_scalar_) <= dt_ / 2.0)
            {
                gait_idx_ = target_gait_idx_;
            }
        }

        inline int get_gait_index()
        {
            return gait_idx_;            
        }

        inline T get_stance_time()
        {
            return STANCE_TIME[gait_idx_];
        }

        inline void reset()
        {
            unit_ = UNIT[0];
            time_ = TIME[0];
            duty_unity_ = DUTY_UNIT[0];
            stance_time_ = STANCE_TIME[0];
            
            q_ << 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0;
            q_dot_.setZero();

            phase_raw_ << - M_PI / 2.0, - M_PI / 2.0, - M_PI / 2.0, - M_PI / 2.0;
            phase_ = {0.0, 0.0, 0.0, 0.0};
            omega_ << - M_PI / unit_time_, - M_PI / unit_time_, - M_PI / unit_time_, - M_PI / unit_time_;

            phase_scalar_ = 0.0;
        }

        inline void step()
        {
            if (phase_scalar_ > TIME[gait_idx_])
            {
                phase_scalar_ = 0.0;
            }
            change_gait_();

            get_q_and_q_dot();

            phase_scalar_ += dt_;
        }

        inline void get_q_and_q_dot()
        {
            if (gait_idx_ == 0)
            {
                T theta = - phase_scalar_ / unit_time_ * M_PI + M_PI;
                for (int i = 0; i < 4; ++i)
                {
                    q_[2 * i] = std::cos(theta);
                    q_[2 * i + 1] = std::sin(theta);
                }
                omega_.setConstant(- M_PI / unit_time_);
            }

            if (gait_idx_ == 1)
            {
                T theta_fast = - phase_scalar_ / unit_time_ * M_PI + M_PI;
                T theta_slow = - phase_scalar_ / unit_time_ * M_PI / 3;

                if (phase_scalar_ / unit_time_ < 3)
                {
                    // 0, 3
                    q_[0] = std::cos(theta_slow);
                    q_[1] = std::sin(theta_slow);
                    q_[6] = std::cos(theta_slow);
                    q_[7] = std::sin(theta_slow);

                    omega_[0] = - M_PI / unit_time_ / 3;
                    omega_[3] = - M_PI / unit_time_ / 3;

                    // 1, 2
                    q_[2] = std::cos(theta_fast);
                    q_[3] = std::sin(theta_fast);
                    q_[4] = std::cos(theta_fast);
                    q_[5] = std::sin(theta_fast);

                    omega_[1] = - M_PI / unit_time_;
                    omega_[2] = - M_PI / unit_time_;
                }
                else
                {
                    theta_fast += M_PI;
                    theta_slow += M_PI;

                    // 0, 3
                    q_[0] = std::cos(theta_fast);
                    q_[1] = std::sin(theta_fast);
                    q_[6] = std::cos(theta_fast);
                    q_[7] = std::sin(theta_fast);

                    omega_[0] = - M_PI / unit_time_;
                    omega_[3] = - M_PI / unit_time_;

                    // 1, 2
                    q_[2] = std::cos(theta_slow);
                    q_[3] = std::sin(theta_slow);
                    q_[4] = std::cos(theta_slow);
                    q_[5] = std::sin(theta_slow);

                    omega_[1] = - M_PI / unit_time_ / 3;
                    omega_[2] = - M_PI / unit_time_ / 3;
                }
            }

            if (gait_idx_ == 2)
            {
                T theta_fast = - phase_scalar_ / unit_time_ * M_PI + M_PI;
                T theta_slow = - phase_scalar_ / unit_time_ * M_PI / 2 - M_PI / 2;

                // 0, 3
                q_[0] = std::cos(theta_slow);
                q_[1] = std::sin(theta_slow);
                q_[6] = std::cos(theta_slow);
                q_[7] = std::sin(theta_slow);

                omega_[0] = - M_PI / unit_time_ / 2;
                omega_[3] = - M_PI / unit_time_ / 2;

                // 1, 2
                q_[2] = std::cos(theta_fast);
                q_[3] = std::sin(theta_fast);
                q_[4] = std::cos(theta_fast);
                q_[5] = std::sin(theta_fast);

                omega_[1] = - M_PI / unit_time_;
                omega_[2] = - M_PI / unit_time_;
            }

            if (gait_idx_ == 3)
            {
                T theta_fast = - phase_scalar_ / unit_time_ * M_PI + M_PI;
                T theta_slow = - phase_scalar_ / unit_time_ * M_PI / 2 - M_PI;

                // 0, 3
                q_[0] = std::cos(theta_fast);
                q_[1] = std::sin(theta_fast);
                q_[6] = std::cos(theta_fast);
                q_[7] = std::sin(theta_fast);

                omega_[0] = - M_PI / unit_time_;
                omega_[3] = - M_PI / unit_time_;

                // 1, 2
                q_[2] = std::cos(theta_slow);
                q_[3] = std::sin(theta_slow);
                q_[4] = std::cos(theta_slow);
                q_[5] = std::sin(theta_slow);

                omega_[0] = - M_PI / unit_time_ / 2;
                omega_[3] = - M_PI / unit_time_ / 2;
            }

            if (gait_idx_ == 4)
            {
                T theta_fast = - phase_scalar_ / unit_time_ * M_PI + M_PI;
                T theta_slow = - phase_scalar_ / unit_time_ * M_PI / 3.0 + M_PI;

                // 0, 3
                q_[0] = std::cos(theta_fast);
                q_[1] = std::sin(theta_fast);
                q_[6] = std::cos(theta_fast);
                q_[7] = std::sin(theta_fast);

                omega_[0] = - M_PI / unit_time_;
                omega_[3] = - M_PI / unit_time_;

                // 1, 2
                q_[2] = std::cos(theta_slow);
                q_[3] = std::sin(theta_slow);
                q_[4] = std::cos(theta_slow);
                q_[5] = std::sin(theta_slow);

                omega_[0] = - M_PI / unit_time_ / 3;
                omega_[3] = - M_PI / unit_time_ / 3;
            }

            if (gait_idx_ == 5)
            {
                T theta_fast = - phase_scalar_ / unit_time_ * M_PI + M_PI;
                T theta_slow = - phase_scalar_ / unit_time_ * M_PI / 3.0 + M_PI * 2.0 / 3.0;

                // 0, 3
                q_[0] = std::cos(theta_slow);
                q_[1] = std::sin(theta_slow);
                q_[6] = std::cos(theta_slow);
                q_[7] = std::sin(theta_slow);

                omega_[0] = - M_PI / unit_time_ / 3;
                omega_[3] = - M_PI / unit_time_ / 3;

                // 1, 2
                q_[2] = std::cos(theta_fast);
                q_[3] = std::sin(theta_fast);
                q_[4] = std::cos(theta_fast);
                q_[5] = std::sin(theta_fast);

                omega_[1] = - M_PI / unit_time_;
                omega_[2] = - M_PI / unit_time_;
            }

            if (gait_idx_ == 6)
            {
                T theta_1 = - phase_scalar_ / unit_time_ * M_PI / 2.0 + M_PI;
                T theta_2 = - phase_scalar_ / unit_time_ * M_PI / 2.0 + M_PI / 2.0;

                // 0, 3
                q_[0] = std::cos(theta_1);
                q_[1] = std::sin(theta_1);
                q_[6] = std::cos(theta_1);
                q_[7] = std::sin(theta_1);

                omega_[0] = - M_PI / unit_time_ / 2;
                omega_[3] = - M_PI / unit_time_ / 2;

                // 1, 2
                q_[2] = std::cos(theta_2);
                q_[3] = std::sin(theta_2);
                q_[4] = std::cos(theta_2);
                q_[5] = std::sin(theta_2);

                omega_[0] = - M_PI / unit_time_ / 2;
                omega_[3] = - M_PI / unit_time_ / 2;
            }

            if (gait_idx_ == 7)
            {
                if (phase_scalar_ / unit_time_ < 2.0)
                {
                    T theta_1 = - phase_scalar_ / unit_time_ * M_PI + M_PI;

                    // 0, 3
                    q_[0] = std::cos(theta_1);
                    q_[1] = std::sin(theta_1);
                    q_[6] = std::cos(theta_1);
                    q_[7] = std::sin(theta_1);

                    omega_[0] = - M_PI / unit_time_;
                    omega_[3] = - M_PI / unit_time_;
                }
                else
                {
                    T theta_1 = - (phase_scalar_ / unit_time_ - 2.0) * M_PI / 2.0 + M_PI;
                    
                    // 0, 3
                    q_[0] = std::cos(theta_1);
                    q_[1] = std::sin(theta_1);
                    q_[6] = std::cos(theta_1);
                    q_[7] = std::sin(theta_1);

                    omega_[0] = - M_PI / unit_time_ / 2.0;
                    omega_[3] = - M_PI / unit_time_ / 2.0;
                }

                if (phase_scalar_ / unit_time_ < 3.0 || phase_scalar_ / unit_time_ >= 5.0)
                {
                    T theta_2 = phase_scalar_ / unit_time_ < 3.0 ? - (phase_scalar_ / unit_time_ + 1.0) * M_PI / 2.0 + M_PI :  - (phase_scalar_ / unit_time_ - 5.0) * M_PI / 2.0 + M_PI;
                    
                    // 1, 2
                    q_[2] = std::cos(theta_2);
                    q_[3] = std::sin(theta_2);
                    q_[4] = std::cos(theta_2);
                    q_[5] = std::sin(theta_2);

                    omega_[1] = - M_PI / unit_time_ / 2.0;
                    omega_[2] = - M_PI / unit_time_ / 2.0;
                }
                if (phase_scalar_ / unit_time_ >= 3.0 && phase_scalar_ / unit_time_ < 5.0)
                {
                    T theta_2 = - (phase_scalar_ / unit_time_ - 3.0) * M_PI + M_PI;
                    
                    // 1, 2
                    q_[2] = std::cos(theta_2);
                    q_[3] = std::sin(theta_2);
                    q_[4] = std::cos(theta_2);
                    q_[5] = std::sin(theta_2);

                    omega_[1] = - M_PI / unit_time_;
                    omega_[2] = - M_PI / unit_time_;
                }
            }

            if (gait_idx_ == 8)
            {
                T theta_1 = phase_scalar_ / unit_time_ < 2.0 ?  - (phase_scalar_ / unit_time_) * M_PI + M_PI : - (phase_scalar_ / unit_time_ - 2.0) * M_PI / 2.0 + M_PI;
                T theta_2 = phase_scalar_ / unit_time_ < 4.0 ?  - (phase_scalar_ / unit_time_) * M_PI / 2.0 + M_PI : - (phase_scalar_ / unit_time_ - 4.0) * M_PI + M_PI;

                T omega_1 = phase_scalar_ / unit_time_ < 2.0 ? - M_PI / unit_time_ : - M_PI / unit_time_ / 2.0;
                T omega_2 = phase_scalar_ / unit_time_ < 4.0 ? - M_PI / unit_time_ / 2.0 : - M_PI / unit_time_;

                // 0, 3
                q_[0] = std::cos(theta_1);
                q_[1] = std::sin(theta_1);
                q_[6] = std::cos(theta_1);
                q_[7] = std::sin(theta_1);

                omega_[0] = omega_1;
                omega_[3] = omega_1;

                // 1, 2
                q_[2] = std::cos(theta_2);
                q_[3] = std::sin(theta_2);
                q_[4] = std::cos(theta_2);
                q_[5] = std::sin(theta_2);

                omega_[0] = omega_2;
                omega_[3] = omega_2;
            }
        }

        Eigen::Matrix<T, 8, 1> get_status()
        {
            return q_;
        }

        Eigen::Matrix<T, 8, 1> get_velocity()
        {
            // not implemented in manual phase
            return q_dot_;
        }

    private:
        int gait_idx_ = 0;
        int prev_gait_idx_ = 0;
        int target_gait_idx_ = 0;

        T phase_scalar_ = 0.0;
        T dt_ = 0.01;
        T unit_time_ = 0.15;

        std::array<T, n_gait> UNIT = {2.0, 6.0, 4.0, 4.0, 6.0, 6.0, 4.0, 6.0, 6.0};
        std::array<T, n_gait> DUTY_UNIT = {1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0};
        std::array<T, n_gait> TIME;
        std::array<T, n_gait> STANCE_TIME;

        T unit_, duty_unity_, time_, stance_time_;

        Eigen::Matrix<T, 8, 1> q_;
        Eigen::Matrix<T, 8, 1> q_dot_;

        Eigen::Matrix<T, 4, 1> phase_raw_; // -pi to +pi
        std::array<T, 4> phase_; // 0 to 2 * pi
        Eigen::Matrix<T, 4, 1> omega_;
    };
}

#endif