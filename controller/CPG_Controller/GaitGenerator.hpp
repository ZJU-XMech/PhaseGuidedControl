#include <eigen3/Eigen/Core>
#include <cmath>

template<typename T>
inline void getEndEffectorRefByPhase(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> EndeffectorRef, Eigen::Matrix<T, Eigen::Dynamic, 1> phase, T dx, T dy, T ddy)
{
    for (int i = 0; i < 4; ++i)
    {
        EndeffectorRef[3*i] = std::cos(phase[i] * M_PI) * dx;
        if (i < 2)
        {
            EndeffectorRef[3*i+1] = std::cos(phase[i] * M_PI) * (dy + ddy);
        }
        else
        {
            EndeffectorRef[3*i+1] = std::cos(phase[i] * M_PI) * (dy - ddy);
        }
        
        if (phase[i] > 0)
        {
            // swing, phase is from 1 to 0
            EndeffectorRef[3*i+2] = - 0.3 - 0.08 * (std::cos(phase[i] * 2.0 * M_PI) - 1.0) * 0.5;
        }
        else
        {
            // stance, phase is from 0 to -1
            EndeffectorRef[3*i+2] = - 0.3;
        }
    }
};