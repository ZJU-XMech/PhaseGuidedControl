#include <eigen3/Eigen/Dense>

void InverseKinematics_v2(Eigen::Ref<Eigen::VectorXd> EndeffectorRef, Eigen::Ref<Eigen::VectorXd> jointRef)
{
    double toe_x = 0.0;
    double toe_y = 0.0;
    double toe_z = 0.0;
    double l_leg = 0.0;
    double abad = 0.0;
    double knee = 0.0;
    double hip_1 = 0.0;
    double hip_2 = 0.0;
    double hip = 0.0;
    // double toe_x, toe_y, toe_z;
    // double l_leg, abad, knee, hip_1, hip_2, hip;
    double tmp_x = 0.0;
    double tmp_y = 0.0;
    double tmp_z = 0.0;
    double tmp_yz = 0.0;
    double tmp_sa = 0.0;
    double tmp_sb = 0.0;
    double tmp_ca = 0.0;
    double tmp_cb = 0.0;
    double l_leg_yz = 0.0;
    double tmp_st = 0.0;
    // double tmp_x, tmp_y, tmp_z, tmp_yz, tmp_sa, tmp_sb, tmp_ca, tmp_cb, l_leg_yz, tmp_st;

    double l_abad = 0.085;
    double l_thigh = 0.201;
    double l_shank = 0.19;

    double max_leg = l_shank + l_thigh - 0.001;

    for (int i = 0; i < 4; ++i)
    {
        toe_x = EndeffectorRef[3 * i];
        toe_y = EndeffectorRef[3 * i + 1];
        toe_z = EndeffectorRef[3 * i + 2] - 0.3;

        if (i % 2 == 0)
        {
            tmp_x = fabs(toe_x);
            tmp_z = fabs(toe_z);
            tmp_y = fabs(toe_y - l_abad);

            tmp_yz = std::sqrt(tmp_y * tmp_y + tmp_z * tmp_z);
            l_leg_yz = std::sqrt(tmp_yz * tmp_yz - l_abad * l_abad); 

            tmp_sb = l_abad / tmp_yz;
            tmp_sb = std::min(1.0, tmp_sb);
            tmp_cb = std::sqrt(1 - tmp_sb * tmp_sb);

            if (toe_y < 0)
            {
                tmp_sa = tmp_z / tmp_yz;
                tmp_ca = tmp_y / tmp_yz;

                tmp_st = tmp_ca * tmp_cb - tmp_sa * tmp_sb;
                abad = - asin(tmp_st);
            }
            else
            {
                if (toe_y - l_abad < 0)
                {
                    tmp_sa = tmp_y / tmp_yz;
                    tmp_ca = tmp_z / tmp_yz;
                    
                    tmp_st = tmp_sb * tmp_ca - tmp_cb * tmp_sa;
                    abad = asin(tmp_st);
                }
                else
                {
                    tmp_sa = tmp_y / tmp_yz;
                    tmp_ca = tmp_z / tmp_yz;

                    tmp_st = tmp_sa * tmp_cb + tmp_sb * tmp_ca;
                    abad = asin(tmp_st);
                }
            }
        }
        else
        {
            tmp_x = fabs(toe_x);
            tmp_z = fabs(toe_z);
            tmp_y = fabs(toe_y + l_abad);

            tmp_yz = std::sqrt(tmp_y * tmp_y + tmp_z * tmp_z);
            l_leg_yz = std::sqrt(tmp_yz * tmp_yz - l_abad * l_abad); 

            tmp_sb = l_abad / tmp_yz;
            tmp_sb = std::min(1.0, tmp_sb);
            tmp_cb = std::sqrt(1 - tmp_sb * tmp_sb);

            if (toe_y > 0)
            {
                tmp_sa = tmp_z / tmp_yz;
                tmp_ca = tmp_y / tmp_yz;

                tmp_st = tmp_ca * tmp_cb - tmp_sa * tmp_sb;
                abad = asin(tmp_st);
            }
            else
            {
                if (toe_y + l_abad > 0)
                {
                    tmp_sa = tmp_y / tmp_yz;
                    tmp_ca = tmp_z / tmp_yz;
                    
                    tmp_st = tmp_sb * tmp_ca - tmp_cb * tmp_sa;
                    abad = -asin(tmp_st);
                }
                else
                {
                    tmp_sa = tmp_y / tmp_yz;
                    tmp_ca = tmp_z / tmp_yz;

                    tmp_st = tmp_sa * tmp_cb + tmp_sb * tmp_ca;
                    abad = -asin(tmp_st);
                }
            }
        }

        // hip and knee
        l_leg = std::sqrt(tmp_yz * tmp_yz - l_abad * l_abad + tmp_x * tmp_x);

        if (l_leg > max_leg)
        {
            toe_x *= max_leg / l_leg;
            toe_y *= max_leg / l_leg;
            toe_z *= max_leg / l_leg;
            l_leg = max_leg;
        }

        knee = M_PI - acos((l_thigh * l_thigh + l_shank * l_shank - l_leg * l_leg) / 2.0 / l_thigh / l_shank);
        hip_1 = std::atan(toe_x / l_leg_yz);
        hip_2 = acos((l_thigh * l_thigh + l_leg * l_leg - l_shank * l_shank) / 2.0 / l_thigh / l_leg);
        hip = hip_1 - hip_2;

        jointRef[3 * i] = abad;
        jointRef[3 * i + 1] = hip;
        jointRef[3 * i + 2] = knee;
    }
}