//
// Created by wooden on 2020/6/13.
//

#ifndef BLACKPANTHER_MP_USERPARAMETERS_H
#define BLACKPANTHER_MP_USERPARAMETERS_H

#include "ControlParameters/ControlParameters.h"
#define Ttype double

class MP_UserParameters : public ControlParameters {
public:
    MP_UserParameters()
            : ControlParameters("user-parameters"),
              INIT_PARAMETER(stand_height),
              INIT_PARAMETER(up_height),
              INIT_PARAMETER(down_height),
              INIT_PARAMETER(Freq),
              INIT_PARAMETER(Stiffness),
              INIT_PARAMETER(DDamping),
              INIT_PARAMETER(Vx),
              INIT_PARAMETER(Vy),
              INIT_PARAMETER(Omega),
              INIT_PARAMETER(Lean_middle),
              INIT_PARAMETER(Period),
              INIT_PARAMETER(lam),
              INIT_PARAMETER(HeightVariable),
              INIT_PARAMETER(OrientationThreshold),
              INIT_PARAMETER(LinVelThreshold),
              INIT_PARAMETER(AngVelThreshold)
    {}
    DECLARE_PARAMETER(Ttype, stand_height);
    DECLARE_PARAMETER(Ttype, up_height);
    DECLARE_PARAMETER(Ttype, down_height);
    DECLARE_PARAMETER(Ttype, Freq);
    DECLARE_PARAMETER(Ttype, Stiffness);
    DECLARE_PARAMETER(Ttype, DDamping);
    DECLARE_PARAMETER(Ttype, Vx);
    DECLARE_PARAMETER(Ttype, Vy);
    DECLARE_PARAMETER(Ttype, Omega);
    DECLARE_PARAMETER(Ttype, Lean_middle);
    DECLARE_PARAMETER(Ttype, Period);
    DECLARE_PARAMETER(Ttype, lam);
    DECLARE_PARAMETER(Ttype, HeightVariable);
    DECLARE_PARAMETER(Ttype, OrientationThreshold);
    DECLARE_PARAMETER(Ttype, LinVelThreshold);
    DECLARE_PARAMETER(Ttype, AngVelThreshold);
};

#endif //BLACKPANTHER_MP_USERPARAMETERS_H
