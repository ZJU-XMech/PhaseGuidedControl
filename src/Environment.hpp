#include <stdlib.h>
#include <cstdint>
#include <set>
#include <random>
#include <iostream>
#include <cmath>

#include "raisim/OgreVis.hpp"
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"
#include "CPG.hpp"
#include "ManualPhase.hpp"

#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
#include "raisimCustomizedImguiPanel.hpp"


#define deg2rad(ang) ((ang)*M_PI / 180.0)
#define rad2deg(ang) ((ang)*180.0 / M_PI)
#define PI 3.1415926
#define MAGENTA "\033[35m"           /* Magenta */
#define BOLDCYAN "\033[1m\033[36m"   /* Bold Cyan */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */

#define motor_kt 0.05
#define motor_R 0.173
#define motor_tau_max 3.0
#define motor_battery_v 24
#define motor_damping 0.01
#define motor_friction 0.2

namespace raisim
{
constexpr int historyLength_ = 13;
// constexpr int commandHistoryLength_ = 61;
constexpr int nJoints_ = 12;
constexpr int max_steps = 960;
constexpr int cols = 31;
constexpr int num_reference = 729;

inline void modifyEndeffectorRef(Eigen::Ref<Eigen::VectorXd> EndeffectorRef, double gx, double gy, double factor)
{
    // from world frame to body frame
    for (int i = 0; i < 4; ++i)
    {
        if (i < 1.5)
        {
            EndeffectorRef[3*i+2] -= (0.212 + EndeffectorRef[3*i]) * gx * factor;
        }
        else
        {
            EndeffectorRef[3*i+2] += (0.212 - EndeffectorRef[3*i]) * gx * factor;
        }

        if (i % 2 == 0)
        {
            EndeffectorRef[3*i+2] += (0.136 - EndeffectorRef[3*i+1]) * gy * factor;
        }
        else
        {
            EndeffectorRef[3*i+2] -= (0.136 + EndeffectorRef[3*i+1]) * gy * factor;
        }
    }
}

inline void InverseKinematics_v2(Eigen::Ref<Eigen::VectorXd> EndeffectorRef, Eigen::Ref<Eigen::VectorXd> jointRef)
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
    double min_leg = std::sqrt(l_shank * l_shank + l_thigh * l_thigh - 2 * l_thigh * l_shank * std::cos(3.14-2.2)) + 0.001;

    for (int i = 0; i < 4; ++i)
    {
        toe_x = EndeffectorRef[3 * i];
        toe_y = EndeffectorRef[3 * i + 1];
        toe_z = EndeffectorRef[3 * i + 2];

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

        if (l_leg < min_leg)
        {
            toe_x *= min_leg / l_leg;
            toe_y *= min_leg / l_leg;
            toe_z *= min_leg / l_leg;
            l_leg = min_leg;
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

class ENVIRONMENT : public RaisimGymEnv
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef typename Eigen::Matrix<double, 3, 1> Position;

    explicit ENVIRONMENT(const std::string &resourceDir, const YAML::Node &cfg, bool visualizable) : RaisimGymEnv(resourceDir, cfg), distribution_(0.0, 0.2), visualizable_(visualizable)
    {

        // specify materials
        materials_.setMaterialPairProp("robot", "robot", 0.8, 0.0, 0.0);
        materials_.setMaterialPairProp("default", "robot", 0.8, 0.0, 0.0);
        world_->updateMaterialProp(materials_);

        // READ_YAML(int, model_idx, cfg["model"])

        // add objects
        minicheetah_ = world_->addArticulatedSystem(resourceDir_ + "/urdf/black_panther.urdf");
        minicheetahRef_ = world_->addArticulatedSystem(resourceDir_ + "/urdf/black_panther.urdf");
        minicheetah_->setControlMode(ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
        // auto ground = world_->addGround();

        int xSamples = 100;
        int ySamples = 100;
        double xScale = 20;
        double yScale = 20;
        READ_YAML(float, groundAmplitude_, cfg["groundAmplitude_"])
        noisifyHeightMap(xSamples, ySamples, groundAmplitude_);
        auto ground = world_->addHeightMap(xSamples, ySamples, xScale, yScale, 0.0, 0.0, heights_, "terrain");
        world_->setERP(0, 0);

        // get robot dim
        gcDim_ = minicheetah_->getGeneralizedCoordinateDim();
        gvDim_ = minicheetah_->getDOF();

        // initialize containers
        gc_.setZero(gcDim_);
        gc_init_.setZero(gcDim_);
        gv_.setZero(gvDim_);
        gv_init_.setZero(gvDim_);
        torque_.setZero(nJoints_);
        torqueFull_.setZero(gvDim_);
        pTarget_.setZero(gcDim_);
        vTarget_.setZero(gvDim_);
        pTarget12_.setZero(nJoints_);
        pTargetPast_.setZero(nJoints_);

        // this is nominal configuration
        jointNominalConfig_ << 0.0, -0.7, 1.4, // RF
            0.0, -0.7, 1.4,                    // LF
            0.0, -0.7, 1.4,                    // RH
            0.0, -0.7, 1.4;                    // LH
        gc_init_ << 0.0, 0.0, body_height_ + 0.0275 + 0.003,
            1.0, 0.0, 0.0, 0.0,
            jointNominalConfig_;
        
        EndeffectorVelScale_ << 1.0, 1.0, 0.0,
                                1.0, 1.0, 0.0,
                                1.0, 1.0, 0.0,
                                1.0, 1.0, 0.0;

        random_init.setZero(gcDim_);
        random_init.tail<19>() = gc_init_.tail<19>();

        minicheetah_->setState(gc_init_, gv_init_);
        gc_init_[0] = 4.5;
        gc_init_[1] = 4.5;
        minicheetahRef_->setState(gc_init_, gv_init_);

        EndEffector_.setZero(nJoints_);
        EndeffectorRef_.setZero(nJoints_);
        EndeffectorRefFuture_.setZero(3 * nJoints_);
        
        EndEffectorOffset_ << 0.212, -0.143, 0.0,                 // FR
                            0.212, 0.143, 0.0,                    // FL
                            -0.212, -0.143, 0.0,                  // HR
                            -0.212, 0.143, 0.0;                   // HL
        
        // set pd gains
        // Eigen::VectorXd jointPgain(gvDim_), jointDGain_(gvDim_);
        jointPGain_.setZero(gvDim_);
        jointPGain_.tail<nJoints_>().setConstant(40.0);
        jointDGain_.setZero(gvDim_);
        jointDGain_.tail<nJoints_>().setConstant(1.0);
        minicheetah_->setPdGains(jointPGain_, jointDGain_);
        minicheetah_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

        torqueStd_.setZero(12);
        torqueStd_ << 18.0, 18.0, 27.0,
                        18.0, 18.0, 27.0,
                        18.0, 18.0, 27.0,
                        18.0, 18.0, 27.0;
        jointUpperLimits_.setZero(gvDim_);
        jointUpperLimits_.tail<nJoints_>() = torqueStd_ * 0.8;
        minicheetah_->setActuationLimits(jointUpperLimits_, - jointUpperLimits_);

        // init noise scales
        gcNoiseScale_.setZero(gcDim_);
        gvNoiseScale_.setZero(gvDim_);

        READ_YAML(double, bodyHeightNoiseScale_, cfg["bodyHeightNoiseScale_"])
        READ_YAML(double, bodyAngularPositionNoiseScale_, cfg["bodyAngularPositionNoiseScale_"])
        READ_YAML(double, JointPositionNoiseScale_, cfg["JointPositionNoiseScale_"])
        READ_YAML(double, bodyLinearVelocityNoiseScale_, cfg["bodyLinearVelocityNoiseScale_"])
        READ_YAML(double, bodyAngularVelocityNoiseScale_, cfg["bodyAngularVelocityNoiseScale_"])
        READ_YAML(double, JointVelocityNoiseScale_, cfg["JointVelocityNoiseScale_"])
        READ_YAML(double, bodyForceAmplitude_, cfg["bodyForceAmplitude_"])
        READ_YAML(double, noiseFtr, cfg["noiseFtr"])
        READ_YAML(double, bodyMassNoiseScale_, cfg["bodyMassNoiseScale_"])
        READ_YAML(double, bodyComOffsetNoiseScale_, cfg["bodyComOffsetNoiseScale_"])
        READ_YAML(double, bodyShapeNoiseScale_, cfg["bodyShapeNoiseScale_"])
        READ_YAML(double, mixParam_, cfg["mixParam_"])
        READ_YAML(double, contactMixParam_, cfg["contactMixParam_"])
        READ_YAML(double, frictionNoiseScale_, cfg["frictionNoiseScale_"])
        READ_YAML(double, frictionMean_, cfg["frictionMean_"])
        READ_YAML(double, costScaleIncreaseRate_, cfg["costScaleIncreaseRate_"])
        READ_YAML(double, default_swing_height_, cfg["swingHeight_"])
        READ_YAML(double, bodyPosModification_, cfg["bodyPosModification_"])
        READ_YAML(bool, autoChangeGait_, cfg["autoChangeGait"])
        READ_YAML(double, cmdCostCoeff_, cfg["cmdCostCoeff"])
        READ_YAML(double, bodyPosRewardCoeff_, cfg["bodyPosRewardCoeff"])

        swing_height_ = default_swing_height_;

        gcNoiseScale_.segment<1>(2) = Eigen::VectorXd::Constant(1, bodyHeightNoiseScale_);
        gcNoiseScale_.segment<4>(3) = Eigen::VectorXd::Constant(4, 0.0);
        gcNoiseScale_.segment<12>(7) = Eigen::VectorXd::Constant(12, JointPositionNoiseScale_);

        gvNoiseScale_.segment<3>(0) = Eigen::VectorXd::Constant(3, bodyLinearVelocityNoiseScale_);
        gvNoiseScale_.segment<3>(3) = Eigen::VectorXd::Constant(3, bodyAngularVelocityNoiseScale_);
        gvNoiseScale_.segment<12>(6) = Eigen::VectorXd::Constant(12, JointVelocityNoiseScale_);

        // cpg
        cpg_ = CPG<double>();
        READ_YAML(bool, walkOnly_, cfg["walkOnly"])
        READ_YAML(bool, gallopOnly_, cfg["gallopOnly"])
        READ_YAML(bool, useManualPhase_, cfg["useManualPhase"])
        READ_YAML(bool, threeLegGait_, cfg["threeLegGait"])
        READ_YAML(int, specificGait_, cfg["specificGait"])

        manual_phase_ = ManualPhase<double>();
        
        // MUST BE DONE FOR ALL ENVIRONMENTS
        obDim_ = 44;
        actionDim_ = nJoints_;
        actionMean_.setZero(actionDim_);
        actionStd_.setZero(actionDim_);
        obMean_.setZero(obDim_);
        obStd_.setZero(obDim_);

        // action & observation scaling
        READ_YAML(double, actionStdScalar_, cfg["actionStdScalar_"])
        actionMean_ = gc_init_.tail<nJoints_>();
        // actionStd_.setConstant(actionStdScalar_);
        actionStd_.setConstant(actionStdScalar_);

        // torqueStd_ = 18.0;
        jointVelStd_ = 30.0;
        jointAngStd_ = actionStdScalar_;
        bodyLinVelStd_ = 1.0;
        bodyAngVelStd_ = 3.0;
        bodyOrientationStd_ = 0.3;

        obMean_ << 0.0, 0.0, 1.0,                            // gravity axis
            Eigen::VectorXd::Constant(6, 0.0),        // body lin/ang vel
            jointNominalConfig_, // joint position
            Eigen::VectorXd::Constant(12, 0.0),       // joint velocity
            Eigen::VectorXd::Constant(8, 0.0),        // contact
            0.0, 0.0, 0.0;                            // goal

        obStd_ << Eigen::VectorXd::Constant(3, 1.0 / bodyOrientationStd_), // gravity axis
            Eigen::VectorXd::Constant(3, 1.0 / bodyLinVelStd_), // body linear velocities
            Eigen::VectorXd::Constant(3, 1.0 / bodyAngVelStd_), // body angular velocities
            Eigen::VectorXd::Constant(12, 1.0 / jointAngStd_),  // joint angles
            1.0/12.0, 1.0/30.0, 1.0/30.0, 1.0/12.0, 1.0/30.0, 1.0/30.0, 1.0/12.0, 1.0/30.0, 1.0/30.0, 1.0/12.0, 1.0/30.0, 1.0/30.0,
            Eigen::VectorXd::Constant(8, 1.0 / 1.0),                  // contact
            1.0, 1.0 / 0.5, 1.0;                                // goal
        
        extraRef_.setZero(52); // body vel: 3; body height: 1; end effector ref: 12; joint pos ref: 12; joint vel ref: 12; sim torque: 12;

        // ob filter
        joint_filter_freq = 20.0;
        joint_filter_alpha = 2.0 * M_PI * 0.01 * joint_filter_freq / (2.0 * M_PI * 0.01 * joint_filter_freq + 1.0);

        // Reward coefficients
        READ_YAML(double, costScale1_, cfg["costScale1_"])
        READ_YAML(double, costScale2_, cfg["costScale2_"])

        READ_YAML(double, torqueCostScale_, cfg["torqueCostScale_"])

        READ_YAML(double, terminalRewardCoeff_, cfg["terminalRewardCoeff_"])
        READ_YAML(double, EndEffectorRewardFactor_, cfg["EndEffectorRewardFactor_"])
        READ_YAML(double, JointRewardFactor_, cfg["JointRewardFactor_"])
        READ_YAML(double, BodyPosRewardFactor_, cfg["BodyPosRewardFactor_"])
        READ_YAML(double, BodyHeightRewardFactor_, cfg["BodyHeightRewardFactor_"])
        READ_YAML(double, cmdRewardFactor_, cfg["cmdRewardFactor_"])
        READ_YAML(double, flipFriction_, cfg["flipFriction_"])
        READ_YAML(double, toeLateralOffset_, cfg["toeLateralOffset_"])

        // Control distribution
        getRandomCommand();

        jointRef_.setZero(nJoints_);
        jointRefVel_.setZero(nJoints_);
        jointRefTemp_.setZero(2*nJoints_);
        EndeffectorRefTemp_.setZero(2*nJoints_);

        READ_YAML(double, cricculumFactorCmd_, cfg["cricculumFactorCmdInit_"]);
        READ_YAML(double, cricculumFactorMimic_, cfg["cricculumFactorMimicInit_"]);
        READ_YAML(double, cricculumFactorCmdFinal_, cfg["cricculumFactorCmdFinal_"]);
        READ_YAML(double, cricculumFactorMimicFinal_, cfg["cricculumFactorMimicFinal_"]);
        READ_YAML(double, cricculumIncreaseFactor_, cfg["cricculumIncreaseFactor_"]);

        // current_step_ = static_cast<int>((u(e) / 2.0 + 0.5) * 15);
        initInnerState();

        gui::rewardLogger.init({"vx_des",
                                "vy_des",
                                "yawrate_des",
                                "vx",
                                "vy",
                                "yawrate",
                                "cmdFactor",
                                "mimicFactor",
                                "z",
                                "jointVel",
                                "height",
                                "cmdReward",
                                "mimicReward",
                                "End_Effector",
                                "Joint_Traj",
                                "Body_pos",
                                "torqueCost",
                                "torqueMax",
                                "bodyHeight",
                                "linVelReward",
                                "angVelReward"});
        gui::showContacts = true;

        // print collision bofy index in first env
        if (visualizable_)
        {
            RSINFO("get collision bodies")
            for (auto &colBody : minicheetah_->getCollisionBodies())
            {
                RSINFO(colBody.name)
                RSINFO(colBody.localIdx)
                RSINFO(colBody.posOffset)
            }
            for (auto &colBody : minicheetah_->getVisColOb())
            {
                RSINFO("get vis obj")
                RSINFO(colBody.name)
                RSINFO(colBody.localIdx)
                RSINFO(colBody.offset)
                RSINFO(colBody.fileName)
                RSINFO(colBody.visShapeParam[0])
                RSINFO(colBody.shape)
            }
            RSINFO("get index by name")
            minicheetah_->printOutBodyNamesInOrder();
            RSINFO("Frames")
            minicheetah_->printOutFrameNamesInOrder();
        }

        // foot collision and contact
        footCollision_.push_back(minicheetah_->getCollisionBody("toe_fr/0"));
        footCollision_.push_back(minicheetah_->getCollisionBody("toe_fl/0"));
        footCollision_.push_back(minicheetah_->getCollisionBody("toe_hr/0"));
        footCollision_.push_back(minicheetah_->getCollisionBody("toe_hl/0"));

        // assign materials to feet
        for (auto &foot : footCollision_)
        {
            foot.setMaterial("robot");
        }

        // visualize if it is the first environment
        if (visualizable_)
        {
            auto vis = OgreVis::get();

            // these method must be called before initApp
            vis->setWorld(world_.get());
            vis->setWindowSize(1440, 1080);
            vis->setImguiSetupCallback(imguiSetupCallback);
            vis->setImguiRenderCallback(imguiRenderCallBack);
            vis->setKeyboardCallback(raisimKeyboardCallback);
            vis->setSetUpCallback(setupCallback);
            vis->setAntiAliasing(2);

            // starts visualizer thread
            vis->initApp();

            minicheetahVisual_ = vis->createGraphicalObject(minicheetah_, "MiniCheetah");
            // vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
            vis->createGraphicalObject(ground, "floor", "checkerboard_green");
            arrow_ = vis->addVisualObject("arrow", "arrowMesh", "red", {0.1, 0.1, 0.2}, false, OgreVis::RAISIM_OBJECT_GROUP);
            desired_fps_ = 60.;
            vis->setDesiredFPS(desired_fps_);
        }
    }

    ~ENVIRONMENT() final = default;

    void init() final {}

    void initInnerState()
    {
        badlyConditioned_ = false;
        bodyForceTimer_ = u(e) / 2.0 + 1.0;

        if (u(e) > 0.5)
        {
            bodyForceTimer_ = 10.0;
        }

        bodyForceVisTimer_ = 0.0;

        bodyForceExt_.setZero();
        bodyForcePos_.setZero();

        if (u(e) > -0.9)
        {
            switchCommandTimer_ = u(e) * 0.5 + 1.5;
        }
        else
        {
            switchCommandTimer_ = 10.0;
        }

        if (!autoChangeGait_)
        {
            switchCommandTimer_ = 86400.0;
        }

        if (walkOnly_ || gallopOnly_)
        {
            switchCommandTimer_ = 10.0;
        }

        jointPositionHistory_.setZero(static_cast<int>(nJoints_ * historyLength_));
        jointVelocityHistory_.setZero(static_cast<int>(nJoints_ * historyLength_));
        pTargetPast_.setZero(nJoints_);
        smoothContactState_.setZero(4);
        prevBodyAngularVelocity_.setZero();
        // commandHistory_.setZero(static_cast<int>(4 * commandHistoryLength_));

        // target velocity
        targetLinearVelocity_.setZero(3);
        targetAngularVelocity_ = 0.0;

        // cpg_status
        cpg_status_.setZero(8);
        cpg_status_dot_.setZero(8);
        stance_time_ = cpg_.get_stance_time();

        for (auto& force: footContactForce_)
        {
            force.setZero();
        }
    }

    void reset() final
    {
        gc_init_[0] = 0.0;
        gc_init_[1] = 0.0;
        minicheetah_->setState(gc_init_, gv_init_);

        gc_init_[0] = 4.5;
        gc_init_[1] = 4.5;
        minicheetahRef_->setState(gc_init_, gv_init_);

        cpg_.reset();
        manual_phase_.reset();
        manual_phase_command_ = false;
        getRandomCommand();

        initInnerState();
        noisifyDynamics();
        updateState();
        updateObservation();

        if (visualizable_)
        {
            gui::rewardLogger.clean();
        }
    }

    void noisifyHeightMap(const int xSamples, const int ySamples, const double amplitude)
    {
        int numPixels_ = xSamples * ySamples;
        heights_.resize(numPixels_);
        bool zeroFlag = false;

        for (auto &pixel : heights_)
        {
            if (zeroFlag)
            {
                pixel = 0.0;
                zeroFlag = false;
            }
            else
            {
                pixel = amplitude * u(e);
                zeroFlag = true;
            }
        }

        heights_.resize(numPixels_, 0);
    }

    inline void applyRandomForce()
    {
        if (bodyForceTimer_ < 0)
        {
            // RSINFO_IF(visualizable_, "bodyForceTimer_" << bodyForceTimer_)

            bodyForcePos_ = {u(e)*0.1, u(e)*0.2, 0};
            bodyForceExt_ = {u(e), u(e), u(e)};
            bodyForceExt_ *= bodyForceAmplitude_ * costScale1_;

            bodyForceVisTimer_ = 0.4 * costScale2_;
            bodyForceTimer_ = u(e) + 1.5 + bodyForceVisTimer_;

            if (u(e) > 0.2)
            {
                bodyForceTimer_ = 10.0;
            }

            // RSINFO_IF(visualizable_, "Next bodyForceTimer_" << bodyForceTimer_)

            minicheetah_->setExternalForce(0, bodyForcePos_, bodyForceExt_);
        }
        else
        {
            if (bodyForceVisTimer_ > 0)
            {
                minicheetah_->setExternalForce(0, bodyForcePos_, bodyForceExt_);
            }
        }
        
        bodyForceTimer_ -= simulation_dt_;
        bodyForceVisTimer_ -= simulation_dt_;
    }

    float step(const Eigen::Ref<EigenVec> &action) final
    {
        // prevent bad conditions
        RSFATAL_IF(isinf(action.norm()), "action is inf" << std::endl
                                                        << action.transpose());
        RSFATAL_IF(isnan(action.norm()), "action is nan" << std::endl
                                                        << action.transpose());
        if (isnan(action.norm()))
        {
            badlyConditioned_ = true;
        }
        if (isinf(action.norm()))
        {
            badlyConditioned_ = true;
        }

        double stepReward = 0.0;

        updateState();

        // action scaling
        pTarget12_ = action.cast<double>().head<nJoints_>();
        pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
        pTarget12_ += actionMean_;

        int hold_leg = -1;
        if (useManualPhase_ && manual_phase_command_)
        {
            hold_leg = manual_phase_.get_hold_leg();
        }
        else
        {
            hold_leg = cpg_.get_hold_leg();
        }

        if (hold_leg >= 0)
        {
            double effectfactor = (cpg_status_.segment<2>(2*hold_leg) - Eigen::Matrix<double, 2, 1>(0.0, -1.0)).norm();
            effectfactor = effectfactor > 0.2 ? 1.0 : effectfactor / 0.2;
        
            Eigen::Matrix<double, 3, 1> hold_state(0.0, -1.0, 2.0);
            pTarget12_.segment<3>(3*hold_leg) = pTarget12_.segment<3>(3*hold_leg) * effectfactor + hold_state * (1.0 - effectfactor);
        }

        auto pTargetMax = gc_.tail<12>() + (torqueStd_*0.8 + gv_.tail<12>().cwiseProduct(jointDGain_.tail<12>())).cwiseQuotient(jointPGain_.tail<12>());
        auto pTargetMin = gc_.tail<12>() + (- torqueStd_*0.8 + gv_.tail<12>().cwiseProduct(jointDGain_.tail<12>())).cwiseQuotient(jointPGain_.tail<12>());
        pTarget12_ = pTarget12_.cwiseMax(pTargetMin);
        pTarget12_ = pTarget12_.cwiseMin(pTargetMax);
        actionFilter();

        // pTarget12_ = gc_init_.tail<12>();
        pTarget_.tail<nJoints_>() = pTarget12_;

        torque0_ = torque_;
        updatePDGeneralizedForces();
        
        // minicheetah_->setGeneralizedForce(torqueFull_);
        minicheetah_->setPdTarget(pTarget_, vTarget_);
        auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
        auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);

        for (int i = 0; i < loopCount; i++)
        {
            checkConditionInSimulation();

            world_->integrate();
            
            applyRandomForce();
            if (switchCommandTimer_ > 0.0)
            {
                switchCommandTimer_ -= simulation_dt_;
            }
            else
            {
                switchCommandTimer_ = 10.0;
                if (u(e) > 0)
                {
                    // 50% change speed
                    changeSpeed();
                }
                else
                {
                    if (u(e) < 0.8)
                    {
                        // 50% * 90% change gait
                        changeGait();
                    }
                    else
                    {
                        // 50% * 10% change both
                        changeSpeed();
                        changeGait();
                    }
                }
            }

            if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
            {
                bodyForceExtNorm_ = bodyForceExt_.norm();
                minicheetah_->getPosition(0, bodyForcePos_, bodyForcePosGlobal_);
                bodyForceExtVis_ = bodyForceExt_;
                bodyForceExtVis_ /= bodyForceExtNorm_;

                if (bodyForceVisTimer_ > 0)
                {
                    // visualize force
                    updateRot();
                    zaxisToRotMat(bodyForceExtVis_, rotForceLocal_);
                    rotmatmul(rot_, rotForceLocal_, rotForceGlobal_);

                    arrow_->setPosition(bodyForcePosGlobal_);
                    arrow_->setScale(bodyForceExtNorm_/bodyForceAmplitude_/2, bodyForceExtNorm_/bodyForceAmplitude_/2, bodyForceExtNorm_/bodyForceAmplitude_/2);
                    arrow_->setOrientation(rotForceGlobal_);
                }
                else
                {
                    // unvisiable
                    rotForceGlobal_.setZero();

                    arrow_->setPosition(bodyForcePosGlobal_);
                    arrow_->setScale(0.001, 0.001, 0.001);
                    arrow_->setOrientation(rotForceGlobal_);
                }

                OgreVis::get()->renderOneFrame();
            }

            visualizationCounter_++;
        }

        updateCPG();
        updateState();
        if (!badlyConditioned_)
        {
            updateRot();
            stepReward += calculateReward();
        }
        else
        {
            stepReward += 0.0;
        }

        if (visualizeThisStep_)
        {
            // reset camera
            auto vis = OgreVis::get();

            vis->select(minicheetahVisual_->at(0), false);
            vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
        }

        RSWARN_IF(isnan(stepReward), "stepReward is NaN!!!" << std::endl);
        RSWARN_IF(isinf(stepReward), "stepReward is Inf!!!" << std::endl);

        return stepReward;
    }

    inline double calculateReward()
    {
        double reward = 0.0;
        double mimicReward;
        double cmdReward;
        double torqueCost;

        double linVelReward;
        double angVelReward;
        double posVelReward;

        double maxTorqueRatio;

        mimicReward = DeepMimicRewardUpdate();

        bodyLinearVel_ = rot_.e().transpose() * gv_.segment<3>(0);
        bodyLinearVel_[2] = 0.0;

        linVelReward = cmdCostCoeff_ * (bodyLinearVel_ - targetLinearVelocity_).squaredNorm();
        angVelReward = cmdCostCoeff_ * ((rot_.e().transpose() * gv_.segment(3, 3))(2) - targetAngularVelocity_) * ((rot_.e().transpose() * gv_.segment(3, 3))(2) - targetAngularVelocity_);
        posVelReward = cmdCostCoeff_ * (rot_.e().transpose() * gv_.segment(3, 3)).head<2>().squaredNorm();
        cmdReward = cmdRewardFactor_ * exp(linVelReward + angVelReward + posVelReward);

        torqueCost = 0.5 * torqueCostScale_ * exp(-torque_.squaredNorm() * 0.0025) + 0.5 * torqueCostScale_ * exp(-(torque_ - torque0_).squaredNorm() / control_dt_ * 0.25);

        reward = cricculumFactorMimic_ * mimicReward + cricculumFactorCmd_ * cmdReward + torqueCost;

        if (visualizeThisStep_)
        {
            gui::rewardLogger.log("vx_des", targetLinearVelocity_[0]);
            gui::rewardLogger.log("vx", bodyLinearVel_[0]);
            gui::rewardLogger.log("vy_des", targetLinearVelocity_[1]);
            gui::rewardLogger.log("vy", bodyLinearVel_[1]);
            gui::rewardLogger.log("yawrate_des", targetAngularVelocity_);
            gui::rewardLogger.log("yawrate", gv_(5));
            gui::rewardLogger.log("cmdFactor", cricculumFactorCmd_);
            gui::rewardLogger.log("mimicFactor", cricculumFactorMimic_);
            gui::rewardLogger.log("z", rot_.e().row(2)[2]);
            gui::rewardLogger.log("height", gc_(2));
            gui::rewardLogger.log("jointVel", JointVelReward);
            gui::rewardLogger.log("cmdReward", cmdReward);
            gui::rewardLogger.log("linVelReward", linVelReward);
            gui::rewardLogger.log("angVelReward", angVelReward);
            gui::rewardLogger.log("mimicReward", mimicReward);
            gui::rewardLogger.log("End_Effector", EndEffectorReward);
            gui::rewardLogger.log("Joint_Traj", JointReward);
            gui::rewardLogger.log("Body_pos", BodyPosReward);
            // gui::rewardLogger.log("Attitude", BodyAttitudeReawrd);
            gui::rewardLogger.log("torqueCost", torqueCost);
            gui::rewardLogger.log("bodyHeight", BodyHeightReward);
            gui::rewardLogger.log("torqueMax", torque_.cwiseProduct(torqueStd_.cwiseInverse()).cwiseAbs().maxCoeff());
        }

        RSWARN_IF(isinf(reward), "reward is Inf!!!" << std::endl
                                << "mimicReward: " << mimicReward << "; cricculumFactorMimic_: " << cricculumFactorMimic_ << std::endl
                                << "cmdReward: " << cmdReward << "; cricculumFactorCmd_: " << cricculumFactorCmd_ << std::endl
                                << "torqueCost: " << torqueCost << std::endl);
        RSWARN_IF(isnan(reward), "reward is NaN!!!" << std::endl
                                << "mimicReward: " << mimicReward << "; cricculumFactorMimic_: " << cricculumFactorMimic_ << std::endl
                                << "cmdReward: " << cmdReward << "; cricculumFactorCmd_: " << cricculumFactorCmd_ << std::endl
                                << "torqueCost: " << torqueCost << std::endl);


        return reward;
    }

    inline void actionFilter()
    {
        pTarget12_ = mixParam_ * pTarget12_ + (1 - mixParam_) * pTargetPast_;
        pTargetPast_ = pTarget12_;
    }

    inline void updatePDGeneralizedForces()
    {
        torque_ = (pTarget_.tail<nJoints_>() - gc_.tail<nJoints_>()).cwiseProduct(jointPGain_.tail<nJoints_>()) - (vTarget_.tail<nJoints_>() - gv_.tail<nJoints_>()).cwiseProduct(jointDGain_.tail<nJoints_>());
    }

    inline double sgn(double x)
    {
        if (x > 0) { return 1.0; }
        else { return -1.0; }
    }

    inline void applyMotorModel(Eigen::VectorXd q_des)
    {
        double gear_ratio[12] = {6, 6, 9.33, 6, 6, 9.33, 6, 6, 9.33, 6, 6, 9.33};
        double tauDesMotor = 0.0;    // desired motor torque
        double iDes = 0.0;           // desired motor iq
        double bemf = 0.0;           // Back electromotive force
        double vDes = 0.0;           // desired motor voltage
        double vActual = 0.0;        // real motor torque
        double tauActMotor = 0.0;    // real motor torque

        for (int i = 0; i < 12; i++) {
            tauDesMotor = torque_[i] / gear_ratio[i];
            iDes = tauDesMotor / (motor_kt * 1.5);
            bemf = q_des[i] * gear_ratio[i] * motor_kt * 2;
            vDes = iDes * motor_R + bemf;
            vActual = fmin(fmax(vDes, -motor_battery_v), motor_battery_v);
            tauActMotor = 1.5 * motor_kt * (vActual - bemf) / motor_R;
            torque_[i] = gear_ratio[i] * fmin(fmin(-motor_tau_max, tauActMotor), motor_tau_max);
            
            torque_[i] = torque_[i] - motor_damping * q_des[i] - motor_friction * sgn(q_des[i]);
        }
    }

    void increaseCostScale(const double in)
    {
        costScale1_ = std::pow(costScale1_, in);
        costScale2_ = std::pow(costScale2_, in);
    }

    void increaseCostScale1(const double in)
    {
        costScale1_ = std::pow(costScale1_, in);
    }

    void increaseCostScale2(const double in)
    {
        costScale2_ = std::pow(costScale2_, in);
    }

    void setcostScale1(const double in)
    {
        costScale1_ = in;
    }
    void setcostScale2(const double in)
    {
        costScale2_ = in;
    }

    double &getcostScale1()
    {
        return costScale1_;
    }
    double &getcostScale2()
    {
        return costScale2_;
    }

    void updateExtraInfo() final
    {
        extraInfo_["base height"] = gc_[2];
    }

    inline void checkConditionInSimulation()
    {
        RSWARN_IF(isnan(gv_.norm()), "error in simulation!!" << std::endl
                                                             << "action" << pTarget_.transpose() << std::endl
                                                             << "gc_" << gc_.transpose() << std::endl
                                                             << "gv_" << gv_.transpose());

        if (isnan(gc_.norm()))
            badlyConditioned_ = true;
        if (isinf(gc_.norm()))
            badlyConditioned_ = true;
        if (isnan(gv_.norm()))
            badlyConditioned_ = true;
        if (isinf(gv_.norm()))
            badlyConditioned_ = true;
        if (std::abs(gv_.norm()) > 1000)
            badlyConditioned_ = true;
    }

    void updateObservation()
    {
        // 0    3   z axis
        // 3    6   body velocity
        // 9    12  joint position
        // 21   12  joint velocity
        // 33   8   cpg
        // 41  3    command
                
        obDouble_.setZero(obDim_);
        obScaled_.setZero(obDim_);

        // body orientation
        updateRot();
        obDouble_.segment<3>(0) = rot_.e().row(2);
        obDouble_[0] += bodyAngularPositionNoiseScale_ * n(e);
        obDouble_[1] += bodyAngularPositionNoiseScale_ * n(e);
        if (obDouble_[0] >= 1.0)
        {
            obDouble_[0] = 1.0 - 0.1 * bodyAngularPositionNoiseScale_ * (n(e) + 1.0);
        }
        if (obDouble_[1] >= 1.0)
        {
            obDouble_[1] = 1.0 - 0.1 * bodyAngularPositionNoiseScale_ * (n(e) + 1.0);
        }
        // obDouble_[2] += bodyAngularPositionNoiseScale_ * n(e);
        obDouble_[2] = std::sqrt(1 - obDouble_[0] * obDouble_[0] - obDouble_[1] * obDouble_[1]);
        if (isnan(obDouble_[2]))
        {
            obDouble_[2] = 0.0;
        }

        // body velocity
        bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
        obDouble_.segment<3>(3) = bodyLinearVel_;
        obDouble_.segment<3>(3) = Eigen::VectorXd::Constant(3, 0.0); // our model does not rely on body lin vel
        bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);
        obDouble_.segment<3>(6) = bodyAngularVel_;

        if (lowPassFilterCounter_ == 0) {
            prevBodyAngularVelocity_ = bodyAngularVel_;
        } else {
            obDouble_.segment<3>(6) = prevBodyAngularVelocity_;
        }
        if (lowPassFilterCounter_ < 3)
        {
            lowPassFilterCounter_ ++;
        }
        else
        {
            lowPassFilterCounter_ = 0;
        }
        
        // joint position
        obDouble_.segment<12>(9) = gc_.tail<12>();

        // joint velocity
        obDouble_.segment<12>(21) = gv_.tail<12>();

        // cpg
        if (useManualPhase_ && manual_phase_command_)
        {
            manual_phase_.get_status();
        }
        else
        {
            cpg_status_ = cpg_.get_status();
        }
        obDouble_.segment<8>(33) = cpg_status_;
        if (specificGait_ >= 0)
        {
            obDouble_.segment<6>(33).setZero();
        }

        obDouble_[41] = targetLinearVelocity_[0];
        obDouble_[42] = targetLinearVelocity_[1];
        obDouble_[43] = targetAngularVelocity_;

        // obDouble_[61] = gc_[2];

        obScaled_ = (obDouble_ - obMean_).cwiseProduct(obStd_);

        // prevent bad conditions
        RSWARN_IF(isinf(obDouble_.norm()), "observation is inf" << std::endl
                                                        << obDouble_.transpose());
        RSWARN_IF(isnan(obDouble_.norm()), "observation is nan" << std::endl
                                                        << obDouble_.transpose());
        if (isnan(obDouble_.norm()))
        {
            exit(1);
        }
    }

    Eigen::Matrix<double, 4, 1> getContacts()
    {
        // clear previous data
        for (auto& force: footContactForce_)
        {
            force.setZero();
        }

        auto robotContacts = minicheetah_->getContacts();
        for (auto& contact: robotContacts)
        {
            int idx = contact.getlocalBodyIndex();
            if (idx == 3)
            {
                footContactForce_[0] += (contact.getImpulse()->e() / simulation_dt_);
            }
            if (idx == 6)
            {
                footContactForce_[1] += (contact.getImpulse()->e() / simulation_dt_);
            }
            if (idx == 9)
            {
                footContactForce_[2] += (contact.getImpulse()->e() / simulation_dt_);
            }
            if (idx == 12)
            {
                footContactForce_[3] += (contact.getImpulse()->e() / simulation_dt_);
            }
        }

        auto contactState = Eigen::Matrix<double, 4, 1>(-1.0, -1.0, -1.0, -1.0);
        for (int i = 0; i < 4; ++i)
        {
            if (footContactForce_[i].lpNorm<2>() > 12.5) {
                contactState[i] = 1.0;
            }
        }
        return contactState;
    }

    inline void updateState()
    {
        minicheetah_->getState(gc_, gv_);
    }

    inline void noisifyState()
    {
        for (int i = 0; i < gc_.size(); ++i)
        {
            gc_[i] += gcNoiseScale_[i] * noiseFtr * costScale2_ * n(e);
        }
        for (int i = 0; i < gv_.size(); ++i)
        {
            gv_[i] += gvNoiseScale_[i] * noiseFtr * costScale2_ * n(e);
        }
    }

    inline void noisifyDynamics()
    {
        Vec<3> ComOffset;
        Vec<3> Ones = {1.0, 1.0, 1.0};
        Vec<4> quatTmp;
        Mat<3, 3> rotMatTmp;
        double normTmp;

        for (auto &linkName : linkNames_)
        {
            ComOffset = {u(e), u(e), u(e)};
            ComOffset *= bodyComOffsetNoiseScale_ * costScale2_ * noiseFtr;
            ComOffset += Ones;

            minicheetah_->getLink(linkName).setWeight(minicheetahRef_->getLink(linkName).getWeight() * (1 + bodyMassNoiseScale_ * costScale2_ * noiseFtr * u(e)));
            vecvecCwiseMul(minicheetahRef_->getLink(linkName).getComPositionInParentFrame(), ComOffset, ComOffset);
            minicheetah_->getLink(linkName).setComPositionInParentFrame(ComOffset);
        }

        for (int i = 1; i < 13; ++i)
        {
            // RSINFO_IF(visualizable_, frameName)
            ComOffset = {u(e), u(e), u(e)};
            ComOffset *= bodyShapeNoiseScale_ * costScale2_ * noiseFtr;
            ComOffset += Ones;

            vecvecCwiseMul(minicheetahRef_->getJointPos_P()[i], ComOffset, minicheetah_->getJointPos_P()[i]);

            ComOffset = {u(e), u(e), u(e)};
            ComOffset *= bodyShapeNoiseScale_ * costScale2_ * noiseFtr;
            ComOffset += Ones;

            vecvecCwiseMul(minicheetahRef_->getJointAxis_P()[i], ComOffset, minicheetah_->getJointAxis_P()[i]);
            minicheetah_->getJointAxis_P()[i] /= minicheetah_->getJointAxis_P()[i].e().lpNorm<2>();
        }

        double material_prop_0 = frictionMean_ + frictionNoiseScale_ * noiseFtr * u(e);
        double material_prop_1 = u(e) * 0.3 + 0.3;
        double material_prop_2 = u(e) * 5.0 + 5.0;

        materials_.setMaterialPairProp("default", "robot", material_prop_0, material_prop_1, material_prop_2);
        materials_.setMaterialPairProp("robot", "robot", material_prop_0, material_prop_1, material_prop_2);
        materials_.setMaterialPairProp("default", "default", material_prop_0, material_prop_1, material_prop_2);
        world_->updateMaterialProp(materials_);
    }

    inline void updateRot()
    {
        quat_[0] = gc_[3];
        quat_[1] = gc_[4];
        quat_[2] = gc_[5];
        quat_[3] = gc_[6];
        quatToRotMat(quat_, rot_);
    }

    void observe(Eigen::Ref<EigenVec> ob) final
    {
        // convert it to float
        updateState();
        noisifyState();

        updateObservation();
        
        ob = obScaled_.cast<float>();

    }

    bool isTerminalState(float &terminalReward) final
    {
        updateState();
        updateRot();
        terminalReward = float(terminalRewardCoeff_);

        if (gc_[2] < 0.2 or rot_.e().row(2)[2] < 0.8 or gc_[2] > 0.4)
        // if (gc_[2] < 0.1 or rot_.e().row(2)[2] < 0.2)
        {
            return true;
        }

        if (badlyConditioned_)
        {
            badlyConditioned_ = false;
            return true;
        }

        terminalReward = 0.0;
        return false;
    }

    void setSeed(int seed) final
    {
        std::srand(seed);
    }

    void close() final
    {
    }

    void updateCPG()
    {
        if (useManualPhase_ && manual_phase_command_)
        {
            manual_phase_.step();
            cpg_status_ = manual_phase_.get_status();
            cpg_status_dot_ = manual_phase_.get_velocity();
            stance_time_ = stance_time_ * 0.9 + manual_phase_.get_stance_time() * 0.1;
        }
        else
        {
            cpg_.step();
            cpg_status_ = cpg_.get_status();
            cpg_status_dot_ = cpg_.get_velocity();
            stance_time_ = stance_time_ * 0.9 + cpg_.get_stance_time() * 0.1;
        }

        targetLinearVelocity_ = targetLinearVelocity_ * 0.97 + targetLinearVelocityNew_ * 0.03;
        targetAngularVelocity_ = targetAngularVelocity_ * 0.97 + targetAngularVelocityNew_ * 0.03;

        double dx = targetLinearVelocity_[0] * stance_time_ / 2;
        double dy = targetLinearVelocity_[1] * stance_time_ / 2;
        double ddy = - targetAngularVelocity_ * 0.212 * stance_time_ / 2;

        Eigen::VectorXd phase;
        Eigen::VectorXd omega;
        phase.setZero(4);
        phase = cpg_.get_raw_phase() / M_PI;
        omega.setZero(4);
        omega = cpg_.get_omega() / M_PI;
        int hold_leg = cpg_.get_hold_leg();

        if (useManualPhase_ && manual_phase_command_)
        {
            phase = manual_phase_.get_raw_phase() / M_PI;
            omega = manual_phase_.get_omega() / M_PI;
            hold_leg = manual_phase_.get_hold_leg();
        }


        updateState();
        updateRot();

        getEndEffectorRefByPhaseHighOrderPolynomial(EndeffectorRef_, phase, dx, dy, ddy);

        modifyEndeffectorRef(EndeffectorRef_, rot_.e()(2, 0), rot_.e()(2, 1), bodyPosModification_);
        InverseKinematics_v2(EndeffectorRef_, jointRef_);

        // Difference for joint velocity ref
        // getEndEffectorRef(EndeffectorRefTemp_.head<12>(), cpg_status_ + cpg_status_dot_ * 0.01, dx, dy, ddy);
        // getEndEffectorRef(EndeffectorRefTemp_.tail<12>(), cpg_status_ - cpg_status_dot_ * 0.01, dx, dy, ddy);
        // getEndEffectorRefByPhaseGaussian(EndeffectorRefTemp_.head<12>(), phase + omega * 0.01, dx, dy, ddy);
        // getEndEffectorRefByPhaseGaussian(EndeffectorRefTemp_.tail<12>(), phase - omega * 0.01, dx, dy, ddy);
        getEndEffectorRefByPhaseHighOrderPolynomial(EndeffectorRefTemp_.head<12>(), phase + omega * 0.01, dx, dy, ddy);
        getEndEffectorRefByPhaseHighOrderPolynomial(EndeffectorRefTemp_.tail<12>(), phase - omega * 0.01, dx, dy, ddy);
        // getEndEffectorRefByPhaseCosine(EndeffectorRefTemp_.head<12>(), phase + omega * 0.01, dx, dy, ddy);
        // getEndEffectorRefByPhaseCosine(EndeffectorRefTemp_.tail<12>(), phase - omega * 0.01, dx, dy, ddy);
        
        modifyEndeffectorRef(EndeffectorRefTemp_.head<12>(), rot_.e()(2, 0), rot_.e()(2, 1), bodyPosModification_);
        modifyEndeffectorRef(EndeffectorRefTemp_.tail<12>(), rot_.e()(2, 0), rot_.e()(2, 1), bodyPosModification_);
        InverseKinematics_v2(EndeffectorRefTemp_.head<12>(), jointRefTemp_.head<12>());
        InverseKinematics_v2(EndeffectorRefTemp_.tail<12>(), jointRefTemp_.tail<12>());
        jointRefVel_ = (jointRefTemp_.head<12>() - jointRefTemp_.tail<12>()) / 0.02;

        if (hold_leg >= 0)
        {
            jointRef_.segment<3>(3*hold_leg) << 0.0, -1.0, 2.0;
            jointRefVel_.segment<3>(3*hold_leg) << 0.0, 0.0, 0.0;
        }

        if (isnan(jointRef_.norm()) || isinf(jointRef_.norm()) || isnan(jointRefVel_.norm()) || isinf(jointRefVel_.norm()))
        {
            badlyConditioned_ = true;
            
            RSINFO("jointRef_ is nan!" << std::endl
                << "vx: "<< targetLinearVelocity_[0] << std::endl
                << "vy: "<< targetLinearVelocity_[1] << std::endl
                << "yawrate: "<< targetAngularVelocity_ << std::endl
                << "stance_time_: "<< stance_time_ << std::endl
                << "EndeffectorRef_: "<< EndeffectorRef_.transpose() << std::endl
                << "cpg_status_: " << cpg_status_.transpose() << std::endl);
        }
    }

    inline void getEndEffectorRef(Eigen::Ref<Eigen::VectorXd> EndeffectorRef, Eigen::VectorXd phase, double dx, double dy, double ddy)
    {
        for (int i = 0; i < 4; ++i)
        {
            EndeffectorRef[3*i] = 0.5 * (phase[2*i] + 1) * (- dx) + 0.5 * (1 - phase[2*i]) * dx;
            if (i < 2)
            {
                EndeffectorRef[3*i+1] = 0.5 * (phase[2*i] + 1) * (- dy - ddy) + 0.5 * (1 - phase[2*i]) * (dy + ddy);
            }
            else
            {
                EndeffectorRef[3*i+1] = 0.5 * (phase[2*i] + 1) * (- dy + ddy) + 0.5 * (1 - phase[2*i]) * (dy - ddy);
            }

            if (phase[2*i+1] > 0)
            {
                // rise, stance phase
                EndeffectorRef[3*i+2] = - body_height_;
            }
            else
            {
                // decrease, swing phase
                EndeffectorRef[3*i+2] = swing_height_ * std::cos(phase[2*i] * M_PI * 0.5) - body_height_;
            }
        }
    }

    inline void getEndEffectorRefByPhaseCosine(Eigen::Ref<Eigen::VectorXd> EndeffectorRef, Eigen::VectorXd phase, double dx, double dy, double ddy)
    {
        for (int i = 0; i < 4; ++i)
        {
            EndeffectorRef[3*i] = -std::cos(phase[i] * M_PI) * dx;
            if (i < 2)
            {
                EndeffectorRef[3*i+1] = -std::cos(phase[i] * M_PI) * (dy + ddy);
            }
            else
            {
                EndeffectorRef[3*i+1] = -std::cos(phase[i] * M_PI) * (dy - ddy);
            }
            
            if (phase[i] < 0)
            {
                // swing, phase is from 0 to -1
                EndeffectorRef[3*i+2] = - body_height_ - swing_height_ * (std::cos(phase[i] * 2.0 * M_PI) - 1.0) * 0.5;
                // EndeffectorRef[3*i+2] = - body_height_ + swing_height_ * exp(-0.5 * (6.0 * (- 0.5 - phase[i])) * (6.0 * (- 0.5 - phase[i])));
            }
            else
            {
                // stance, phase is from 1 to 0
                EndeffectorRef[3*i+2] = - body_height_;
            }
        }
    }

    inline void getEndEffectorRefByPhaseGaussian(Eigen::Ref<Eigen::VectorXd> EndeffectorRef, Eigen::VectorXd phase, double dx, double dy, double ddy)
    {
        for (int i = 0; i < 4; ++i)
        {
            phase[i] = phase[i] > 1.0 ? phase[i] - 2.0 : phase[i];
            phase[i] = phase[i] < -1.0 ? phase[i] + 2.0 : phase[i];

            if (phase[i] < 0)
            {
                // swing, phase is from 0 to -1
                // EndeffectorRef[3*i+2] = - body_height_ - swing_height_ * (std::cos(phase[i] * 2.0 * M_PI) - 1.0) * 0.5;
                EndeffectorRef[3*i] = ((- 2.0 * (-phase[i]) * (-phase[i]) * (-phase[i]) + 3.0 * (-phase[i]) * (-phase[i])) - 0.5) * 2.0 * dx;
                EndeffectorRef[3*i+1] = ((- 2.0 * (-phase[i]) * (-phase[i]) * (-phase[i]) + 3.0 * (-phase[i]) * (-phase[i])) - 0.5) * 2.0;
                EndeffectorRef[3*i+2] = - body_height_ + swing_height_ * exp(-0.5 * (6.0 * (- 0.5 - phase[i])) * (6.0 * (- 0.5 - phase[i])));
            }
            else
            {
                // stance, phase is from 1 to 0
                EndeffectorRef[3*i] = (- 2.0 * phase[i] * phase[i] * phase[i] + 3.0 * phase[i] * phase[i] - 0.5) * 2.0 * dx;
                EndeffectorRef[3*i+1] = (- 2.0 * phase[i] * phase[i] * phase[i] + 3.0 * phase[i] * phase[i] - 0.5) * 2.0;
                EndeffectorRef[3*i+2] = - body_height_;
            }

            if (i < 2)
            {
                EndeffectorRef[3*i+1] *= (dy + ddy);
            }
            else
            {
                EndeffectorRef[3*i+1] *= (dy - ddy);
            }
        }
    }

    inline void getEndEffectorRefByPhaseHighOrderPolynomial(Eigen::Ref<Eigen::VectorXd> EndeffectorRef, Eigen::VectorXd phase, double dx, double dy, double ddy)
    {
        for (int i = 0; i < 4; ++i)
        {
            phase[i] = phase[i] > 1.0 ? phase[i] - 2.0 : phase[i];
            phase[i] = phase[i] < -1.0 ? phase[i] + 2.0 : phase[i];

            if (phase[i] < 0)
            {
                // swing, phase is from 0 to -1
                // EndeffectorRef[3*i+2] = - body_height_ - swing_height_ * (std::cos(phase[i] * 2.0 * M_PI) - 1.0) * 0.5;
                EndeffectorRef[3*i] = ((6.0 * (-phase[i]) * (-phase[i]) * (-phase[i]) * (-phase[i]) * (-phase[i]) - 15.0 * (-phase[i]) * (-phase[i]) * (-phase[i]) * (-phase[i]) + 10.0 * (-phase[i]) * (-phase[i]) * (-phase[i])) - 0.5) * 2.0 * dx;
                EndeffectorRef[3*i+1] = ((6.0 * (-phase[i]) * (-phase[i]) * (-phase[i]) * (-phase[i]) * (-phase[i]) - 15.0 * (-phase[i]) * (-phase[i]) * (-phase[i]) * (-phase[i]) + 10.0 * (-phase[i]) * (-phase[i]) * (-phase[i])) - 0.5) * 2.0;
                EndeffectorRef[3*i+2] = - body_height_ + swing_height_ * 
                                        (1.0 - 3.0 * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0) + 3.0 * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0)
                                        - (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0) * (-2.0 * phase[i] - 1.0));

                if (i == cpg_.get_hold_leg() && !useManualPhase_)
                {
                    EndeffectorRef[3*i+2] = - body_height_ + swing_height_ * 1.5;
                }
            }
            else
            {
                // stance, phase is from 1 to 0
                EndeffectorRef[3*i] = (6.0 * phase[i] * phase[i] * phase[i] * phase[i] * phase[i] - 15.0 * phase[i] * phase[i] * phase[i] * phase[i] + 10.0 * phase[i] * phase[i] * phase[i] - 0.5) * 2.0 * dx;
                EndeffectorRef[3*i+1] = (6.0 * phase[i] * phase[i] * phase[i] * phase[i] * phase[i] - 15.0 * phase[i] * phase[i] * phase[i] * phase[i] + 10.0 * phase[i] * phase[i] * phase[i] - 0.5) * 2.0;
                EndeffectorRef[3*i+2] = - body_height_;
            }

            if (i < 2)
            {
                EndeffectorRef[3*i+1] *= (dy + ddy);
            }
            else
            {
                EndeffectorRef[3*i+1] *= (dy - ddy);
            }
        }

        if (cpg_.get_hold_leg() >= 0 && !useManualPhase_)
        {
            int shift_leg = 2 * (std::floor(cpg_.get_hold_leg() / 2.0) + 0.5) - cpg_.get_hold_leg();
            if (shift_leg % 2 == 0)
            {
                EndeffectorRef[3*shift_leg+1] += toeLateralOffset_;
            }
            else
            {
                EndeffectorRef[3*shift_leg+1] -= toeLateralOffset_;
            }
        }
    }

    double DeepMimicRewardUpdate()
    {
        // <<<<<<<<<<<<Calculate end effector reward>>>>>>>>>>>>>>>>>>>>>>>>
        bodyPos_ << gc_[0], gc_[1], gc_[2];
        Eigen::Vector3d temp;
        // rotate in horizental plane
        Eigen::Vector3d body_plane_x = rot_.e().row(0);
        Eigen::Matrix3d plane_rot;
        body_plane_x[2] = 0.0;
        body_plane_x.head<2>() = body_plane_x.head<2>() / body_plane_x.head<2>().lpNorm<2>();
        Eigen::Vector3d body_plane_y = Eigen::Vector3d(0, 0, 1).cross(body_plane_x);
        plane_rot.row(0) = body_plane_x;
        plane_rot.row(1) = body_plane_y;
        plane_rot.row(2) << 0.0, 0.0, 1.0;

        double foot_slip = 0.0;
        for (int i = 0; i < 4; i++)
        {
            minicheetah_->getFramePosition(minicheetah_->getFrameIdxByName(ToeName_[i]), TempPositionHolder_);
            temp << TempPositionHolder_[0], TempPositionHolder_[1], TempPositionHolder_[2];
            EndEffector_.segment(3 * i, 3) << rot_.e().transpose() * (temp - bodyPos_);
        }
        EndEffector_ = EndEffector_ - EndEffectorOffset_;

        auto contact_state = getContacts();

        for (int i = 0; i < 4; i++)
        {
            minicheetah_->getFrameVelocity(minicheetah_->getFrameIdxByName(ToeName_[i]), TempPositionHolder_);
            temp << TempPositionHolder_[0], TempPositionHolder_[1], 0.0;
            EndEffectorVel_.segment<3>(3*i) = plane_rot.transpose() * temp;
        }
        for (int i = 0; i < 4; ++i)
        {
            if (cpg_status_[2*i+1] > 0)
            {
                // stance
                foot_slip += 4.0 * EndEffectorVel_.segment<2>(3*i).squaredNorm() * cpg_status_[2*i+1] * cpg_status_[2*i+1];
            }
            else
            {
                // swing
                foot_slip += (footContactForce_[i].lpNorm<2>() / 12.5) * (footContactForce_[i].lpNorm<2>() / 12.5) * cpg_status_[2*i+1] * cpg_status_[2*i+1] * 2.0;

                if (i == cpg_.get_hold_leg() && footContactForce_[i].lpNorm<2>() > 12.5 && cpg_status_[2*i+1] < -0.9 && !useManualPhase_)
                {
                    badlyConditioned_ = true;
                }
            }
        }
        EndEffectorReward = flipFriction_ * EndEffectorRewardFactor_ * exp(-foot_slip) + (1.0 - flipFriction_) * EndEffectorRewardFactor_ * exp(-40.0 * (EndEffector_ - EndeffectorRef_).squaredNorm());

        // ==================================================================
        BodyHeightReward = 80.0 * (gc_[2] - body_height_ - 0.0275) * (gc_[2] - body_height_ - 0.0275);
        BodyHeightReward = BodyHeightRewardFactor_ * exp(-BodyHeightReward);

        BodyPosReward = bodyPosRewardCoeff_ * rot_.e().row(2).head<2>().squaredNorm();
        BodyPosReward = BodyPosRewardFactor_ * exp(-BodyPosReward);

        // ==================================================================
        // <<<<<<<<<<<<<<<<<< Calculate Joint Mimic Reward>>>>>>>>>>>>>>>>>>>
        // InverseKinematics();
        JointReward = 2.0 * (jointRef_ - gc_.tail<12>()).squaredNorm();
        JointReward = JointRewardFactor_ * exp(-JointReward) * 0.25;

        JointVelReward = control_dt_ * (jointRefVel_ - gv_.tail<12>()).squaredNorm() * 2.0;
        JointVelReward = JointRewardFactor_ * exp(-JointVelReward) * 0.75;

        RSWARN_IF(isnan(EndEffectorReward), "EndEffectorReward: " << EndEffectorReward << std::endl);
        RSWARN_IF(isinf(EndEffectorReward), "EndEffectorReward: " << EndEffectorReward << std::endl);
        RSWARN_IF(isnan(JointReward), "JointReward: " << JointReward << std::endl);
        RSWARN_IF(isinf(JointReward), "JointReward: " << JointReward << std::endl);
        RSWARN_IF(isnan(JointVelReward), "JointVelReward: " << JointVelReward << std::endl);
        RSWARN_IF(isinf(JointVelReward), "JointVelReward: " << JointVelReward << std::endl);
        RSWARN_IF(isnan(BodyPosReward), "BodyPosReward: " << BodyPosReward << std::endl);
        RSWARN_IF(isinf(BodyPosReward), "BodyPosReward: " << BodyPosReward << std::endl);
        RSWARN_IF(isnan(BodyHeightReward), "BodyHeightReward: " << EndEffectorReward << std::endl);
        RSWARN_IF(isinf(BodyHeightReward), "BodyHeightReward: " << EndEffectorReward << std::endl);

        return EndEffectorReward + JointReward + JointVelReward + BodyPosReward + BodyHeightReward;
    }

    void getRandomCommand()
    {
        double rate = 0.0;
        if (useManualPhase_)
        {
            rate = 0.4;
        }
        else
        {
            if (specificGait_ < 0 && threeLegGait_)
            {
                rate = - 0.7;
            }
            else
            {
                rate = - 10.0;
            }
        }
        if (u(e) < rate)
        {
            if (!useManualPhase_)
            {
                cpg_.three_leg_mode(std::floor(u(e) * 2.0 + 2.0));

                targetAngularVelocityNew_ = u(e) * 0.5;
                targetLinearVelocityNew_[0] = u(e) * 0.4;
                targetLinearVelocityNew_[1] = std::min((std::sqrt(0.2 * 0.2 - targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.75 * 0.75 * 0.5 * 0.5) - targetAngularVelocityNew_ * 0.212 * 0.75 * 0.5) / 0.5 / 0.75, 0.4) * u(e);
                targetLinearVelocityNew_[2] = 0.0;

                cpg_.change_gait(0);

                if (u(e) < -0.8)
                {
                    targetAngularVelocityNew_ = 0.0;
                    targetLinearVelocityNew_.setZero();
                    
                    cpg_.change_gait(-1);
                }
            }
            else
            {
                targetAngularVelocityNew_ = u(e);
                targetLinearVelocityNew_[0] = u(e);
                targetLinearVelocityNew_[1] = u(e) * 0.5;
                targetLinearVelocityNew_[2] = 0.0;

                manual_phase_command_ = true;

                // manual_phase_.change_gait(std::floor(u(e) * 2.0 + 2.0));
                manual_phase_.change_gait(std::floor(u(e) * 4.5 + 4.5));
            }
        }
        else
        {
            int gait_idx = 0;
            double r = u(e);
            if (r < -0.6)
            {
                gait_idx = 0;
            }
            else
            {
                if (r < 0.0)
                {
                    gait_idx = 1;
                }
                else
                {
                    if (r < 0.6)
                    {
                        gait_idx = 2;
                    }
                    else
                    {
                        gait_idx = 3;
                    }
                }
            }

            // debug: only walk and trot
            // if (r < 0)
            // {
            //     gait_idx = 1;
            // }
            // else
            // {
            //     gait_idx = 2;
            // }

            if (specificGait_ >= 0)
            {
                gait_idx = specificGait_;
            }

            if (gait_idx == 0)
            {
                // while (std::fabs(targetLinearVelocityNew_[0]) < 0.1 && std::fabs(targetLinearVelocityNew_[1]) < 0.1 && std::fabs(targetAngularVelocityNew_) < 0.1)
                // {
                //     v_norm = 0.3 * std::fabs(u(e));
                //     theta = u(e) * M_PI;
                //     targetLinearVelocityNew_[0] = v_norm * std::cos(theta);
                //     targetLinearVelocityNew_[1] = v_norm * std::sin(theta);
                //     targetLinearVelocityNew_[2] = 0.0;
                //     targetAngularVelocityNew_ = u(e) * 0.3;
                // }

                // targetLinearVelocityNew_[0] = std::fabs(targetLinearVelocityNew_[0]) < 0.1 ? 0.0 : targetLinearVelocityNew_[0];
                // targetLinearVelocityNew_[1] = std::fabs(targetLinearVelocityNew_[1]) < 0.1 ? 0.0 : targetLinearVelocityNew_[1];
                // targetAngularVelocityNew_ = std::fabs(targetAngularVelocityNew_) < 0.1 ? 0.0 : targetAngularVelocityNew_;            

                targetAngularVelocityNew_ = u(e) * 0.5;
                targetLinearVelocityNew_[0] = u(e) * 0.4;
                targetLinearVelocityNew_[1] = std::min((std::sqrt(0.2 * 0.2 - targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.75 * 0.75 * 0.5 * 0.5) - targetAngularVelocityNew_ * 0.212 * 0.75 * 0.5) / 0.5 / 0.75, 0.4) * u(e);
                targetLinearVelocityNew_[2] = 0.0;
            }

            if (gait_idx == 1 || gait_idx == 2)
            {
                // targetLinearVelocityNew_[0] = u(e);
                // targetLinearVelocityNew_[1] = u(e) * 0.5;
                // targetLinearVelocityNew_[2] = 0.0;
                targetAngularVelocityNew_ = u(e);
                // targetAngularVelocityNew_ = std::fabs(targetAngularVelocityNew_) < 0.3 ? 0.0 : targetAngularVelocityNew_;

                targetLinearVelocityNew_[0] = u(e) * 1.2;
                // targetLinearVelocityNew_[0] = std::fabs(targetLinearVelocityNew_[0]) < 0.1 ? 0.0 : targetLinearVelocityNew_[0];

                targetLinearVelocityNew_[1] = std::min((std::sqrt(0.2 * 0.2 - targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.25 * 0.25 * 0.5 * 0.5) - targetAngularVelocityNew_ * 0.212 * 0.25 * 0.5) / 0.5 / 0.25, 0.5) * u(e);
                // targetLinearVelocityNew_[1] = std::fabs(targetLinearVelocityNew_[1]) < 0.1 ? 0.0 : targetLinearVelocityNew_[1];

                targetLinearVelocityNew_[2] = 0.0;
            }

            if (gait_idx == 3)
            {
                targetLinearVelocityNew_[0] = u(e) * 0.5;
                if (u(e) > 0)
                {
                    targetLinearVelocityNew_[0] += 1.0;
                }
                else
                {
                    targetLinearVelocityNew_[0] -= 1.0;
                }

                targetLinearVelocityNew_[1] = u(e) * 0.5;
                targetLinearVelocityNew_[2] = 0.0;
                targetAngularVelocityNew_ = u(e) * 0.5;

                // targetAngularVelocityNew_ = std::fabs(targetAngularVelocityNew_) < 0.3 ? 0.0 : targetAngularVelocityNew_;
                // targetLinearVelocityNew_[1] = std::fabs(targetLinearVelocityNew_[1]) < 0.2 ? 0.0 : targetLinearVelocityNew_[1];
            }

            if (u(e) > 0.9)
            {
                targetLinearVelocityNew_[0] = 0.0;
                targetLinearVelocityNew_[1] = 0.0;
                targetLinearVelocityNew_[2] = 0.0;
                targetAngularVelocityNew_ = 0.0;

                gait_idx = -1;
            }

            if (walkOnly_)
            {
                gait_idx = 0;
            }
            if (gallopOnly_)
            {
                gait_idx = 3;
            }

            cpg_.change_gait(gait_idx);
        }
    }

    void changeGait()
    {
        int previous_gait_idx = cpg_.get_gait_index();
        int hold_leg = cpg_.get_hold_leg();
        int gait_idx = 1;
        double rate = -0.4;
        if (threeLegGait_)
        {
            rate = -0.4;
        }
        else
        {
            rate = -10.0;
        }

        // if (targetLinearVelocityNew_.head<2>().lpNorm<2>() < 0.3 && fabs(targetAngularVelocityNew_) < 0.3)
        if (previous_gait_idx != -1)
        {
            if ((targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.140625 + (std::fabs(targetLinearVelocityNew_[1]) + std::fabs(targetAngularVelocityNew_) * 0.212) * (std::fabs(targetLinearVelocityNew_[1]) + std::fabs(targetAngularVelocityNew_) * 0.212) * 0.140625) < 0.2 * 0.2)
            {
                gait_idx = std::floor((u(e) * 0.5 + 0.5) * 2.5 + 0.5);
                while (gait_idx == previous_gait_idx)
                {
                    gait_idx = std::floor((u(e) * 0.5 + 0.5) * 2.5 + 0.5);
                }

                if (!useManualPhase_)
                {
                    if (hold_leg < 0)
                    {
                        if (u(e) < rate)
                        {
                            cpg_.three_leg_mode(std::floor(u(e) * 2.0 + 2.0));
                            gait_idx = 0;
                        }
                    }
                    else
                    {
                        if (u(e) < 0.0)
                        {
                            cpg_.three_leg_mode(-1);
                        }
                    }
                }
            }
            else
            {
                // if (std::fabs(targetLinearVelocityNew_[0]) > 0.5 && std::fabs(targetLinearVelocityNew_[1]) < 0.5)
                if ((targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.015625 + (std::fabs(targetLinearVelocityNew_[1]) + std::fabs(targetAngularVelocityNew_) * 0.212) * (std::fabs(targetLinearVelocityNew_[1]) + std::fabs(targetAngularVelocityNew_) * 0.212) * 0.015625) < 0.2 * 0.2)
                {
                    gait_idx = std::floor((u(e) * 0.5 + 0.5) * 2.5 + 1.0);
                    while (gait_idx == previous_gait_idx)
                    {
                        gait_idx = std::floor((u(e) * 0.5 + 0.5) * 2.5 + 1.0);
                    }
                }
                else
                {
                    // gait_idx = std::floor((u(e) * 0.5 + 0.5) * 2.0 + 1.0);
                    // while (gait_idx == previous_gait_idx)
                    // {
                    //     gait_idx = std::floor((u(e) * 0.5 + 0.5) * 2.0 + 1.0);
                    // }
                    gait_idx = 3;
                }
            }
        }
        else
        {
            getRandomCommand();
        }

        if (u(e) > 0.9)
        {
            targetLinearVelocityNew_[0] = 0.0;
            targetLinearVelocityNew_[1] = 0.0;
            targetLinearVelocityNew_[2] = 0.0;
            targetAngularVelocityNew_ = 0.0;

            gait_idx = -1;
        }
        // if (std::fabs(targetLinearVelocityNew_[0]) < 0.05 && std::fabs(targetLinearVelocityNew_[1]) < 0.05 && std::fabs(targetAngularVelocityNew_) < 0.15)
        // {
        //     gait_idx = -1;
        // }

        // if (gait_idx == 0)
        // {
        //     swing_height_ = default_swing_height_ * 1.5;
        // }
        // else
        // {
        //     swing_height_ = default_swing_height_;
        // }

        // debug: only walk and trot
        // if (previous_gait_idx != -1)
        // {
        //     gait_idx = 3 - previous_gait_idx;
        //     // gait_idx = 1;
        // }
        // else
        // {
        //     gait_idx = u(e) > 0 ? 1 : 2;
        //     // gait_idx = 1;
        // }

        if (useManualPhase_ && manual_phase_command_)
        {
            gait_idx = std::floor(u(e) * 4.5 + 4.5);
            manual_phase_.change_gait(gait_idx);
        }
        else
        {
            cpg_.change_gait(gait_idx);
        }
    }

    void changeSpeed()
    {
        // targetLinearVelocityNew_[0] = u(e);
        // targetLinearVelocityNew_[1] = u(e) * 0.5;
        // targetLinearVelocityNew_[2] = 0.0;
        // targetAngularVelocityNew_ = u(e);

        // targetLinearVelocityNew_[0] = std::fabs(targetLinearVelocityNew_[0]) > 0.1 ? targetLinearVelocityNew_[0] : 0.0;
        // targetLinearVelocityNew_[1] = std::fabs(targetLinearVelocityNew_[1]) > 0.1 ? targetLinearVelocityNew_[1] : 0.0;
        // targetAngularVelocityNew_ = std::fabs(targetAngularVelocityNew_) > 0.3 ? targetAngularVelocityNew_ : 0.0;

        // targetAngularVelocityNew_ = u(e);
        // targetAngularVelocityNew_ = std::fabs(targetAngularVelocityNew_) < 0.3 ? 0.0 : targetAngularVelocityNew_;

        // targetLinearVelocityNew_[0] = u(e) * 1.2;
        // targetLinearVelocityNew_[0] = std::fabs(targetLinearVelocityNew_[0]) < 0.1 ? 0.0 : targetLinearVelocityNew_[0];

        // targetLinearVelocityNew_[1] = std::min((std::sqrt(0.2 * 0.2 - targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.25 * 0.25 * 0.5 * 0.5) - targetAngularVelocityNew_ * 0.212 * 0.25 * 0.5) / 0.5 / 0.25, 0.5) * u(e);
        // targetLinearVelocityNew_[1] = std::fabs(targetLinearVelocityNew_[1]) < 0.1 ? 0.0 : targetLinearVelocityNew_[1];

        // targetLinearVelocityNew_[2] = 0.0;

        if (u(e) > 0.9)
        {
            targetLinearVelocityNew_.setZero(3);
            targetAngularVelocityNew_ = 0.0;
            
            cpg_.change_gait(-1);
        }

        // check previous gait
        bool check_previous_gait = false;
        if (cpg_.get_gait_index() == -1 && !check_previous_gait && !manual_phase_command_)
        {
            if (u(e) > -0.5)
            {
                changeGait();
            }
            check_previous_gait = true;
        }

        if ((cpg_.get_gait_index() == 0 || cpg_.get_hold_leg() >= 0) && !check_previous_gait && !manual_phase_command_)
        {
            // if (targetLinearVelocityNew_.head<2>().lpNorm<2>() >= 0.3)
            // {
            //     targetLinearVelocityNew_[0] *= 0.3;
            //     targetLinearVelocityNew_[1] *= 0.3;
            //     targetAngularVelocityNew_ *= 0.3;

            // }

            targetAngularVelocityNew_ = u(e) * 0.3;
            targetLinearVelocityNew_[0] = u(e) * 0.4;
            targetLinearVelocityNew_[1] = std::min((std::sqrt(0.2 * 0.2 - targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.75 * 0.75 * 0.5 * 0.5) - targetAngularVelocityNew_ * 0.212 * 0.75 * 0.5) / 0.5 / 0.75, 0.4) * u(e);
            targetLinearVelocityNew_[2] = 0.0;
            
            check_previous_gait = true;
        }

        if ((cpg_.get_gait_index() == 1 || cpg_.get_gait_index() == 2 || manual_phase_command_) && !check_previous_gait)
        {
            targetAngularVelocityNew_ = u(e);
            targetLinearVelocityNew_[0] = u(e) * 1.2;
            targetLinearVelocityNew_[1] = std::min((std::sqrt(0.2 * 0.2 - targetLinearVelocityNew_[0] * targetLinearVelocityNew_[0] * 0.25 * 0.25 * 0.5 * 0.5) - targetAngularVelocityNew_ * 0.212 * 0.25 * 0.5) / 0.5 / 0.25, 0.5) * u(e);
            targetLinearVelocityNew_[2] = 0.0;
            
            check_previous_gait = true;
        }

        if (cpg_.get_gait_index() == 3 && !check_previous_gait && !manual_phase_command_)
        {
            targetLinearVelocityNew_[0] = u(e) * 0.5;
            if (u(e) > 0)
            {
                targetLinearVelocityNew_[0] += 1.0;
            }
            else
            {
                targetLinearVelocityNew_[0] -= 1.0;
            }

            targetLinearVelocityNew_[1] = u(e) * 0.5;
            targetLinearVelocityNew_[2] = 0.0;
            targetAngularVelocityNew_ = u(e) * 0.5;

            check_previous_gait = true;
        }

        // check new gait

        // if (std::fabs(targetLinearVelocityNew_[0]) < 0.05 && std::fabs(targetLinearVelocityNew_[1]) < 0.05 && std::fabs(targetAngularVelocityNew_) < 0.15)
        // {
        //     cpg_.change_gait(-1);
        // }
    }

    void curriculumUpdate()
    {
        cricculumFactorCmd_ = cricculumFactorCmd_ * cricculumIncreaseFactor_ + cricculumFactorCmdFinal_ * (1 - cricculumIncreaseFactor_);
        cricculumFactorMimic_ = cricculumFactorMimic_ * cricculumIncreaseFactor_ + cricculumFactorMimicFinal_ * (1 - cricculumIncreaseFactor_);
        costScale1_ = std::pow(costScale1_, costScaleIncreaseRate_);
    }

    void enable_repeat_mode()
    {
        repeat_mode_ = true;
    }

    void disable_repeat_mode()
    {
        repeat_mode_ = false;
    }

    void enable_joint_record()
    {
        record_joint_velocity_ = true;
    }

    void disable_joint_record()
    {
        record_joint_velocity_ = false;
    }

    void getExtraDynamicsInfo(Eigen::Ref<EigenVec> info)
    {
        updateState();

        extraRef_.head<3>() = rot_.e().transpose() * gv_.segment<3>(0);
        extraRef_[3] = gc_[2];

        extraRef_.segment<12>(4) = EndeffectorRef_;
        extraRef_.segment<12>(16) = jointRef_;
        extraRef_.segment<12>(28) = jointRefVel_;

        extraRef_.segment<12>(40) = minicheetah_->getGeneralizedForce().e().tail<12>();

        info = extraRef_.cast<float>();
    }

    void getContactInfo(Eigen::Ref<EigenVec> cont)
    {
        auto contactState = getContacts();
        cont = contactState.cast<float>();
    }

    void setGait(int gait_idx)
    {
        cpg_.change_gait(gait_idx);
    }

private:
    // simulation related
    ArticulatedSystem *minicheetah_;
    ArticulatedSystem *minicheetahRef_;
    std::vector<GraphicObject> *minicheetahVisual_;
    bool visualizable_ = false;
    std::normal_distribution<double> distribution_;
    Eigen::VectorXd jointPGain_, jointDGain_;
    Eigen::VectorXd jointUpperLimits_, jointLowerLimits_;

    Eigen::Matrix<double, 12, 1> jointNominalConfig_;

    MaterialManager materials_;

    // int gcDim_, gvDim_, nJoints_;
    int gcDim_, gvDim_;
    Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
    Eigen::VectorXd pTargetPast_;
    Eigen::VectorXd random_init;

    int visualizationCounter_ = 0;
    double desired_fps_ = 60.;

    // structure
    std::vector<std::string> linkNames_ = {"body",
                                            "abduct_fr", "thigh_fr", "shank_fr",
                                            "abduct_fl", "thigh_fl", "shank_fl",
                                            "abduct_hr", "thigh_hr", "shank_hr",
                                            "abduct_hl", "thigh_hl", "shank_hl"};

    std::vector<std::string> frameNames_ = {"torso_to_abduct_fr_j",
                                            "torso_to_abduct_fl_j",
                                            "torso_to_abduct_hr_j",
                                            "torso_to_abduct_hl_j",
                                            "abduct_fr_to_thigh_fr_j",
                                            "abduct_fl_to_thigh_fl_j",
                                            "abduct_hr_to_thigh_hr_j",
                                            "abduct_hl_to_thigh_hl_j",
                                            "thigh_fr_to_knee_fr_j",
                                            "thigh_fl_to_knee_fl_j",
                                            "thigh_hr_to_knee_hr_j",
                                            "thigh_hl_to_knee_hl_j",
                                            "toe_fr_joint",
                                            "toe_fl_joint",
                                            "toe_hr_joint",
                                            "toe_hl_joint"};

    // model
    // int model_idx = 0;
    
    // terrian related
    std::vector<double> heights_;
    double groundAmplitude_;

    // command related
    // Eigen::Vector4d command_;
    double switchCommandTimer_;
    Eigen::Vector3d targetLinearVelocity_;
    Eigen::Vector3d targetLinearVelocityNew_;
    double targetAngularVelocity_;
    double targetAngularVelocityNew_;
    bool autoChangeGait_;

    // foot and contact related
    std::vector<CollisionDefinition> footCollision_;
    std::array<Eigen::Matrix<double, 3, 1>, 4> footContactForce_;

    // action and obervation related
    Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
    Eigen::VectorXd obDouble_, obScaled_;
    Eigen::VectorXd jointPositionHistory_;
    Eigen::VectorXd jointVelocityHistory_;
    Eigen::VectorXd smoothContactState_;
    // Eigen::VectorXd commandHistory_;
    double actionStdScalar_;
    double mixParam_;
    double contactMixParam_;
    // int historyLength_ = 13;

    // noise related
    std::default_random_engine e;
    std::uniform_real_distribution<double> u{-1.0, 1.0};
    std::normal_distribution<double> n{0.0, 1.0};

    Eigen::VectorXd gcNoiseScale_, gvNoiseScale_;
    double bodyPlaneLinearPositionNoiseScale_, bodyHeightNoiseScale_, bodyAngularPositionNoiseScale_, JointPositionNoiseScale_;
    double bodyLinearVelocityNoiseScale_, bodyAngularVelocityNoiseScale_, JointVelocityNoiseScale_;
    double noiseFtr;

    double bodyComOffsetNoiseScale_;
    double bodyMassNoiseScale_;
    double bodyShapeNoiseScale_;
    double frictionNoiseScale_;
    double frictionMean_;

    double bodyForceTimer_;
    double bodyForceVisTimer_;
    double bodyForceAmplitude_;
    double bodyForceExtNorm_;
    Vec<3> bodyForceExt_;
    Vec<3> bodyForceExtVis_;
    Vec<3> bodyForcePos_;
    Vec<3> bodyForcePosGlobal_;
    Mat<3, 3> rotForceLocal_;
    Mat<3, 3> rotForceGlobal_;
    Eigen::Vector3d prevBodyAngularVelocity_;
    int lowPassFilterCounter_ = 0;

    VisualObject* arrow_;

    // cost related
    Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
    bool badlyConditioned_;
    Eigen::Vector3d desiredLinearSpeed;
    Eigen::VectorXd torque_;
    Eigen::Matrix<double, 12, 1> torque0_;
    Eigen::VectorXd torqueFull_;

    double terminalRewardCoeff_;

    // mimic related
    // gait
    // double current_time_ = 0.0;  // current time
    // double intro_time_;
    // bool isIntroState_;
    Eigen::VectorXd jointRef_;   // store the reference joint angle at current time, length should be 12
    Eigen::VectorXd EndeffectorRef_;
    Eigen::VectorXd jointRefTemp_;
    Eigen::VectorXd EndeffectorRefTemp_;
    Eigen::Matrix<double, 12, 1> EndeffectorRefPos_;
    Eigen::Matrix<double, 12, 1> EndeffectorRefVel_;
    Eigen::Matrix<double, 12, 1> EndeffectorVelScale_;
    Eigen::Matrix<double, 12, 1> EndEffectorVel_;
    Eigen::VectorXd jointRefVel_;   
    Eigen::VectorXd EndeffectorRefFuture_;
    double body_height_ = 0.3;
    double swing_height_ = 0.0;
    double default_swing_height_ = 0.0;
    double stance_time_ = 0.25;

    // CPG
    CPG<double> cpg_;
    Eigen::VectorXd cpg_status_;
    Eigen::VectorXd cpg_status_dot_;
    // bool cpg_init_flag_;
    bool cpg_transition_status_;
    int specificGait_ = -1;

    // Manual Phase
    ManualPhase<double> manual_phase_;

    // input filter
    double joint_filter_freq, joint_filter_alpha;

    // special gaits
    bool walkOnly_ = false;
    bool gallopOnly_ = false;
    bool useManualPhase_ = false;
    bool manual_phase_command_ = false;
    bool threeLegGait_ = false;

    // mimic cost calculation
    Vec<3> TempPositionHolder_;                                                         // tempory position holder to get toe position
    Eigen::VectorXd EndEffector_;                                                               // storage minicheetah storage four toe position relative to body frame
    Eigen::Matrix<double, 12, 1> EndEffectorOffset_;                                                         // offset between hip coordinates and body center
    std::string ToeName_[4] = {"toe_fr_joint", "toe_fl_joint", "toe_hr_joint", "toe_hl_joint"}; // storage minicheetah toe frame name
    Eigen::Vector3d bodyPos_;
    double DeepMimicReward_ = 0;

    double EndEffectorReward = 0;
    double DirectionKeepReward = 0;
    double JointReward = 0;
    double JointVelReward = 0;
    double VelocityReward = 0;
    double BodyPosReward = 0;
    double BodyHeightReward = 0;

    double bodyPosModification_ = 1.0;

    double toeLateralOffset_ = 0.06;

    // cricculum learning
    double costScale1_;
    double costScale2_;
    double costScaleIncreaseRate_;
    double cricculumFactorCmd_;
    double cricculumFactorMimic_;
    double cricculumFactorCmdFinal_;
    double cricculumFactorMimicFinal_;
    double cricculumIncreaseFactor_;

    // weights
    double torqueCostScale_;
    // double linVelCostScale_;
    double cmdCostCoeff_;
    double bodyPosRewardCoeff_;

    double EndEffectorRewardFactor_;
    double JointRewardFactor_;
    double BodyPosRewardFactor_;
    double BodyHeightRewardFactor_;
    double cmdRewardFactor_;
    double flipFriction_ = 1.0;

    // normalization for cost calculation
    // double torqueStd_;
    Eigen::VectorXd torqueStd_;
    double jointVelStd_;
    double jointAngStd_;
    double bodyLinVelStd_;
    double bodyAngVelStd_;
    double bodyOrientationStd_;

    // body orientation
    Vec<4> quat_;
    Mat<3, 3> rot_;

    // extra dynamics info
    Eigen::VectorXd extraRef_;

    // test options
    bool repeat_mode_ = false;
    bool record_joint_velocity_ = false;
};

} // namespace raisim