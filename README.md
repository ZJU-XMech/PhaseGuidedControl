# PhaseGuidedControl

> The current version is developed based on the old version of RaiSim series, and possibly requires further modification. It will be upgraded to the current version of RaiSim soon.

This repository contains the RL environment and serveral other necessary code for the paper: 

"[Learning Free Gait Transition for Quadruped Robots vis Phase-Guided Controller](https://ieeexplore.ieee.org/document/9656601)" (DOI: 10.1109/LRA.2021.3136645)

```
@ARTICLE{9656601,  
author={Shao, Yecheng and Jin, Yongbin and Liu, Xianwei and He, Weiyan and Wang, Hongtao and Yang, Wei},  
journal={IEEE Robotics and Automation Letters},   
title={Learning Free Gait Transition for Quadruped Robots via Phase-Guided Controller},   
year={2021},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/LRA.2021.3136645}
}
```

## Dependencies

This environment uses RaiSim as the physics engine, which requires an activation key. Since the current version is developed based on the old version of raisim, raisimOgre is used for visualization and raisimGym is used for training.

- [RaiSim](https://github.com/raisimTech/raisimLib)

- [raisimOgre](https://github.com/raisimTech/raisimOgre)

- [raisimGym](https://github.com/ZJU-XMech/raisimGym) (use the `cpg` branch)

For the algorithms, we use tensorflow and stable-baselines.

- [TensorFlow](https://www.tensorflow.org/) (<=1.15)

- [Stable Baselines](https://stable-baselines.readthedocs.io/) (2.80.0)

## Compile

Please follow the guide of raisimGym:

```bash
# navigate to the raisimGym folder
cd /WHERE/YOUR/RAISIMGYM/REPO/IS
# first switch to the cpg branch
git checkout cpg
# compile the environment
python3 setup.py install --CMAKE_PREFIX_PATH $LOCAL_BUILD --env /WHERE/YOUR/CUSTOM/ENVIRONMENT/IS
```

## Train

The configuration file `cfg/default_cfg.yaml` contains some parameters for the training envrionment. Here list some important parameters.

- `useManualPhase`: (bool) determines whether to use manually-designed gaits during training

- `threeLegGait`: (bool) determines whether to use three-legged gaits during training

- `specificGait`: (int) an option to use only one kind of CPG gait during training, `-1` for all four gaits

- `noiseFtr`: (double) overall amplitude of all the noises, `0` to disable all the noises.

Run the following command to train the model.

```
python3 scripts/commanded_locomotion_lstm.py
```

## Test

We use the [Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software) for test and hardware depolyment. Copy the files in `controller` folder to the `Cheetah-Software/user` and modify the `CMakeLists.txt` to compile the controller. Our C++ implemented LSTM is also included in those files. Use `scripts/params_to_csv.py` to generate csv files for the parameters of the LSTM, and save the files to `Cheetah-Software/config` with proper names.
