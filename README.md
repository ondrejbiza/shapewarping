# One-shot Imitation Learning via Interation Warping

This repo is a refactored version of the code for my paper Interaction Warping published at CoRL'23.

Website: https://shapewarping.github.io/

Paper: https://arxiv.org/abs/2306.12392

Original code (includes real-world UR5 code): https://github.com/ondrejbiza/fewshot

## Install

1. Optional: create a virtual environment.
```
python3 -m venv venv
source venv/bin/activate
```

2. Install `shapewarping`.
```
pip install -e .
```

3. Install `airobot` and `relational-ndf`.
```
git submodule init
git submodule update
pip install cython wheel
pip install git+https://github.com/ondrejbiza/airobot.git@panda-2f140#egg=airobot
pip install -e relational_ndf
```

4. Build and install `v-hacd`.
```
cd v-hacd/install
python run.py --cmake
cd ../build/linux
make
sudo make install
```

## Download experiment data
```
cd relational_ndf
source ./rndf_env.sh
./scripts/download_obj_mesh_data.bash
./scripts/download_relational_demonstrations.bash
# For a baseline.
./scripts/download_demo_weights.bash
```

## Usage

### Learn Warps

Mugs:
```
python shapewarping/learn_warp.py data/shapenet/mug_centered_obj_normalized/ data/mug_warp.pkl --rot-x 1.5708 --set-canon-index 5
```

Bottles:
```
python shapewarping/learn_warp.py data/shapenet/mug_centered_obj_normalized/ data/mug_warp.pkl --rot-x 1.5708 --set-canon-index 9
```

Bowls:
```
python shapewarping/learn_warp.py data/shapenet/mug_centered_obj_normalized/ data/mug_warp.pkl --rot-x 1.5708 --set-canon-index 0
```

Mug racks:
```
python shapewarping/learn_warp.py data/syn_racks_easy/ data/mug_warp.pkl --rot-x 1.5708 --num-surface-samples 2000 --set-canon-index 0
```

## Reproduce results in paper

See `shell_scripts` for reproducing Interaction Warping (ours) and R-NDF (baseline) results with 1, 5, and 10 demos.

For example:
```
python -m scripts.run_warp --parent_class mug --child_class bowl \
  --exp bowl_on_mug_upright_pose_new \
  --parent_model_path ndf_vnn/rndf_weights/ndf_mug.pth \
  --child_model_path ndf_vnn/rndf_weights/ndf_bowl.pth \
  --is_parent_shapenet_obj --is_child_shapenet_obj \
  --rel_demo_exp release_demos/bowl_on_mug_relation --pybullet_server \
  --opt_iterations 650 \
  --parent_load_pose_type random_upright --child_load_pose_type random_upright &> outputs/bowl_on_mug_upright.txt
```

## Code structure

## Acknowledgements

* Simulation and benchmark: [Relational Neural Descriptor Fields](https://github.com/anthonysimeonov/relational_ndf).
* Parts of shape warping code: [Shape-based Skill Transfer](https://lis.csail.mit.edu/wp-content/uploads/2021/05/thompson_icra_2021_compressed.pdf).
* We re-distrubute an older version of [v-hacd](https://github.com/kmammou/v-hacd) in `/v-hacd`.

## Troubleshooting

1. `pip install cycpd` doesn't work on Mac. But, the following works:
```
git clone https://github.com/gattia/cycpd
cd cycpd
pip install -e .
```

## Citation

```
@inproceedings{biza23oneshot,
  author       = {Ondrej Biza and
                  Skye Thompson and
                  Kishore Reddy Pagidi and
                  Abhinav Kumar and
                  Elise van der Pol and
                  Robin Walters and
                  Thomas Kipf and
                  Jan{-}Willem van de Meent and
                  Lawson L. S. Wong and
                  Robert Platt},
  title        = {One-shot Imitation Learning via Interaction Warping},
  booktitle    = {CoRL},
  year         = {2023}
}
```
