# One-shot Imitation Learning via Interation Warping

TODO:
* Make sure I have the right dependencies.
* Provide a docker image for R-NDF.

This repo is a refactored version of the code for my paper Interaction Warping published at CoRL'23.

Website: https://shapewarping.github.io/

Paper: https://arxiv.org/abs/2306.12392

Original code (includes real-world UR5 code): https://github.com/ondrejbiza/fewshot

## Install

## Usage

### Learn Warps

## Reproduce results in paper

(Original code: TODO)

## Code structure

## Acknowledgements

* Simulation and benchmark: [Relational Neural Descriptor Fields](https://github.com/anthonysimeonov/relational_ndf).
* Parts of shape warping code: [Shape-based Skill Transfer](https://lis.csail.mit.edu/wp-content/uploads/2021/05/thompson_icra_2021_compressed.pdf).
* We re-distrubute an older version of [v-hacd] in `/v-hacd`.

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
