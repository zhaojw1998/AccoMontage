# AccoMontage

## Introduction
AccoMontage is a novel accompaniment arrangement system. It introduces a novel hybrid pathway, in which rule-based optimization and deep learning are both leveraged to complement each other for high-quality generation. Our paper [*AccoMontage: Accompaniment Arrangement via Phrase Selection and Style Transfer*](https://arxiv.org/abs/2108.11213) is accepted by [ISMIR 2021](https://ismir2021.ismir.net/). This repository stores codes and demos of our work.

## Data and Weights Download
Data and weights required to reproduce our work can be downloaded [here](https://drive.google.com/drive/folders/14sR11NR7jDPMLtCAYbuK5KwLdc7jSKZv?usp=sharing). The files should be put to `./data files`.

For the sake of fast accompaniment arrangement, our model does not run the deep learning inference evrey time, but loads pre-inferenced transition matrix. This pre-inferenced file currently supports transition among 4, 6, and 8-bar phrases. `./util_tools/edge_weights_inference.py` can be used to compute transition matrix for other situations.

## Run
To inference with AccoMontage and generate for your own music, run `AccoMontage_inference.py`. Line 197~207 should be configured by yourself.

## Demos
Generated demos are listed in `./demo generate upload`. The original query lead sheets are listed in `demo lead sheets`. AccoMontage is applied as one of the backbones to rearrange the university song of East China Normal University (ECNU). Demos are included.

## New Feature: Reference Spotlight
This new feature allows you to specify which reference song(s) (from [POP909](https://github.com/music-x-lab/POP909-Dataset)) you would like to use and apply its/their texture to the query lead sheet.

## Acknowledgement
Thanks to Prof. Gus Xia, Yixiao Zhang, Liwei Lin, Junyan Jiang, Ziyu Wang, and Shuqi Dai for your generous help to this work. Thanks to all my friends at NYUSH Music X Lab for your encouragement and companion. The following repositories are referred to by this work:

- https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement
- https://github.com/ZZWaang/PianoTree-VAE 
- https://github.com/Dsqvival/hierarchical-structure-analysis
- https://github.com/buggyyang/Deep-Music-Analogy-Demos

## Contact
For inquries about our work, feel free to contact me at jz4807@nyu.edu.

Jingwei Zhao (Ph.D. Student in Data Science, NUS Graduate School)

Aug. 06, 2021