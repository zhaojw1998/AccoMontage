# AccoMontage

## Introduction
AccoMontage is an accompaniment arrangement system. It introduces a novel hybrid pathway, in which rule-based optimization and deep learning are both leveraged to complement each other for high-quality generation. Our paper [*AccoMontage: Accompaniment Arrangement via Phrase Selection and Style Transfer*](https://arxiv.org/abs/2108.11213) is accepted by [ISMIR 2021](https://ismir2021.ismir.net/). This repository stores codes and demos of our work.

## Data and Weights Download
Data and weights required to reproduce our work can be downloaded [here](https://drive.google.com/drive/folders/14sR11NR7jDPMLtCAYbuK5KwLdc7jSKZv?usp=sharing). The files should be put to `./data files`.

For the sake of fast accompaniment arrangement, our model does not run the deep learning inference evrey time, but loads pre-inferenced transition matrix. This pre-inferenced file currently supports transition among 4, 6, and 8-bar phrases. `./util_tools/edge_weights_inference.py` can be used to compute transition matrix for other situations.

## Run
To inference with AccoMontage and generate for your own music, run `AccoMontage_inference.py`. Line 44~57 should be configured by yourself. Concretely, your should specify:

-   Required:
    -   `SONG_NAME` & `SONG_ROOT` -- directory to a MIDI lead sheet file. This MIDI file should consists of two tracks, each containing melody (monophonic) and chord (polyphonic). Now complex chords (9th, 11th, and more) is supported.
    -   `SEGMENTATION` -- phrase annotation (string) of the MIDI file. For example, for an AABB song with 8 bars for each phrase, the annotation should be in the format `'A8A8B8B8\n'`. Note that by default we only support the transition among 4-bar, 6-bar, and 8-bar phrases
    -   `NOTE_SHIFT` -- The number of upbeats in the  pickup bar (can be float). If no upbeat, specify 0.
-   Optional:
    -   `SPOTLIGHT` -- a list of names of your prefered reference songs. See all 1001 supported reference songs (Chinese POP) at `./data files/POP909 hierarchical-structure-analysis/index.xlsx`.
    -   `PREFILTER` -- a tuple (a, b) controlling rhythmic patters. a, b can be integers in [0, 4], each controlling horrizontal rhythmic density and vertical voice number. Ther higher number, the denser rhythms.


## Demos
Generated demos are listed in `./demo/demo generate upload`. The original query lead sheets are listed in `./demo/demo lead sheets`. AccoMontage is applied as the backbones to rearrange the Alma Mater Music of East China Normal University (ECNU). Performance demos can be accessed [here](https://zhaojw1998.github.io/accomontage_demo).

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