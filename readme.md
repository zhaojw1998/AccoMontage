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


## Reference
If you find this code useful, please consider citing:

<div class="snippet-clipboard-content position-relative overflow-auto"><pre><code>@inproceedings{zhao2021accomontage,
  author    = {Jingwei Zhao and Gus Xia},
  title     = {AccoMontage: Accompaniment Arrangement via Phrase Selection and Style Transfer},
  booktitle = {Proceedings of the 22nd International Society for Music Information Retrieval Conference, {ISMIR} 2021, Online, November 7-12, 2021},
  pages     = {833--840},
  year      = {2021},
  url       = {https://archives.ismir.net/ismir2021/paper/000104.pdf}
}
</code></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@inproceedings{zhao2021accomontage,
  author    = {Jingwei Zhao and Gus Xia},
  title     = {AccoMontage: Accompaniment Arrangement via Phrase Selection and Style Transfer},
  booktitle = {Proceedings of the 22nd International Society for Music Information Retrieval Conference, {ISMIR} 2021, Online, November 7-12, 2021},
  pages     = {833--840},
  year      = {2021},
  url       = {https://archives.ismir.net/ismir2021/paper/000104.pdf}
}
" tabindex="0" role="button" style="display: inherit;">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>


For inquries about our work, feel free to contact me at jz4807@nyu.edu.

Jingwei Zhao (Ph.D. Student in Data Science, NUS Graduate School)

Jan. 08, 2022