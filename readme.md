# AccoMontage
<a href="https://colab.research.google.com/drive/1F4saDkh45KNxePD5yEcje61b0F09buDW?usp=sharing)" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

AccoMontage is a piano accompaniment arrangement system. It introduces a novel hybrid pathway, in which rule-based optimization (for high-level structure) and learning-based style transfer (for fien-grained conherency) are both leveraged to complement each other for high-quality generation. Our paper [*AccoMontage: Accompaniment Arrangement via Phrase Selection and Style Transfer*](https://arxiv.org/abs/2108.11213) is accepted by [ISMIR 2021](https://ismir2021.ismir.net/). This repository stores codes and demos of our work.

## New Features
AccoMontage now supports a few new features as follows:
* Generation with **MIDI velocity** and **pedal control messages**.
* Transitions among **phrases of any length** (1-bar, 2-bar, ..., 16-bar).
* Input of general MIDI with **arbituary tracks** (besides melody) and arbituarily **complex chords** (e.g., 9th chords). Yet, we still quantize the chord sequence at 1-beat unit. If the melody is not on the first track, then the melody track index is also requested.
* Whole pieces arrangement with **intro**, **interlude**, and **outro**.
* **Spotlight** certain reference pieces from [POP909 dataset](https://github.com/music-x-lab/POP909-Dataset) as the donor of piano textures. Currently supported spotlight options include POP909 song index (e.g., 905), song name (e.g., '小城故事'), and/or artist name (e.g., '邓丽君'). For complete supported options, refer to the checktable `./checkpoints/pop909_quadraple_meters_index.xlsx`.

## Run
* Data and checkpoints required to run AccoMontage can be downloaded [here](https://drive.google.com/file/d/1zQ5xds8oeeAlnn_PK5e0PWNKyM7unUFO/view?usp=sharing) (updated May 29, 2023). After extraction, you should have a `./checkpoints/` folder with relevant pt and npz files inside. 
* Our code is now arranged in a portable manner. You can follow the guidance in [`./AccoMontage_inference.ipynb`](./AccoMontage_inference.ipynb) and run AccoMontage.
* Alternatively, AccoMontage is now accessible on [Google Colab](https://colab.research.google.com/drive/1F4saDkh45KNxePD5yEcje61b0F09buDW?usp=sharing), where you can quickly test it online. 

## Demos
Generated demos are listed in `./demo`. AccoMontage was applied as the backbones to rearrange the Alma Mater Music of East China Normal University (ECNU). Performance demos can be accessed [here](https://zhaojw1998.github.io/accomontage_demo).

## Acknowledgement
Thanks to Prof. Gus Xia, Yixiao Zhang, Liwei Lin, Junyan Jiang, Ziyu Wang, and Shuqi Dai for their generous help to this work. Thanks to all NYUSH Music X Lab citizens for their encouragement and companion. The following repositories are referred to by this work:

- https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement
- https://github.com/ZZWaang/PianoTree-VAE 
- https://github.com/Dsqvival/hierarchical-structure-analysis
- https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition
- https://github.com/buggyyang/Deep-Music-Analogy-Demos


## Cite Our Work
If you find our paper and this repository useful, please consider citing our work:

<div class="snippet-clipboard-content position-relative overflow-auto"><pre><code>@inproceedings{zhao2021accomontage,
  author    = {Jingwei Zhao and Gus Xia},
  title     = {AccoMontage: Accompaniment Arrangement via Phrase Selection and Style Transfer},
  booktitle = {Proceedings of the 22nd International Society for Music Information Retrieval Conference ({ISMIR} 2021)},
  pages     = {833--840},
  year      = {2021},
}
</code></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@inproceedings{zhao2021accomontage,
  author    = {Jingwei Zhao and Gus Xia},
  title     = {AccoMontage: Accompaniment Arrangement via Phrase Selection and Style Transfer},
  booktitle = {Proceedings of the 22nd International Society for Music Information Retrieval Conference ({ISMIR} 2021)},
  pages     = {833--840},
  year      = {2021},
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


For inquries about our work, feel free to contact me at jzhao@u.nus.edu.

Jingwei Zhao (Ph.D. Student in Data Science, NUS Graduate School)

May 29, 2023