
# ___***DirectedDiffusion***___

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2302.13153)
[![Project Page](https://img.shields.io/badge/TrailBlazer-Website-green?logo=googlechrome&logoColor=green)](https://hohonu-vicml.github.io/DirectedDiffusion.Page/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://github.com/hohonu-vicml/DirectedDiffusion?tab=readme-ov-file#TODO)
[![Video](https://img.shields.io/badge/YouTube-Project-c4302b?logo=youtube&logoColor=red)](https://github.com/hohonu-vicml/DirectedDiffusion?tab=readme-ov-file#TODO)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhohonu-vicml%2FDirectedDiffusion&count_bg=%23EA00FF&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

This repository contains the implementation of the following paper:
> **Directed Diffusion: Direct Control of Object Placement through Attention Guidance**<br>
> [Wan-Duo Kurt Ma](https://www.linkedin.com/in/kurt-ma/)<sup>1</sup>,[Avisek Lahiri](https://scholar.google.co.in/citations?user=4zgNd2UAAAAJ&hl=en)<sup>1</sup>, [J.P. Lewis](http://www.scribblethink.org/)<sup>3</sup>,[Thomas Leung](https://scholar.google.ca/citations?user=sUK_w2QAAAAJ&hl=en)<sup>1</sup>, [ W. Bastiaan Kleijn](https://people.wgtn.ac.nz/bastiaan.kleijn)<sup>1</sup>,<br>
Victoria University of Wellington<sup>1</sup>, Google Research<sup>2</sup>, NVIDIA Research<sup>3</sup>

## :fire: Overview
![teaser](./assets/figs/teaser.gif)

Text-guided diffusion models such as DALLE-2, Imagen, eDiff-I, and Stable Diffusion are able to generate an effectively endless variety of images given only a short text prompt describing the desired image content. In many cases the images are of very high quality. However, these models often struggle to compose scenes containing several key objects such as characters in specified positional relationships. The missing capability to *direct* the placement of characters and objects both within and across images is crucial in storytelling, as recognized in the literature on film and animation theory. In this work, we take a particularly straightforward approach to providing the needed direction. Drawing on the observation that the cross-attention maps for prompt words reflect the spatial layout of objects denoted by those words, we introduce an optimization objective that produces ``activation'' at desired positions in these cross-attention maps. The resulting approach is a step toward generalizing the applicability of text-guided diffusion models beyond single images to collections of related images, as in storybooks. *Directed Diffusion* provides easy high-level positional control over multiple objects, while making use of an existing pre-trained model and maintaining a coherent blend between the positioned objects and the background. Moreover, it requires only a few lines to implement.

## :fire: Requirements

The codebase is tested under **NVIDIA GeForce RTX 3090** with the python library **pytorch-2.1.2+cu121** and **diffusers-0.21.4**. We strongly recommend using a specific version of Diffusers as it is continuously evolving. For PyTorch, you could probably use other version under 2.x.x. With RTX 3090, I follow the [post](https://discuss.pytorch.org/t/geforce-rtx-3090-with-cuda-capability-sm-86-is-not-compatible-with-the-current-pytorch-installation/123499) to avoid the compatibility of sm_86 issue.

## :fire: Timeline

-   [2024/02/22]: We will present our poster at Poster Session 1 in Vancouver Convention Centre.
-   [2023/12/08]: Directed Diffusion has been accepted by AAAI 2024
-   [2023/09/28]: We have updated the latest version v3 of our ArXiv paper. More validations, more ablations, with enhanced methodologies.
-   [2023/03/16]: Gradio Web UI integrated! See [walk-through](doc/walk-through.org) for more info.
-   [2023/03/01]: First version submitted on ArXiv

## :fire: Usage

Please refer to our [walk-through](doc/walk-through.org) for more information
from scratch. Once you get familiar with it, then you could reproduce our paper
result and alter some of the parameter in [paper-result](doc/paper-result.org).
We also provide some of the experiments that are not listed in our paper for fun
[other-prompt](doc/other-prompt.org)!

## :fire: Acknolwedgements

We thank Jason Baldridge, Avisek Lahiri, and Arkanath Pathak for helpful
feedback.

## :fire: TODO

We apologize this repository is not fully udpated and stays at our v1 ArXiv
version [link](https://arxiv.org/abs/2302.13153v1). We are actively working on
enhancing our codebase, which is currently at the first ArXiv version. The
updates are scheduled for release in March 2024, along with an online
HuggingFace demo.

## :fire: Citation

 [TrailBlazer](https://hohonu-vicml.github.io/Trailblazer.Page/) is the recent
 descendant project extended from Directed Diffusion in the context of video
 generation. If you find our work useful for your research, please consider
 citing our paper.

   ```bibtex
   @article{ma2023trailblazer,
       title={TrailBlazer: Trajectory Control for Diffusion-Based Video Generation},
       author={Wan-Duo Kurt Ma and J. P. Lewis and W. Bastiaan Kleijn},
       year={2023},
       eprint={2401.00896},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
   }

   @article{ma2023directed,
       title={Directed Diffusion: Direct Control of Object Placement through Attention Guidance},
       author={Wan-Duo Kurt Ma and J. P. Lewis and Avisek Lahiri and Thomas Leung and W. Bastiaan Kleijn},
       year={2023},
       eprint={2302.13153},
       archivePrefix={arXiv},
       primaryClass={cs.CV}
   }
   ```
