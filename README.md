## NeRFFaceEditing: Disentangled Face Editing in Neural Radiance Fields<br><sub>Official PyTorch implementation of the SIGGRAPH Asia 2022 Conference paper</sub>

![Teaser image](./docs/teaser.png)

**NeRFFaceEditing: Disentangled Face Editing in Neural Radiance Fields**<br>
Kaiwen Jiang, Shu-Yu Chen, Feng-Lin Liu, Hongbo Fu, Lin Gao<br>

Abstract: *Recent methods for synthesizing 3D-aware face images have achieved rapid development thanks to neural radiance fields, allowing for high quality and fast inference speed. However, existing solutions for editing facial geometry and appearance independently usually require retraining and are not optimized for the recent work of generation, thus tending to lag behind the generation process. To address these issues, we introduce NeRFFaceEditing, which enables editing and decoupling geometry and appearance in the pretrained tri-plane-based neural radiance field while retaining its high quality and fast inference speed. Our key idea for disentanglement is to use the statistics of the tri-plane to represent the high-level appearance of its corresponding facial volume. Moreover, we leverage a generated 3D-continuous semantic mask as an intermediary for geometry editing. We devise a geometry decoder (whose output is unchanged when the appearance changes) and an appearance decoder. The geometry decoder aligns the original facial volume with the semantic mask volume. We also enhance the disentanglement by explicitly regularizing rendered images with the same appearance but different geometry to be similar in terms of color distribution for each facial component separately. Our method allows users to edit via semantic masks with decoupled control of geometry and appearance. Both qualitative and quantitative evaluations show the superior geometry and appearance control abilities of our method compared to existing and alternative solutions.*

[Project Page](http://geometrylearning.com/NeRFFaceEditing/) | [Paper Arxiv](https://arxiv.org/pdf/2211.07968.pdf)

## Requirements
* The code is tested only on the Linux platform.
* Environment setup commands:
```shell
$ conda env create -f environment.yml
$ conda activate NeRFFaceEditing
$ python -m ipykernel install --user --name=NeRFFaceEditing
```

## Getting started
Please download the pre-trained checkpoints from [link](https://drive.google.com/file/d/1PEtz2_TtxB6MTdaaoV8ya3eBlOJsjQkp/view?usp=share_link) and put them under `./pretrained/`. The link contains the checkpoint for the reimplemented EG3D and the NeRFFaceEditing.

We provide a notebook `inference.ipynb` to demonstrate the generation ability of reimplemented EG3D and the editing ability of NeRFFaceEditing.

### Projection (Optional)
We use the [DECA](https://github.com/YadiraF/DECA) to estimate the pose of faces.
For projection, please follow its description of preparing the data and then copy the `decalib` and `data` from the official repository to the `./external_dependencies/`.

### Training the EG3D (Optional)
We tweaked a few details for reimplementation which can be found in our supplementary.
To train the EG3D from scratch, please download the FFHQ dataset from [link](https://github.com/NVlabs/ffhq-dataset) and the estimated position data `pos.pkl` from [link](https://drive.google.com/file/d/1t7SVoZ12O_l0WwAGt16v5qWQTXW-6WFZ/view?usp=sharing) and put the `pos.pkl` under the directory of the dataset.

We train the model with 4 Tesla V100 GPU with the following command:
```shell
$ ./start.sh
```

## Citation
```
@InProceedings{NerfFaceEditing,
  author    = {Kaiwen Jiang and Shu-Yu Chen and Feng-Lin Liu and Hongbo Fu and Lin Gao},
  title     = {NeRFFaceEditing: Disentangled Face Editing in Neural Radiance Fields},
  year      = {2022},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  booktitle = {ACM SIGGRAPH Asia 2022 Conference Proceedings},
  location = {Daegu, Korea},
  series = {SIGGRAPH Asia'22}
}
```

## Development
We plan to release the training script and the version of NeRFFaceEditing implemented on the official EG3D.

## Acknowledgements
Part of the codes are borrowed from [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), [PTI](https://github.com/danielroich/PTI), [pi-GAN](https://github.com/marcoamonteiro/pi-GAN) and [pytorch3d](https://github.com/facebookresearch/pytorch3d).