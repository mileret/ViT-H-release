<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">ViT-H: Generative Video-To-Humanoid Control with Motion Tracking Feedback</h1>

[**Zimo He**](https://mileret.github.io/)<sup>&ast;</sup> · [**Tianyi Bian**](https://pku.ai/author/tianyi-bian/)<sup>&ast;</sup> · [**Zizhong Yang**](https://pku.ai/author/zizhong-yang/)<sup>&ast;</sup> · [**Hongjie Li**](https://awfuact.github.io/)<sup>&ast;&dagger;</sup> · [**Nan Jiang**](https://jnnan.github.io/) · [**Jieming Cui**](https://jiemingcui.github.io/) · [**Baoxiong Jia**](https://buzz-beater.github.io/) · [**Yizhou Wang**](https://cfcs.pku.edu.cn/wangyizhou/)<sup>&ddagger;</sup> · [**Siyuan Huang**](https://siyuanhuang.com/)<sup>&ddagger;</sup> · [**Yixin Zhu**](https://yzhu.io/)<sup>&ddagger;</sup> 
<br>
&ast;Equal Contribution&emsp;&dagger;Project Lead&emsp;&ddagger;Corresponding Authors&emsp;

<a href='https://arxiv.org/abs/2610.32966'><img src='https://img.shields.io/badge/arXiv-ViT--H-red' alt='arXiv'></a>
<a href='https://awfuact.github.io/vit-h/'><img src='https://img.shields.io/badge/Project_Page-ViT--H-green' alt='Project Page'></a>
<a href='https://youtu.be/j3q7f4-EUpk'><img src='https://img.shields.io/badge/Video-ViT--H-yellow' alt='Video'></a>
</div>

![alt text](assets/teaser.png)

This is the official implementation of <i>ViT-H: Generative Video-To-Humanoid Control with Motion Tracking Feedback.</i>

## Plans

- [x] Release code for offline video-to-humanoid control.
- [ ] Release code for humanoid video teleoperation by December 1.

## Installation

The offline release targets Linux, Python 3.10, CUDA 12.1, and PyTorch 2.3.

```bash
conda create -n vith python=3.10 -y
conda activate vith
pip install -r requirements.txt
pip install -e .
```

## Dataset

The training dataset is distributed separately as five compressed
archives: `amass`, `bedlam`, `bedlam2`, `h36m`, and `bones_seed`. Download the
[ViT-H training dataset](https://huggingface.co/datasets/eerrr1/ViT-H) and
extract all archives into one directory. The original BONES-SEED dataset is
available from its [official Hugging Face repository](https://huggingface.co/datasets/bones-studio/seed).

```bash
hf download eerrr1/ViT-H \
  --repo-type dataset \
  --local-dir /path/to/training_set

cd /path/to/training_set
for archive in amass bedlam bedlam2 h36m bones_seed; do
  tar --use-compress-program=unzstd -xf "${archive}.tar.zst"
done

export VITH_DATASET_ROOT=/path/to/training_set
```

The extracted layout is documented in [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md).
Dataset components and body-model assets may have terms separate from this
repository's code license.

## Training

The released experiment uses a 16-layer transformer with latent dimension
1024, 16 attention heads, 120-frame windows, and mixed-source batches of 32.

```bash
export VITH_DATASET_ROOT=/path/to/training_set
export NUM_DEVICES=8
export WANDB_MODE=offline
bash scripts/train.sh
```

For Slurm clusters, adjust the site-specific directives in
`slurm/train.sbatch`, export `VITH_DATASET_ROOT`, and submit:

```bash
sbatch slurm/train.sbatch
```

## Offline Inference

Model checkpoints are not stored in this repository. Pass the checkpoint path
explicitly:

```bash
python tools/infer_video.py \
  --video /path/to/input.mp4 \
  --ckpt /path/to/model.ckpt \
  --output outputs/inference/input_g1.npz
```

Video preprocessing additionally requires the external detector, pose,
body-model, and image-encoder assets listed in
[docs/INFERENCE.md](docs/INFERENCE.md). Use `--static-cam` only for videos with
a known static camera; otherwise the default preprocessing estimates camera
motion.

## Release Contents

- `hmr4d/`: datasets, model, preprocessing, and shared utilities.
- `motion_lib/`: G1 kinematics and rotation utilities.
- `scripts/train.sh`: reproducible offline training entry point.
- `tools/infer_video.py`: end-to-end video inference entry point.
- `experiments/g1_training_set_exp1/`: resolved experiment configuration and checksums.
- `docs/`: dataset and inference documentation.

## Citation

If you find ViT-H helpful, please consider citing:
```bibtex
@article{he2026vith,
  title={ViT-H: Generative Video-To-Humanoid Control with Motion Tracking Feedback},
  author={He, Zimo and Bian, Tianyi and Yang, Zizhong and Li, Hongjie and Jiang, Nan and Cui, Jieming and Jia, Baoxiong and Wang, Yizhou and Huang, Siyuan and Zhu, Yixin},
  journal={arXiv preprint arXiv:2610.32966},
  year={2026}
}
```
