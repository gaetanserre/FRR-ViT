## Fundamentals of reproducible research and free software project -- Vision Transformer

This repository uses the original implementation of [Vision Transformer](https://arxiv.org/abs/2010.11929) to check the reproducibility of the paper. The original implementation is [here](https://github.com/google-research/vision_transformer).

We selected two models: ViT-B/16 and ViT-L/16 using weights pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k. We evaluate these models on the validation split of ImageNet-ILSVRC 2012 using Top1-accuracy. The results are shown in the following table.

| Model | Reproduced | Original |
| :---: | :---: | :---: |
| ViT-B/16 | 81.58 | 83.97 |
| ViT-L/16 | 82.73 | 85.15 |

The results are slightly different from the original paper. We believe that it is due to a difference in the evaluation data.

## Usage
1. Install requirements (PyTorch, torchvision, timm, tqdm).
    ```
    pip install -r requirements.txt
    ```

2. Download ImageNet-ILSVRC 2012 validation dataset and place it in a `ImageNet` directory.
3. Execute the Jupiter notebook `reproduce.ipynb`.