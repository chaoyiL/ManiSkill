---
license: cc-by-nc-4.0
tags:
- sparsh-x
- img
- base
---

# Sparsh-X (tactile img only) base-sized model

[Sparsh-X (img)](https://akashsharma02.github.io/sparsh-x-ssl/) is a transformer-based backbone for encodeing the tatcile image from the [Digit 360](https://digit.ml/) sensor. This image captures the 360 elastomer dome, providing a better view of the fingertip in terms of contact. The model is trained using self-distillation SSL (DINO loss), specifically adapted for the Digit 360 touch sensor.

Disclaimer: This model card was written by the Sparsh-X(img) authors. The Transformer architecture and DINO objectives have been adapted for the fish-eye tactile image.

## Model description
The model takes two tactile images from the Digit 360 sensor as input.  These images are sampled at 30fps and passed to the model with a temporal stride of 5 concatenated along the channel dimension $I_t ⊕ I_{t−5} → x ∈ R^{h×w×6}$. We crop to zoom-in the fish-eye image and resize to 224 × 224 × 3. Image patches (16 × 16) are then tokenized to embeddings of 768 dimensions through a linear projection layer. 


## Intended uses & limitations
You can utilize the Sparsh-X model to extract tactile image representations for the Digit 360 sensor. You have two options:

1. Use the frozen Sparsh-X encoder: This allows you to leverage the pre-trained weights of the Sparsh-X model without modifying them.
2. Fine-tune the Sparsh-X encoder: You can fine-tune the Sparsh-X encoder along with the training of your downstream task, allowing the model to adapt to your specific use case.

Both options enable you to take advantage of the powerful touch representations learned by the Sparsh-X model.

## How to Use
For detailed instructions on how to load the encoder and integrate it into your downstream task, please refer to our [GitHub repository](https://github.com/facebookresearch/sparsh-multisensory-touch).

## Citation

```bibtex
@inproceedings{higuera2025tactile,
title={Tactile Beyond Pixels: Multisensory Touch Representations for Robot Manipulation},
author={Carolina Higuera and Akash Sharma and Taosha Fan and Chaithanya Krishna Bodduluri and Byron Boots and Michael Kaess and Mike Lambeta and Tingfan Wu and Zixi Liu and Francois Robert Hogan and Mustafa Mukadam},
booktitle={9th Annual Conference on Robot Learning},
year={2025},
url={https://openreview.net/forum?id=sMs4pJYhWi}
}
```

```bibtex
@article{lambeta2024digitizing,
  title={Digitizing touch with an artificial multimodal fingertip},
  author={Lambeta, Mike and Wu, Tingfan and Sengul, Ali and Most, Victoria Rose and Black, Nolan and Sawyer, Kevin and Mercado, Romeo and Qi, Haozhi and Sohn, Alexander and Taylor, Byron and others},
  journal={arXiv preprint arXiv:2411.02479},
  year={2024}
}
```