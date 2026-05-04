# A Hybrid Approach for Closing the Sim2real Appearance Gap in Game Engine Synthetic Datasets

## Demonstration

<img width="1480" height="628" alt="example" src="https://github.com/user-attachments/assets/d185143e-9f52-4b07-9e0d-c3b319f9a03f" />

---

## Abstract

Video game engines have been an important source for generating large volumes of visual synthetic datasets for training and evaluating computer vision algorithms that are to be deployed in the real-world. While the visual fidelity of modern game engines has been significantly improved with technologies such as ray-tracing, a notable sim2real appearance gap between the synthetic and the real-world images still remains, which limits the utilization of synthetic datasets in real-world applications. In this letter, we investigate the ability of a state-of-the-art diffusion model (FLUX.2-4B Klein) to enhance the photorealism of synthetic datasets and compare its performance against a traditional image-to-image translation model (REGEN). Furthermore, we propose a hybrid approach that combines the strong geometry and material transformations of diffusion-based methods with the distribution-matching capabilities of image-to-image translation techniques. Through experiments, it is demonstrated that REGEN outperforms FLUX.2-4B Klein and that by combining both FLUX.2-4B Klein and REGEN models, better visual realism can be achieved compared to using each model individually, while maintaining semantic consistency.

---

## Citation

If you used the code of this repository in your work, we would appreciate using the following citation:

```bibtex
% Add your BibTeX entry here
```

---

## Requirements

This project relies on the following models:

- **FLUX.2-4B Klein**
- **REGEN**

Please refer to their official repositories for installation and requirements:

- REGEN [(GitHub repository)](https://github.com/stefanos50/REGEN)
- FLUX.2-4B Klein [(Hugging Face repository)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)


## How to Run

1. Configure the parameters in the `options.yaml` configuration file:
   - Path to the **REGEN pretrained model** (can be found [here](https://github.com/stefanos50/REGEN))
   - Path to the **input images**
   - Path to the **output directory**
   - Desired resolution for the photorealism-enhanced (output) images (optimal is null)
   - Hugging Face token for accessing FLUX.2-4B Klein

2. Run the enhancer script:

```bash
python enhancer.py
```
