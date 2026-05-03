# Project Title

## Demonstration

<!-- Add your demo images below -->
<!-- Example:
![Demo 1](path/to/image1.png)
![Demo 2](path/to/image2.png)
-->

---

## Abstract

<!-- Add your abstract here -->

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

1. Configure the parameters in the `X.yaml` configuration file:
   - Path to the **REGEN pretrained model**
   - Prompt of FLUX.2-4B Klein
   - Path to the **input images**
   - Path to the **output directory**
   - Desired resolution for the photorealism-enhanced (output) images

2. Run the enhancer script:

```bash
python enhancer.py
```
