
# Assignment 4: Cat Face Generation with GANs

In this assignment, we explore various GAN techniques on the MNIST and Cat datasets.  
We implement the [original GAN loss](https://arxiv.org/pdf/1406.2661.pdf) and the [least squares GAN loss (LS-GAN)](https://arxiv.org/abs/1611.04076), and analyze their performance.  
We also investigate instances of mode collapse and experiment with alternative GAN formulations for extra credit.

---

## GAN and LS-GAN on MNIST

We trained both GAN and LS-GAN on the MNIST dataset for 10 epochs each.

### Results:
- GAN:  
  Final samples after 10 epochs (4×4 grid of generated images)
  
- LS-GAN:  
  Final samples after 10 epochs (4×4 grid of generated images)

---

## GAN and LS-GAN on Cat Dataset

We trained both GAN and LS-GAN on the Cat Faces dataset.

### Results:
- GAN:  
  Final samples after 49 epochs (4×4 grid of generated images)

- LS-GAN:  
  Final samples after 48 epochs (4×4 grid of generated images)

---

## Observations

- On the Cat dataset:
  - **GAN** started producing cat-like images as early as epoch 7.
  - **LS-GAN** took longer (around 24 epochs) to produce recognizable cat shapes.
  - LS-GAN losses (both discriminator and generator) stayed between 0 and 1, whereas GAN losses fluctuated between 0 and 10.
  - LS-GAN produced more diverse images compared to GAN on both MNIST and Cat datasets.

---

## Mode Collapse

We observed mode collapse in both GAN and LS-GAN training.

### GAN:
- Epoch 37:
  - Multiple cats in the red and yellow boxes displayed nearly identical features.
  - Many cat images showed yellow/green eyes, indicating lack of diversity.

### LS-GAN:
- Epoch 37:
  - Cat images had slanted textures repeatedly.
- Epoch 40:
  - Multiple cats in the red box looked identical.
- Epoch 44:
  - Most cat images had green eyes, indicating overfitting to certain features.

---

## Extra Credit - Alternative GAN Formulation

We experimented with an alternative GAN formulation:
- Adjusted model architecture and hyperparameters.
- Generated additional output images showcasing improved stability and diversity.

### Output:
- Additional samples and configurations are included in `Extra_Credit.ipynb`.

---

## Usage

### Download the Cat Faces dataset:
```bash
sh download_cat.sh
```

Run the main training notebook:
```bash
jupyter notebook MP4.ipynb
```

For extra credit experiments:
```bash
jupyter notebook Extra_Credit.ipynb
```

---

## 👥 Team

- **Name(s):** Pallaw Kumar, Neha Jain
- **NetID(s):** pallawk2, nehaj4

---
