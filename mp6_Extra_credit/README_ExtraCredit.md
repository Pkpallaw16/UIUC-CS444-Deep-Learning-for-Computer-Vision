
# Extra Credit Assignment: Adversarial Attacks  

## Attack Methods  

In this assignment, we attack a fine-tuned ResNet-18 model trained on the ImageNette dataset using the following 3 attack methods described in [this paper](https://arxiv.org/pdf/1607.02533):  

- **Fast Gradient Sign Method (FGSM)**  
  - The gradient of the loss w.r.t the input image is computed.
  - The image is perturbed in the direction of the gradient sign, scaled by ε.
  - Hyperparameters:
    - ε ∈ [0, 2/255, 4/255, 8/255, 16/255, 32/255]
  - Observations: Accuracy dropped from ~98% (ε=0) to ~30% even at smallest ε.

- **Iterative Gradient Sign Method / Basic Iterative Method (BIM)**  
  - Multiple small steps in the gradient sign direction.
  - After each step, the perturbed image is clipped to remain within the ε-ball.
  - Hyperparameters:
    - ε ∈ [0, 2/255, 4/255, 8/255, 16/255, 32/255]
    - Step size (α): 0.0004
    - Iterations: min(eps+1, 1.25×eps, 10)
  - Observations: Accuracy drops gradually; ~19% accuracy at ε=8/255.

- **Iterative Least-Likely Class Method (ILLCM)**  
  - Perturbs the image towards the least-likely class predicted by the model.
  - Hyperparameters:
    - ε ∈ [0, 2/255, 4/255, 8/255, 16/255, 32/255]
    - Step size (α): 0.004
    - Iterations: min(eps+4, 1.25×eps, 10)
  - Observations: Most effective; drops accuracy to 0% at ε ≥ 8/255.

We only attack images from ImageNette.  
The base model achieved ~98.49% accuracy on clean test data.  

---

## Perform an Attack

### Preparation
Place all images to be attacked in a single folder.  

### Usage
Run `main.py` with a config file specifying the attack method, ε, model checkpoint, etc.  

Example (for ResNet-18):  
```bash
python main.py --config configs/resnet18.yaml
```

To generate adversarial examples for all three methods at once, set `method='all'` in the config.  
Configs are in the `config/` folder.  
You can download the ImageNette dataset [here](https://github.com/fastai/imagenette?tab=readme-ov-file).  

---

## Results

### Accuracy vs. ε

| Method | ε=0 | ε=2 | ε=4 | ε=8 | ε=16 | ε=32 |
|--------|-----|-----|-----|-----|------|------|
| **FGSM** | 98% | 30% | 22% | 24% | 30% | 30% |
| **BIM**  | 98% | 90% | 59% | 19% | 13% | 13% |
| **ILLCM**| 98% | 52% | 1.4%| 0%  | 0%  | 0%  |

Plots of accuracy vs. ε and example adversarial images for each method and ε are provided in the report.  

---

## Robustness to Non-Adversarial Distribution Shift

We also evaluated robustness to real-world image degradations using ImageNet-C perturbations on ImageNette.

### Usage
Create a corrupted ImageNette dataset:
```bash
python corrupt.py
```

Train on corrupted data:
```bash
python train_imagenette_c.py --config config/imagenette_c.yaml
```

### Results

| Training & Testing Setup | Accuracy |
|---------------------------|----------|
| Pretrained ResNet, Test on ImageNette | 84.62% |
| Pretrained ResNet, Test on corrupted | 62.01% |
| Train on corrupted, Test on ImageNette | 91.96% |
| Train on corrupted, Test on corrupted | 90.82% |
| Train on both, Test on both | 91.98% |
| Fine-tune pretrained on corrupted, Test on corrupted | 78.39% |
| Fine-tune pretrained on corrupted, Test on both | 82.57% |

---

## Discussion

- FGSM: Fastest, but less effective and produces visible perturbations at high ε.
- BIM: More effective, less visible perturbations, but computationally expensive.
- ILLCM: Most effective, almost no visible perturbations, but computationally the most expensive.

---

## 👥 Team

- **Name(s):** Pallaw Kumar, Neha Jain
- **NetID(s):** pallawk2, nehaj4

---
