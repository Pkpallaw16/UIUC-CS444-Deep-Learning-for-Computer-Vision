
# Assignment 2: Multi-Layer Neural Networks 

## Image Reconstruction
In this [paper](https://bmild.github.io/fourfeat/), the author researched the problem of reconstructing an image via an MLP model. Specifically, the MLP takes the coordinates to a pixel as input and outputs the pixel's RGB values. The authors devised a Fourier feature mapping to enhance the reconstruction of high frequency components of the image, increasing the quality of the reconstructed image.

## Implement and train MLP with Fourier Feature and Positional Encoding  

We experimented with multi-layer perceptrons (MLPs) to reconstruct images from coordinates, with different coordinate mapping techniques: None, Basic Positional Encoding, and Fourier Feature Encoding at different scales.  
The implementations include training and evaluation with both SGD and Adam optimizers.  

---

## 🧪 Experiments  

We conducted experiments on both **low-resolution** and **high-resolution** images, and also tested on a custom image.  
Hyperparameter tuning was done for learning rates, number of layers, hidden size, epochs, optimizers, and Fourier Feature scales (σ).  

Results include MSE test errors and observations on the performance of different settings.

---

## 📄 Results  

### Part 1: Low-resolution with **SGD**
| Mapping         | MSE Test |
|-----------------|----------|
| None            | 0.0227 |
| Basic           | 0.0143 |
| Gauss 1.0       | 0.0104 |
| Gauss 10.0      | 0.0416 |
| Gauss 100.0     | 0.0414 |

- Settings: 5 layers, 256 hidden size, 1000 epochs, learning rate = 1e-1, SGD
- Computation time: ~6 mins

---

### Part 2: Low-resolution with **Adam**
| Mapping         | MSE Test |
|-----------------|----------|
| None            | 0.0127 |
| Basic           | 0.011 |
| Gauss 1.0       | 0.0106 |
| Gauss 10.0      | 0.0427 |
| Gauss 100.0     | 0.0491 |

- Settings: layers: 4–7 (best varies by mapping), 256 hidden size, 1000 epochs, learning rate = 1e-4, Adam
- Computation time: ~1 min per mapping

---

### Part 3: High-resolution  
| Mapping         | MSE Test |
|-----------------|----------|
| None            | 0.0126 |
| Basic           | 0.0061 |
| Gauss 1.0       | 0.00434 |
| Gauss 10.0      | 0.00495 |
| Gauss 100.0     | 0.0577 |

- Settings: 5 layers, 256 hidden size, 1000 epochs, learning rate = 1e-4, Adam
- Computation time: ~1 hr 20 mins

---

### Part 4: High-resolution (custom image)
| Mapping         | MSE Test |
|-----------------|----------|
| Gauss 10.0      | 0.0248 |

- Settings: 5 layers, 256 hidden size, 1000 epochs, learning rate = 1e-4, Adam

---

## 📋 Discussion  

- 🔹 **Learning Rate:** Best performance was achieved with a learning rate of 1e-4. Lower (1e-5) led to underfitting, higher (1e-1) led to divergence.  
- 🔹 **Number of Layers:** Best values varied by mapping, chosen empirically.  
- 🔹 **Optimizers:** Adam outperformed SGD consistently because of its momentum and adaptive learning rate.  
- 🔹 **Coordinate Mappings:**
  - Low-res: Gauss 1.0 performed best; None and Basic underfit, Gauss 100 overfit.
  - High-res: Gauss 10.0 performed best; None/Basic/1.0 underfit, Gauss 100 overfit and noisy.
  - General insight: higher resolution benefits from slightly larger σ (e.g., 10), but too high (e.g., 100) harms reconstruction.

### Insights from Custom Image:
> We chose a colorful, detailed image which was harder to reconstruct than simpler images like the lion. The lion image reconstructed better, possibly because of fewer features and simpler structure.

---

## 👥 Team  

- **Name(s):** Pallaw Kumar, Neha Jain 
- **NetID(s):** pallawk2, nehaj4  

---
