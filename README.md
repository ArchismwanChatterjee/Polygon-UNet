# **Polygon Coloring with Conditional UNet**

WanDB Report : https://api.wandb.ai/links/archismwancmail-work/365h0gb9

WanDB Project : https://wandb.ai/archismwancmail-work/polygon-coloring-unet

Model Link : https://drive.google.com/drive/folders/1CL79mI8K-kAomEmfi0zaOD13VbnXfVnL?usp=sharing

Google Colab Link : https://colab.research.google.com/drive/1xuvb3bIPDADn7hahDqoraMaLmT22PffI?usp=sharing

Insights Report Link : https://docs.google.com/document/d/1T3U82WREPT_NSBegmrrYOgEDcPQ-jf-950Yjd0aW-g8/edit?usp=sharing

## **Problem**

Train a **UNet from scratch** to color a polygon image based on a given color name.

* **Input**: Grayscale polygon image + color name
* **Output**: Colored polygon image


## ✅ **Model Overview**

* **Base Architecture**: UNet with 4 downsampling and 4 upsampling blocks.
* **Conditioning Mechanism**: Color name encoded as a **one-hot vector**, expanded spatially and **concatenated with input image channels**.
* **Input Shape**: `1-channel grayscale image + color embedding channels`
* **Output Shape**: 3-channel RGB image.
* **Activation**: `Sigmoid` at output → normalized to `[0,1]`.


## ✅ **Hyperparameters**

| Parameter       | Tried Values                | Final   |
| --------------- | --------------------------- | ------- |
| Image Size      | 64×64, 128×128              | 128×128 |
| Batch Size      | 8, 16, 32                   | 16      |
| Learning Rate   | 1e-3, 1e-4, 5e-5            | 1e-4    |
| Optimizer       | Adam, AdamW                 | Adam    |
| Loss            | L1 Loss, Smooth L1, MSE     | L1 Loss |
| Epochs          | 30, 50, 100                 | 50      |
| Color Embedding | One-hot, Dense Linear Layer | One-hot |

**Rationale:**

* **L1 Loss** works well for pixel-based reconstruction tasks (less blurry than MSE).
* **128×128** chosen for balance between detail and speed.
* **Adam optimizer** provides stable convergence.


## ✅ **Architecture Details**

* **Encoder**:

  * 4 blocks of `DoubleConv` (Conv → BN → ReLU ×2)
  * MaxPooling after each block
* **Bottleneck**:

  * `DoubleConv(512 → 1024)`
* **Decoder**:

  * Transposed Convolution for upsampling
  * Skip connections from encoder
* **Conditioning**:

  * One-hot color vector (length = number of colors)
  * Reshaped to `(B, embed_dim, H, W)` and concatenated with input polygon image.


### ✅ **ASCII Diagram of UNet with Conditioning**

```
          Color One-hot Vector (e.g., [0,0,1,0,0])
                        │
                        ▼
             ┌──────────────────────┐
             │ Expand & Tile to H×W │
             └──────────────────────┘
                        │
                        ▼
      Input Polygon Image (1-ch)   Conditioning Map
                    │                      │
                    └──────────┬──────────┘
                               ▼
                   Concatenate along channels
                               │
                               ▼
                        [Encoder Path]
      ┌───────────────────────────────────────────────────┐
      │    Down1 → Down2 → Down3 → Down4 → Bottleneck    │
      └───────────────────────────────────────────────────┘
                               │
                        [Decoder Path]
      ┌───────────────────────────────────────────────────┐
      │    Up4 ← Up3 ← Up2 ← Up1 with skip connections   │
      └───────────────────────────────────────────────────┘
                               │
                               ▼
                         Final Conv (3-ch)
                               │
                               ▼
                      Output: Colored Polygon Image
```


---

## **Training Dynamics**

### **Loss Curves**

* **Train Loss (`train_loss`)** and **Validation Loss (`val_loss`)** both **consistently decrease** across 50 epochs.
* **No signs of overfitting** — validation loss tracks closely with training loss.
* Plateau begins after **\~40 epochs**, indicating convergence.

### **Qualitative Trends**

* **Early Epochs**:

  * Light/faint color fill with intact polygon edges.
* **Mid Epochs**:

  * Colors improve in accuracy.
  * Occasional minor artifacts and slight edge blurring.
* **Final Epochs**:

  * High color fidelity achieved.
  * Small amount of color bleeding observed at sharp boundaries.

## ✅ **Evaluation Metrics and Outputs**

For Sample Validation File Octagon.png :

* **PSNR(Peak Signal to Noise Ratio)**: 14.08 dB
* **SSIM(Structural Similarity Index)**: 0.8726
* Ground truth comparison
    <img width="1182" height="384" alt="output" src="https://github.com/user-attachments/assets/7ef12051-eafb-4e0a-a8f4-cad4a88ad53a" />
* Multiple color inference for the same polygon
   <img width="1182" height="152" alt="output2" src="https://github.com/user-attachments/assets/2a25ff9d-3bb9-484a-ae8d-d481d89a5467" />

## ✅ **Key Learnings** 


* Conditional UNet can handle **image + text conditioning** effectively even with simple one-hot encoding.
* **L1 Loss** outperforms MSE for preserving sharp edges.
* **Data augmentation** is crucial for generalization.
* UNet works well for structured generation tasks with strong spatial priors.


## ✅ **Files**

* `model.py`: Conditional UNet implementation.
* `train.py`: Training loop with wandb logging.
* `inference.ipynb`: Visualizes predictions, computes PSNR & SSIM.
* `output.png` : Visualizes  Ground truth comparison
* `output2.png` : Visualizes multiple color inference for the same polygon
* `unet_polygon_color.pth` : The UNet Model



