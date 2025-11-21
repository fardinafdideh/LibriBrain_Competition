# Fused Multi-Branch Residual Multi-Scale CNN BiLSTM Multi-Head with Dual Attention for Language Decoding from MEG: Application for BCI Systems

## Overview
![](https://neural-processing-lab.github.io/2025-libribrain-competition/images/sherlock1.gif)

This repository details the implementation of a hybrid Deep Learning (DL) architecture designed for Magnetoencephalography (MEG) signal processing and decoding in the [LibriBrain Competition 2025](https://neural-processing-lab.github.io/2025-libribrain-competition/).
The model, referred to as the **Fused Multi-Path/Branch (temporal/spectral/temporal-spectral, ...) Residual Multi-Scale Convolutional Neural Network (CNN) Bidirectional Long Short-Term Memory (BiLSTM) Multi-Head with Dual Attention (spatial/temporal)**, is built for signal processing and decoding tasks with substantial societal impact on Brain-Computer Interface (BCI) systems.
The model's primary objective in the [LibriBrain Competition 2025](https://neural-processing-lab.github.io/2025-libribrain-competition/) is to decode language from MEG recordings of brain activity during a listening session, addressing core BCI tasks such as:

1) [Speech Detection](https://neural-processing-lab.github.io/2025-libribrain-competition/tracks/): speech vs. non-speech (silence) events (binary task);
![Description of GIF](https://neural-processing-lab.github.io/2025-libribrain-competition/images/sherlock3.gif)

2) [Phoneme Classification](https://neural-processing-lab.github.io/2025-libribrain-competition/tracks/): 39 Phonems (multi-class task).
![Description of GIF](https://neural-processing-lab.github.io/2025-libribrain-competition/images/sherlock4.gif)

## Competition Leaderboard 
1) Rank 27/53 in [Speech Detection](https://neural-processing-lab.github.io/2025-libribrain-competition/leaderboard/speech_detection_standard/)
2) Rank 18/30 in [Phoneme Classification](https://neural-processing-lab.github.io/2025-libribrain-competition/leaderboard/phoneme_classification_standard/)

## Architecture Diagram
![](/fig/ArchitectureDiagram_MultiScale.png)

## Performance Monitoring 
Performance logs on [Weights & Biases](https://wandb.ai/fardinafdideh-ki/projects) (an AI developer platform):

### [Speech Detection](https://wandb.ai/fardinafdideh-ki/libribrain-experiments/workspace?nw=nwuserfardinafdideh)
![](/fig/speechDetectionPerf.png)
![](/fig/speechDetectionConfusion.png)

### [Phoneme Classification](https://wandb.ai/fardinafdideh-ki/libribrain-phoneme-trainTestWithVal_ChannelSpatialAttentionReductionKernelSize/workspace?nw=nwuserfardinafdideh)
![](/fig/PhonemeClassifPerf.png)
![](/fig/PhonemeClassifConfusion.png)

## Key Components

### 1. **Input Processing**
- **Input Shape**: `(Batch, Channels, Time)` where:
  - `Batch`: Number of samples
  - `Channels`: Number of MEG sensors (e.g., 306 for Neuromag/Elekta systems)
  - `Time`: Temporal sequence length (e.g., 200 time points)

### 2. **Spatial/Channel Attention Module**
- **Purpose**: Identify and focus on the most relevant MEG channels for speech detection
- **Function**: Learns channel-wise importance weights
- **Output**: Weighted MEG signals emphasizing informative sensors
- **Why it matters**: Not all MEG sensors are equally informative for speech - some are positioned over motor cortex, others over auditory cortex, etc.

### 3. **Multi-Scale CNN Blocks (N blocks in series)**

Each Multi-Scale CNN block contains some (here three) parallel processing scales with different temporal receptive fields:

#### Scale Structure (per block):
```
Scale 1 (K=5):  Conv1d(kernel=5)  → BatchNorm1d → GELU
Scale 2 (K=13): Conv1d(kernel=13) → BatchNorm1d → GELU  
Scale 3 (K=25): Conv1d(kernel=25) → BatchNorm1d → GELU
```

#### Multi-Scale Design Rationale:
- **K=5** (~20ms): Captures **phonetic features** - short-term acoustic patterns
- **K=13** (~52ms): Captures **syllabic patterns** - medium-term linguistic units
- **K=25** (~100ms): Captures **prosodic information** - longer-term intonation patterns

#### Processing Flow:
1. **Parallel Processing**: Input is processed simultaneously through all three scales
2. **Concatenation**: Scale outputs are concatenated along the channel dimension:
   - `[B, D/3, T] + [B, D/3, T] + [B, D/3, T] → [B, D, T]`
3. **Dropout Regularization**: Applied after concatenation (rate = 0.15 for CNN layers)
4. **Output**: Multi-scale feature representation flows to the next block

#### Progressive Dimension Expansion:
Blocks progressively expand feature dimensions:
```
Block 1: D₁ = model_dim / 2^(N-1)
Block 2: D₂ = model_dim / 2^(N-2)
...
Block N: D_N = model_dim
```

For example, with `model_dim=128` and `N=3`:
- Block 1: 32 dimensions
- Block 2: 64 dimensions  
- Block 3: 128 dimensions

### 4. **Long Residual Connection**

A critical skip connection that:
- **Starts**: After Spatial Attention output
- **Ends**: At the addition operation after Block N
- **Purpose**: 
  - Preserves gradient flow during backpropagation
  - Training Stabilization: Enables deeper networks without vanishing gradients
  - Maintains direct path for information from input to deep layers
- **Adaptive Projection**: If dimensions mismatch, a 1×1 convolution projects the residual to match

### 5. **Reshape & Normalization**

- **Permute**: Transforms from CNN format `(B, C, T)` to RNN format `(B, T, C)`
- **Layer Normalization**: Stabilizes activations before temporal processing

### 6. **Temporal Processing Pipeline**

#### Bidirectional LSTM
- **Purpose**: Model temporal sequential dependencies in both forward and backward directions
- **Architecture**: 
  - Multiple stacked layers (typically 2-3)
  - Hidden size: `D_N/2` per direction (total output: `D_N`)
  - Dropout between layers for regularization
- **Why bidirectional**: Speech context flows both ways - future phonemes influence current perception

#### Temporal Attention (Multi-head Self-Attention)
- **Purpose**: Learn which time steps are most important for classification
- **Configuration**: 8 attention heads
- **Mechanism**: Computes attention weights across the time dimension
- **Benefit**: Adaptively focuses on informative temporal segments (e.g., speech onset, vowels)

#### Global Average Pooling
- **Function**: Converts `(B, T, D_N)` to `(B, D_N)`
- **Advantage**: More robust
- **Operation**: `output = mean(sequence, dim=time)`

### 7. **Classifier**

Progressive dimension reduction with non-linearity and regularization:
```
Input (B, D_N)
    ↓
Linear(D_N → D_N/2) + Dropout(0.3)
    ↓
GELU activation
    ↓
Dropout(0.15)
    ↓
Linear(D_N/2 → 1)
    ↓
Output (B, 1) - Binary classification logit
```

### 8. **Output**

- **Binary Classification**: Speech vs No-Speech
- **Format**: Single logit per sample (apply sigmoid for probability)

## Complete Data Flow

```mermaid
graph LR
    A[MEG Input<br/>B,C,T] --> B[Spatial Attention<br/>B,C,T]
    B --> C[MS-CNN Block 1<br/>B,D₁,T]
    C --> D[MS-CNN Block 2<br/>B,D₂,T]
    D --> E[...]
    E --> F[MS-CNN Block N<br/>B,D_N,T]
    F --> G((+))
    B -.Long Residual.-> G
    G --> H[Permute<br/>B,T,D_N]
    H --> I[Layer Norm<br/>B,T,D_N]
    I --> J[Bi-LSTM<br/>B,T,D_N]
    J --> K[Temporal Attention<br/>B,T,D_N]
    K --> L[Global Avg Pool<br/>B,D_N]
    L --> M[Classifier<br/>B,1]
    M --> N[Output<br/>Speech/No-Speech]
```

## Dimension Tracking Example

With `batch_size=32, channels=306, time=200, model_dim=128, N=3`:

```
MEG Input:           [32, 306, 200]
Spatial Attention:   [32, 306, 200]
CNN Block 1:         [32, 32, 200]    # D₁ = 128/4 = 32
CNN Block 2:         [32, 64, 200]    # D₂ = 128/2 = 64
CNN Block 3:         [32, 128, 200]   # D₃ = 128
Residual Added:      [32, 128, 200]
Permuted:            [32, 200, 128]
Bi-LSTM:             [32, 200, 128]
Temporal Attention:  [32, 200, 128]
Global Pooled:       [32, 128]
Classifier Layer 1:  [32, 64]
Final Output:        [32, 1]
```

## Design Principles

### 1. **Multi-Scale Feature Extraction**
Inspired by [EEGNet](https://arxiv.org/abs/1611.08024) and recent MEG research, parallel convolutions with different kernel sizes capture neural patterns at multiple temporal scales simultaneously.

### 2. **Attention Mechanisms**
- **Spatial**: Focus on relevant brain regions
- **Temporal**: Focus on important time points
- Both improve interpretability and performance

### 3. **Residual Learning**
Long skip connections enable:
- Deeper networks (prevents vanishing gradients)
- Faster convergence
- Better gradient flow

### 4. **Regularization Strategy**
- **Dropout**: Higher rates (0.3) in classifier, lower (0.15) in feature extractors
- **Batch Normalization**: After each convolution
- **Layer Normalization**: Before recurrent processing
- **Weight Initialization**: Xavier for linear, Kaiming for conv, Orthogonal for LSTM

### 5. **Activation Functions**
- **GELU**: Preferred over ReLU for neural signal processing (smoother, better for small signals)

## Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `input_dim` | 306 | - | Number of MEG channels |
| `model_dim` | 128 | 64-256 | Hidden dimension size |
| `dropout_rate` | 0.3 | 0.2-0.5 | Dropout probability |
| `lstm_layers` | 2 | 2-3 | Number of LSTM layers |
| `cnn_layers` | 3 | 3-6 | Number of MS-CNN blocks (N) |
| `kernel_sizes` | [5,13,25] | [[5,25], [5,13,25]] | Multi-scale kernel sizes |
| `use_attention` | True | [True, False] | Enable spatial attention |
| `use_residual` | True | [True, False] | Enable residual connections |
| `bi_directional` | True | [True, False] | Bidirectional LSTM |
| `smoothing` | 0.1 | - | Label smoothing |

## Training Considerations

### Loss Function
Binary Cross-Entropy (BCE) for speech/no-speech classification

### Optimizer
Adam optimizer recommended with:
- Learning rate: 1e-3 to 1e-4
- Weight decay: 1e-5 to 1e-2

### Learning Rate Schedule
- Warmup for first 5-10% of training
- Cosine annealing or ReduceLROnPlateau

### Data Augmentation (MEG-specific)
- Temporal cropping/shifting
- Channel dropout
- Gaussian noise injection
- Time warping

## Performance Tips
1. **Batch Size**: Start with 32-64, increase if memory permits
2. **Gradient Clipping**: Use value of 1.0 to prevent exploding gradients
3. **Mixed Precision**: Use FP16 training for speed (careful with numerical stability)
4. **Preprocessing**: 
   - Bandpass filter MEG signals (e.g., 0.5-40 Hz)
   - Z-score normalization per channel
   - Remove bad channels/sensors

## References
- LibriBrain Competition 2025: https://neural-processing-lab.github.io/2025-libribrain-competition/
- EEGNet: https://arxiv.org/abs/1611.08024
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- Deep Residual Learning: https://arxiv.org/abs/1512.03385

## How to cite
* **F. Afdideh**, et al., “Fused Multi-Branch Residual Multi-Scale CNN BiLSTM Multi-Head with Dual Attention for Language Decoding from MEG: Application for BCI Systems,” in preparation.
