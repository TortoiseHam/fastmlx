"""Example applications built with fastmlx.

Image Classification:
    - mnist: MNIST digit classification with LeNet
    - cifar10: CIFAR-10 classification with ResNet9
    - fashion_mnist: Fashion-MNIST classification with LeNet
    - vit_cifar10: CIFAR-10 with Vision Transformer
    - wideresnet_cifar10: CIFAR-10 with WideResNet variants

Image Segmentation:
    - unet_segmentation: UNet on synthetic circular mask data

Training Techniques:
    - super_convergence: 1cycle learning rate policy for fast training
    - fgsm_adversarial: FGSM adversarial training for robustness
    - mixup_training: MixUp data augmentation
    - cutmix_training: CutMix data augmentation
    - early_stopping: Adaptive training with early stopping
    - lr_finder: Learning rate range test
    - multitask_learning: Uncertainty-weighted multi-task learning
    - knowledge_distillation: Teacher-student model training
    - gradient_accumulation: Training with large effective batch sizes
    - transfer_learning: Fine-tuning pretrained models
    - curriculum_learning: Difficulty-based curriculum training
    - focal_loss_imbalanced: Focal loss for class imbalance
    - label_smoothing: Label smoothing regularization
    - gradient_clipping: Gradient clipping for stable training
    - pseudo_labeling: Semi-supervised learning with pseudo labels
    - stochastic_weight_averaging: SWA for better generalization
    - progressive_resizing: Multi-scale training with increasing resolution

Metric Learning:
    - siamese_mnist: Siamese network for one-shot learning
    - triplet_loss: Triplet loss for embedding learning

Self-Supervised Learning:
    - simclr_cifar10: SimCLR contrastive learning

Generative Models:
    - autoencoder_mnist: Autoencoder and VAE on MNIST
    - dcgan_mnist: Deep Convolutional GAN
    - conditional_gan: Class-conditional GAN

Language Modeling:
    - gpt_language_model: Character-level GPT on Shakespeare

Tabular Data:
    - tabular_dnn: DNN for structured data (classification/regression)

Time Series:
    - time_series_forecasting: Neural network forecasting with LSTM/Transformer/MLP

Ensemble Learning:
    - model_ensemble: Combining multiple models for improved predictions

Anomaly Detection:
    - anomaly_detection: Autoencoder-based anomaly detection
"""

from . import (
    mnist,
    cifar10,
    fashion_mnist,
    vit_cifar10,
    wideresnet_cifar10,
    unet_segmentation,
    super_convergence,
    fgsm_adversarial,
    mixup_training,
    cutmix_training,
    early_stopping,
    lr_finder,
    multitask_learning,
    knowledge_distillation,
    gradient_accumulation,
    transfer_learning,
    curriculum_learning,
    focal_loss_imbalanced,
    label_smoothing,
    gradient_clipping,
    pseudo_labeling,
    stochastic_weight_averaging,
    progressive_resizing,
    siamese_mnist,
    triplet_loss,
    simclr_cifar10,
    autoencoder_mnist,
    dcgan_mnist,
    conditional_gan,
    gpt_language_model,
    tabular_dnn,
    time_series_forecasting,
    model_ensemble,
    anomaly_detection,
)

__all__ = [
    # Image Classification
    "mnist",
    "cifar10",
    "fashion_mnist",
    "vit_cifar10",
    "wideresnet_cifar10",
    # Image Segmentation
    "unet_segmentation",
    # Training Techniques
    "super_convergence",
    "fgsm_adversarial",
    "mixup_training",
    "cutmix_training",
    "early_stopping",
    "lr_finder",
    "multitask_learning",
    "knowledge_distillation",
    "gradient_accumulation",
    "transfer_learning",
    "curriculum_learning",
    "focal_loss_imbalanced",
    "label_smoothing",
    "gradient_clipping",
    "pseudo_labeling",
    "stochastic_weight_averaging",
    "progressive_resizing",
    # Metric Learning
    "siamese_mnist",
    "triplet_loss",
    # Self-Supervised Learning
    "simclr_cifar10",
    # Generative Models
    "autoencoder_mnist",
    "dcgan_mnist",
    "conditional_gan",
    # Language Modeling
    "gpt_language_model",
    # Tabular Data
    "tabular_dnn",
    # Time Series
    "time_series_forecasting",
    # Ensemble Learning
    "model_ensemble",
    # Anomaly Detection
    "anomaly_detection",
]
