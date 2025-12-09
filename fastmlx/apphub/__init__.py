"""Example applications built with fastmlx.

Categories:

Image Classification:
    - image_classification.mnist: MNIST digit classification with LeNet
    - image_classification.cifar10: CIFAR-10 classification with ResNet9
    - image_classification.fashion_mnist: Fashion-MNIST classification with LeNet
    - image_classification.vit_cifar10: CIFAR-10 with Vision Transformer
    - image_classification.wideresnet_cifar10: CIFAR-10 with WideResNet variants

Semantic Segmentation:
    - semantic_segmentation.unet_segmentation: UNet on synthetic circular mask data

Image Generation:
    - image_generation.autoencoder_mnist: Autoencoder and VAE on MNIST
    - image_generation.dcgan_mnist: Deep Convolutional GAN
    - image_generation.conditional_gan: Class-conditional GAN

One-Shot Learning:
    - one_shot_learning.siamese_mnist: Siamese network for one-shot learning
    - one_shot_learning.triplet_loss: Triplet loss for embedding learning

Contrastive Learning:
    - contrastive_learning.simclr_cifar10: SimCLR contrastive learning

NLP:
    - NLP.gpt_language_model: Character-level GPT on Shakespeare

Tabular:
    - tabular.tabular_dnn: DNN for structured data (classification/regression)

Time Series:
    - time_series.time_series_forecasting: Neural network forecasting with LSTM/Transformer/MLP

Anomaly Detection:
    - anomaly_detection.anomaly_detection: Autoencoder-based anomaly detection

Adversarial Training:
    - adversarial_training.fgsm_adversarial: FGSM adversarial training for robustness

Curriculum Learning:
    - curriculum_learning.curriculum_learning: Difficulty-based curriculum training

Multi-Task Learning:
    - multi_task_learning.multitask_learning: Uncertainty-weighted multi-task learning
    - multi_task_learning.knowledge_distillation: Teacher-student model training

Learning Rate Control:
    - lr_controller.super_convergence: 1cycle learning rate policy for fast training
    - lr_controller.lr_finder: Learning rate range test

Regularization & Training Techniques:
    - regularization.mixup_training: MixUp data augmentation
    - regularization.cutmix_training: CutMix data augmentation
    - regularization.early_stopping: Adaptive training with early stopping
    - regularization.focal_loss_imbalanced: Focal loss for class imbalance
    - regularization.label_smoothing: Label smoothing regularization
    - regularization.gradient_clipping: Gradient clipping for stable training
    - regularization.gradient_accumulation: Training with large effective batch sizes
    - regularization.pseudo_labeling: Semi-supervised learning with pseudo labels
    - regularization.stochastic_weight_averaging: SWA for better generalization
    - regularization.progressive_resizing: Multi-scale training with increasing resolution

Ensemble Learning:
    - ensemble.model_ensemble: Combining multiple models for improved predictions
    - ensemble.transfer_learning: Fine-tuning pretrained models
"""

from . import (
    NLP,
    adversarial_training,
    anomaly_detection,
    contrastive_learning,
    curriculum_learning,
    ensemble,
    image_classification,
    image_generation,
    lr_controller,
    multi_task_learning,
    one_shot_learning,
    regularization,
    semantic_segmentation,
    tabular,
    time_series,
)

__all__ = [
    # Categories
    "image_classification",
    "semantic_segmentation",
    "image_generation",
    "one_shot_learning",
    "contrastive_learning",
    "NLP",
    "tabular",
    "time_series",
    "anomaly_detection",
    "adversarial_training",
    "curriculum_learning",
    "multi_task_learning",
    "lr_controller",
    "regularization",
    "ensemble",
]
