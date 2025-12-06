"""Knowledge Distillation example using :mod:`fastmlx`.

Demonstrates training a smaller "student" model to mimic a larger "teacher"
model. The student learns from both the ground truth labels and the soft
predictions of the teacher, often achieving better performance than training
on hard labels alone.

Benefits:
- Model compression (smaller, faster models)
- Better generalization through soft targets
- Transfer of learned representations

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014.
"""

from __future__ import annotations

import argparse
import tempfile

import mlx.core as mx
import mlx.nn as nn

import fastmlx as fe
from fastmlx.architecture import ResNet9
from fastmlx.dataset.data import cifair10
from fastmlx.op import (
    CrossEntropy,
    HorizontalFlip,
    ModelOp,
    Normalize,
    Op,
    PadIfNeeded,
    RandomCrop,
    Sometimes,
    UpdateOp,
)
from fastmlx.schedule import warmup_cosine_decay
from fastmlx.trace.adapt import LRScheduler
from fastmlx.trace.io import BestModelSaver
from fastmlx.trace.metric import Accuracy


class TeacherModelOp(Op):
    """Forward pass through teacher model (no gradients)."""

    def __init__(self, model, inputs: str, outputs: str) -> None:
        super().__init__([inputs], outputs)
        self.model = model

    def forward(self, data, state):
        x = data[0]
        # Teacher inference (no training)
        with mx.no_grad():
            logits = self.model(x)
        return logits


class DistillationLoss(Op):
    """Combined distillation and classification loss.

    Loss = alpha * CE(student, labels) + (1-alpha) * KL(student, teacher)

    The KL divergence is computed on softened probabilities using temperature.

    Args:
        inputs: Tuple of (student_logits, teacher_logits, labels).
        outputs: Output key for loss.
        temperature: Temperature for softening probabilities. Higher = softer.
        alpha: Weight for hard label loss. (1-alpha) for soft label loss.
    """

    def __init__(
        self,
        inputs: tuple,
        outputs: str,
        temperature: float = 4.0,
        alpha: float = 0.1
    ) -> None:
        super().__init__(list(inputs), outputs)
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, data, state):
        student_logits, teacher_logits, labels = data

        # Hard label loss (standard cross-entropy)
        log_probs = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
        if labels.ndim > 1:
            hard_loss = -mx.mean(mx.sum(labels * log_probs, axis=-1))
        else:
            hard_loss = -mx.mean(mx.take_along_axis(
                log_probs, labels[:, None].astype(mx.int32), axis=1
            ))

        # Soft label loss (KL divergence with temperature)
        soft_student = mx.softmax(student_logits / self.temperature, axis=-1)
        soft_teacher = mx.softmax(teacher_logits / self.temperature, axis=-1)

        # KL(teacher || student) = sum(teacher * log(teacher/student))
        kl_loss = mx.mean(mx.sum(
            soft_teacher * (mx.log(soft_teacher + 1e-8) - mx.log(soft_student + 1e-8)),
            axis=-1
        ))
        # Scale by T^2 as per original paper
        kl_loss = kl_loss * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * kl_loss

        return total_loss


def train_teacher(
    epochs: int = 50,
    batch_size: int = 128,
    save_path: str = None,
) -> nn.Module:
    """Train the teacher model (ResNet9).

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        save_path: Optional path to save trained weights.

    Returns:
        Trained teacher model.
    """
    print("Training Teacher Model (ResNet9)...")

    train_data, eval_data = cifair10.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x",
                     mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(inputs="x", outputs="x", min_height=40, min_width=40),
            RandomCrop(inputs="x", outputs="x", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
        ],
    )

    teacher = fe.build(
        model_fn=lambda: ResNet9(input_shape=(3, 32, 32)),
        optimizer_fn="adam"
    )

    network = fe.Network([
        ModelOp(model=teacher, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=teacher, loss_name="ce")
    ])

    steps_per_epoch = 50000 // batch_size
    total_steps = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        LRScheduler(
            model=teacher,
            lr_fn=lambda step: warmup_cosine_decay(
                step, warmup_steps=steps_per_epoch * 5,
                total_steps=total_steps, init_lr=1e-3, min_lr=1e-5
            )
        )
    ]

    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )
    estimator.fit()

    if save_path:
        teacher.save_weights(save_path)
        print(f"Teacher saved to {save_path}")

    return teacher


class SmallCNN(nn.Module):
    """Small CNN student model (much smaller than ResNet9)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x):
        x = self.pool(nn.relu(self.conv1(x)))  # 32->16
        x = self.pool(nn.relu(self.conv2(x)))  # 16->8
        x = self.pool(nn.relu(self.conv3(x)))  # 8->4
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


def get_distillation_estimator(
    teacher: nn.Module,
    epochs: int = 50,
    batch_size: int = 128,
    temperature: float = 4.0,
    alpha: float = 0.1,
    save_dir: str = tempfile.mkdtemp(),
) -> fe.Estimator:
    """Create knowledge distillation estimator.

    Args:
        teacher: Pre-trained teacher model.
        epochs: Number of training epochs.
        batch_size: Batch size.
        temperature: Distillation temperature (higher = softer).
        alpha: Weight for hard labels (1-alpha for soft labels).
        save_dir: Directory to save best student model.

    Returns:
        Configured Estimator ready for training.
    """
    train_data, eval_data = cifair10.load_data()

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=batch_size,
        ops=[
            Normalize(inputs="x", outputs="x",
                     mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2471, 0.2435, 0.2616)),
            PadIfNeeded(inputs="x", outputs="x", min_height=40, min_width=40),
            RandomCrop(inputs="x", outputs="x", height=32, width=32),
            Sometimes(HorizontalFlip(inputs="x", outputs="x")),
        ],
    )

    # Create student model (much smaller)
    student = fe.build(
        model_fn=lambda: SmallCNN(num_classes=10),
        optimizer_fn="adam"
    )

    network = fe.Network([
        # Teacher inference (no gradients)
        TeacherModelOp(model=teacher, inputs="x", outputs="teacher_logits"),
        # Student forward pass
        ModelOp(model=student, inputs="x", outputs="student_logits"),
        # Distillation loss
        DistillationLoss(
            inputs=("student_logits", "teacher_logits", "y"),
            outputs="distill_loss",
            temperature=temperature,
            alpha=alpha
        ),
        UpdateOp(model=student, loss_name="distill_loss")
    ])

    steps_per_epoch = 50000 // batch_size
    total_steps = epochs * steps_per_epoch

    traces = [
        Accuracy(true_key="y", pred_key="student_logits"),
        BestModelSaver(model=student, save_dir=save_dir, metric="accuracy"),
        LRScheduler(
            model=student,
            lr_fn=lambda step: warmup_cosine_decay(
                step, warmup_steps=steps_per_epoch * 5,
                total_steps=total_steps, init_lr=1e-3, min_lr=1e-5
            )
        )
    ]

    return fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation with FastMLX")
    parser.add_argument("--teacher-epochs", type=int, default=50,
                        help="Epochs for teacher training")
    parser.add_argument("--student-epochs", type=int, default=50,
                        help="Epochs for student training")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Weight for hard labels")
    args = parser.parse_args()

    print("Knowledge Distillation: ResNet9 -> SmallCNN")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha (hard label weight): {args.alpha}")

    # Train teacher
    teacher = train_teacher(
        epochs=args.teacher_epochs,
        batch_size=args.batch_size,
    )

    # Distill to student
    print("\nDistilling to Student Model (SmallCNN)...")
    est = get_distillation_estimator(
        teacher=teacher,
        epochs=args.student_epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    est.fit()
    est.test()
