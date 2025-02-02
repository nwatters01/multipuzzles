"""Config."""

import torch
import trainer as trainer_lib
from models import handcrafted


def get_trainer():
    """Return configuration."""
    trainer = trainer_lib.Trainer(
        generator=handcrafted.HandcraftedGenerator(
            square_size=50,
            circle_radius=30,
            puzzle_log_dir="../logs/square_5x5_v0",
            arrangement_indices=[0, 1],
            image_sizes=[(100, 100), (100, 100)],
        ),
        optimizer=torch.optim.RMSprop,
        optimizer_kwargs={"lr": 0.01},
        train_steps=500,
        scalar_eval_period=50,
    )
    return trainer
