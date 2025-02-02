"""Trainer class."""

import torch


class Trainer:
    
    def __init__(self,
                 generator,
                 optimizer,
                 optimizer_kwargs,
                 train_steps,
                 scalar_eval_period=100,
                 grad_clip=1.):
        """Initialize the trainer."""
        self._generator = generator
        self._optimizer = optimizer(generator.parameters(), **optimizer_kwargs)
        self._train_steps = train_steps
        self._scalar_eval_period = scalar_eval_period
        self._grad_clip = grad_clip
    
    def __call__(self):
        """Train the generator."""
        for step in range(self._train_steps):
            self._optimizer.zero_grad()
            loss = self._generator.loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._generator.parameters(), self._grad_clip,
            )
            self._optimizer.step()
            
            # Log if necessary
            if step % self._scalar_eval_period == 0:
                print(f"Step: {step}, Loss: {loss.item()}")
        
        # Plot images
        self._generator.plot()
        
    @property
    def generator(self):
        """Return the generator."""
        return self._generator