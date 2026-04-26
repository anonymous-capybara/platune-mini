"""
PyTorch Lightning callbacks for training.
"""

import pytorch_lightning as pl


class SimilarityThresholdCallback(pl.Callback):
    """
    Curriculum learning callback that anneals the cosine similarity threshold
    for grain pair sampling using exponential decay.
    
    The threshold starts at `start_threshold` and decays towards `end_threshold`
    over the course of training using the formula:
        threshold(epoch) = end + (start - end) * decay^epoch
    
    This creates a curriculum where early training sees "easy" pairs (high similarity)
    and later training sees more diverse pairs (lower similarity requirement).
    
    Also logs the current threshold and rejection rate to TensorBoard.
    """
    
    def __init__(
        self,
        start_threshold: float = 0.8,
        end_threshold: float = 0.0,
        decay_rate: float = 0.9,
        log_to_tensorboard: bool = True,
    ):
        """
        Args:
            start_threshold: Initial cosine similarity threshold (epoch 0).
                Higher = more similar pairs required initially.
            end_threshold: Final cosine similarity threshold to decay towards.
                Set to 0 for no constraint at convergence.
            decay_rate: Exponential decay rate per epoch (0 < decay < 1).
                Lower = faster decay. E.g., 0.9 means ~35% of (start-end) 
                remains after 10 epochs.
            log_to_tensorboard: Whether to log threshold and rejection rate.
        """
        super().__init__()
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.decay_rate = decay_rate
        self.log_to_tensorboard = log_to_tensorboard
        
    def _compute_threshold(self, epoch: int) -> float:
        """Compute threshold at given epoch using exponential decay."""
        return self.end_threshold + (self.start_threshold - self.end_threshold) * (self.decay_rate ** epoch)
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Update similarity threshold at the start of each training epoch."""
        # Get the training dataset
        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return
            
        # Handle different dataloader wrapper types
        dataset = train_dataloader.dataset
        if hasattr(dataset, 'dataset'):
            # Handle Subset wrapper
            dataset = dataset.dataset
        
        # Check if this is a GrainPairDataset with similarity threshold support
        if not hasattr(dataset, 'set_similarity_threshold'):
            return
        
        # Compute and set new threshold
        epoch = trainer.current_epoch
        threshold = self._compute_threshold(epoch)
        dataset.set_similarity_threshold(threshold)
        
        # Reset rejection statistics for this epoch
        if hasattr(dataset, 'reset_rejection_stats'):
            dataset.reset_rejection_stats()
        
        # Log to console
        print(f"[SimilarityThresholdCallback] Epoch {epoch}: threshold = {threshold:.4f}")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log rejection rate at the end of each training epoch."""
        if not self.log_to_tensorboard:
            return
            
        # Get the training dataset
        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return
            
        dataset = train_dataloader.dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        
        # Check if this dataset supports rejection stats
        if not hasattr(dataset, 'get_rejection_rate'):
            return
        
        epoch = trainer.current_epoch
        threshold = self._compute_threshold(epoch)
        rejection_rate = dataset.get_rejection_rate()
        avg_attempts = dataset.get_avg_attempts_per_sample() if hasattr(dataset, 'get_avg_attempts_per_sample') else 1.0
        
        # Log to TensorBoard via the model's logger
        if pl_module.logger is not None:
            pl_module.log('curriculum/similarity_threshold', threshold, on_epoch=True)
            pl_module.log('curriculum/rejection_rate', rejection_rate, on_epoch=True)
            pl_module.log('curriculum/avg_attempts_per_sample', avg_attempts, on_epoch=True)
            
            # Also log directly to TensorBoard if available
            if hasattr(pl_module.logger, 'experiment'):
                pl_module.logger.experiment.add_scalar(
                    'curriculum/similarity_threshold', threshold, trainer.global_step
                )
                pl_module.logger.experiment.add_scalar(
                    'curriculum/rejection_rate', rejection_rate, trainer.global_step
                )
                pl_module.logger.experiment.add_scalar(
                    'curriculum/avg_attempts_per_sample', avg_attempts, trainer.global_step
                )
        
        print(f"[SimilarityThresholdCallback] Epoch {epoch} end: "
              f"threshold={threshold:.4f}, rejection_rate={rejection_rate:.2%}, avg_attempts={avg_attempts:.2f}")
