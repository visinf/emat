from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from utils.evaluation import Evaluator


class TBLogger(TensorBoardLogger):
    """Wrapper of the TensorBoardLogger."""
    
    @rank_zero_only
    def save(self) -> None:
        """Flush pending events to disk multi-GPU safe)."""
        self.experiment.flush()


class MetricsCallback(Callback):
    """Attach an `utils.evaluation.Evaluator` to each execution stage."""
    
    def setup(self, trainer, pl_module, stage) -> None:
        """Called once per module; allocate evaluator placeholders."""
        
        pl_module.evaluator = {"trn": None, "val": None, "test": None}
        
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Reset train evaluator and switch module to train mode."""
        
        print(f'\n\n----- Epoch: {trainer.current_epoch:>3} -----')
        pl_module.evaluator["trn"] = Evaluator(trainer.train_dataloader.dataset)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Reset validation evaluator and switch module to eval mode."""
        
        pl_module.evaluator["val"] = Evaluator(trainer.val_dataloaders.dataset)
        pl_module.eval()
        
    def on_test_epoch_start(self, trainer, pl_module) -> None:
        """Reset test evaluator and augmented-task counter, and switch module to 
        eval mode.
        """
        
        pl_module.evaluator["test"] = Evaluator(trainer.test_dataloaders.dataset)
        pl_module.augmented_tasks = 0
        pl_module.eval()
