from abc import ABC, abstractmethod
from typing import Dict

import torch
from coqpit import Coqpit

# trainer import removed!

class BaseTrainerModel(ABC):
    """Dummy BaseTrainerModel to replace TrainerModel from trainer package.

    This ensures TTS can work for inference without needing the full trainer library.
    """

    @staticmethod
    @abstractmethod
    def init_from_config(config: Coqpit):
        """Initialize model from config."""
        pass

    @abstractmethod
    def inference(self, input: torch.Tensor, aux_input={}) -> Dict:
        """Dummy inference method."""
        pass

    @abstractmethod
    def load_checkpoint(
        self, config: Coqpit, checkpoint_path: str, eval: bool = False, strict: bool = True, cache=False
    ) -> None:
        """Dummy checkpoint loader."""
        pass
