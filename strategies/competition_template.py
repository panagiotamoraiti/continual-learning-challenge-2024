from typing import Union, Iterable, Optional, Sequence, Callable

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.core import BasePlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.base_sgd import TDatasetExperience
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer


class CompetitionTemplate(SupervisedTemplate):

    def __init__(self, model: Module, optimizer: Optimizer, criterion=CrossEntropyLoss(), train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: Optional[int] = 1, device: Union[str, torch.device] = "cpu",
                 plugins: Optional[Sequence[BasePlugin]] = None,
                 evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = default_evaluator,
                 eval_every=-1, peval_mode="epoch",):
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device, plugins,
                         evaluator, eval_every, peval_mode)

        self.unlabelled_ds: Optional[AvalancheDataset] = None
        # Labelled data
        self.labelled_ds: Optional[AvalancheDataset] = None

    def train(self, experiences: Union[TDatasetExperience, Iterable[TDatasetExperience]],
              labelled_ds: Union[AvalancheDataset] = None,
              unlabelled_ds: Union[AvalancheDataset] = None,
              unlabelled_ds_with_labels: Union[AvalancheDataset] = None,
              labelled_ds_with_labels: Union[AvalancheDataset] = None,
              lab_features = None,
              unl_features = None,
              prototypes=None,
              eval_streams: Optional[Sequence[Union[TDatasetExperience, Iterable[TDatasetExperience]]]] = None, **kwargs):
        self.unlabelled_ds = unlabelled_ds
        # Labelled data
        self.labelled_ds = labelled_ds
        # With pseudo-labels
        self.unlabelled_ds_with_labels = unlabelled_ds_with_labels
        # With labels
        self.labelled_ds_with_labels = labelled_ds_with_labels
        # Features
        self.lab_features = lab_features
        self.unl_features = unl_features
        self.prototypes = prototypes
        super().train(experiences, eval_streams, **kwargs)
