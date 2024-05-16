import operator
from typing import Optional
import warnings
from copy import deepcopy

from avalanche.training.plugins import SupervisedPlugin


class EarlyStoppingPlugin(SupervisedPlugin):
    """Early stopping and model checkpoint plugin.

    The plugin checks a metric and stops the training loop when the accuracy
    on the metric stopped progressing for `patience` epochs.
    After training, the best model's checkpoint is loaded.

    .. warning::
        The plugin checks the metric value, which is updated by the strategy
        during the evaluation. This means that you must ensure that the
        evaluation is called frequently enough during the training loop.

        For example, if you set `patience=1`, you must also set `eval_every=1`
        in the `BaseTemplate`, otherwise the metric won't be updated after
        every epoch/iteration. Similarly, `peval_mode` must have the same
        value.

    """

    def __init__(
        self,
        patience: int,
        val_stream_name: str,
        metric_name: str = "Top1_Acc_Stream",
        mode: str = "max",
        peval_mode: str = "epoch",
        margin: float = 0.0,
        verbose=False,
        train_epochs=None
        
    ):
        """Init.

        :param patience: Number of epochs to wait before stopping the training.
        :param val_stream_name: Name of the validation stream to search in the
            metrics. The corresponding stream will be used to keep track of the
            evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
            reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
            given metric should me maximized (resp. minimized).
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            early stopping should happen after `patience`
            epochs or iterations (Default='epoch').
        :param margin: a minimal margin of improvements required to be
            considered best than a previous one. It should be an float, the
            default value is 0. That means that any improvement is considered
            better.
        :param verbose: If True, prints a message for each update
            (default: False).
        """
        super().__init__()
        # print("Early stopping is used!")
        self.val_stream_name = val_stream_name
        self.patience = patience
        self.verbose = verbose
        self.dif = None
        self.curr_step = None
        self.train_epochs = train_epochs

        assert peval_mode in {"epoch", "iteration"}
        self.peval_mode = peval_mode

        assert type(margin) == float
        self.margin = margin

        self.metric_name = metric_name
        self.metric_key = f"{self.metric_name}/train_phase/" f"{self.val_stream_name}"
        # print(self.metric_key)

        if mode not in ("max", "min"):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == "max" else operator.lt
        
        if mode == "max":
            self.value = 0
        else:
            self.value = 1000

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step: Optional[int] = None

    def before_training_exp(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None

    def before_training_epoch(self, strategy, **kwargs):
        if self.peval_mode == "epoch":
            ub = self._update_best(strategy)
            
            if ub is None or self.best_step is None:
                return
            self.curr_step = self._get_strategy_counter(strategy)
            self.dif = self.curr_step - self.best_step
            if self.dif > self.patience-1:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()
                print(f"Best model restored from epoch {self.best_step-1} with Top1_Acc_Epoch {self.best_val}")
                
            if self.curr_step == self.train_epochs-1:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()
                print(f"Best model restored from epoch {self.best_step-1} with Top1_Acc_Epoch {self.best_val}")
                
    def after_training_epoch(self, strategy, **kwargs):
        if self.dif is not None:
            if self.dif > self.patience-1:
                self.best_val = self.value
                
        if self.curr_step is not None:
            if self.curr_step == self.train_epochs-1:
                self.best_val = self.value

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        names = [k for k in res.keys() if k.startswith(self.metric_key)]
        # print(res)
        if len(names) == 0:
            return None

        full_name = names[-1]
        val_acc = res.get(full_name)

        if self.best_val is None:
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            self.best_step = self._get_strategy_counter(strategy)
            return None

        delta_val = float(val_acc - self.best_val)
        if self.operator(delta_val, 0) and abs(delta_val) >= self.margin:
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            
            self.best_step = self._get_strategy_counter(strategy)
            if self.verbose:
                print("EarlyStopping: new best value:", val_acc)

        return self.best_val

    def _get_strategy_counter(self, strategy):
        if self.peval_mode == "epoch":
            # print(strategy.clock.train_exp_epochs)
            return strategy.clock.train_exp_epochs
        else:
            raise ValueError("Invalid `peval_mode`:", self.peval_mode)
