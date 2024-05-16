# Implement your strategy in this file
import copy
from typing import List, Optional, Iterator

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from torch.utils.data import DataLoader

from strategies.competition_template import CompetitionTemplate


def extract_features(model, input_data, layer_name):
    # model.eval()

    # Move input data to the same device as the model
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    # Define a hook to extract features from the specified layer
    features = None
    def hook(module, input, output):
        nonlocal features
        features = output
    
    # Register the hook to the specified layer
    target_layer = dict(model.named_modules())[layer_name]
    hook_handle = target_layer.register_forward_hook(hook)
    
    # Forward pass
    # with torch.no_grad():
    _ = model(input_data)

    # Remove the hook
    hook_handle.remove()
    
    return features

class MyStrategyBest(CompetitionTemplate):
    """
    Implemention of MyStrategy.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion=CrossEntropyLoss(), train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: Optional[int] = 1, device="cpu",
                 plugins: Optional[List[SupervisedPlugin]] = None, evaluator=default_evaluator(), eval_every=-1,
                 peval_mode="epoch", ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device,
                         plugins, evaluator, eval_every, peval_mode, )
        self.old_model: Optional[Module] = None
        
        self.unlabelled_dl: Optional[DataLoader] = None
        self.unlabelled_iterator: Optional[Iterator] = None
        
        # Labelled data
        self.labelled_dl: Optional[DataLoader] = None
        self.labelled_iterator: Optional[Iterator] = None
        
        # With pseudo-labels
        self.unlabelled_dl_with_labels: Optional[DataLoader] = None
        self.unlabelled_iterator_with_labels: Optional[Iterator] = None
        
        # With labels
        self.labelled_dl_with_labels: Optional[DataLoader] = None
        self.labelled_iterator_with_labels: Optional[Iterator] = None
        
        # Features
        self.lab_features_dl: Optional[DataLoader] = None
        self.lab_features_iterator: Optional[Iterator] = None
        self.unl_features_dl: Optional[DataLoader] = None
        self.unl_features_iterator: Optional[Iterator] = None
        
        # Weights
        self.unlabelled_alpha = 2.0 # 1.0
        self.labelled_alpha = 2.0
        self.pseudo_labels_alpha = 0.25 # Less than the other losses, because pseudo labels may be wrong
        self.lfl_weight = 1000

    def _before_training_exp(self, **kwargs):
        # Callback before every task
        # Do not remove super call as plugins might not work properly otherwise
        super()._before_training_exp(**kwargs)
        # Create Dataloader for Knowledge Distillation of unlabelled samples!
        if self.old_model is not None:
            self.unlabelled_dl = DataLoader(self.unlabelled_ds, batch_size=self.train_mb_size*2, shuffle=True)
            self.unlabelled_iterator = iter(self.unlabelled_dl)
            # Labelled data
            self.labelled_dl = DataLoader(self.labelled_ds, batch_size=self.train_mb_size, shuffle=True)
            self.labelled_iterator = iter(self.labelled_dl)
            
            if self.unlabelled_ds_with_labels is not None:
                # Pseudo-labels
                self.unlabelled_dl_with_labels = DataLoader(self.unlabelled_ds_with_labels, batch_size=self.train_mb_size*2, shuffle=True)
                self.unlabelled_iterator_with_labels = iter(self.unlabelled_dl_with_labels)
                
            if self.labelled_ds_with_labels is not None:
                # Labels
                self.labelled_dl_with_labels = DataLoader(self.labelled_ds_with_labels, batch_size=self.train_mb_size, shuffle=True)
                self.labelled_iterator_with_labels = iter(self.labelled_dl_with_labels)
                
            if self.lab_features is not None:
                self.lab_features_dl = DataLoader(self.lab_features, batch_size=self.train_mb_size, shuffle=True)
                self.lab_features_iterator = iter(self.lab_features_dl)
                
            if self.unl_features is not None:
                self.unl_features_dl = DataLoader(self.unl_features, batch_size=self.train_mb_size*2, shuffle=True)
                self.unl_features_iterator = iter(self.unl_features_dl)
   
    def training_epoch(self, **kwargs):
        # You can implement your custom training loop here
        # Defaults to a base training loop (see SGDUpdate class)
        # Be careful to add all necessary plugin calls as Avalanche Plugins could not work correctly otherwise!
        super().training_epoch(**kwargs)

    def forward(self):
        # If you need to adjust the forward pass for your model during training
        # you can adjust it here
        return super().forward()

    def criterion(self):
        # Implement your own loss criterion here if needed
        # By default self._criterion gets called which defaults to the CrossEntropy loss
        return super().criterion()

    def _before_backward(self, **kwargs):
        # triggers before backpropagation of the calculated loss from criterion.
        # You can add additional loss terms here. (modify self.loss += ...)
        # For example weight regularization or knowledge distillation
        super()._before_backward(**kwargs)

        if self.old_model is not None and self.unlabelled_iterator is not None:
            try:
                batch_unlabelled = next(self.unlabelled_iterator)
            except StopIteration:
                try:
                    self.unlabelled_iterator = iter(self.unlabelled_dl)
                    batch_unlabelled = next(self.unlabelled_iterator)
                except Exception:
                    batch_unlabelled = None
                    self.unlabelled_iterator = None

            if batch_unlabelled is not None:
                batch_unlabelled = batch_unlabelled.to(self.device)
                pred_old_model = self.old_model(batch_unlabelled)
                old_size = pred_old_model.size(1)
                pred_current_model = self.model(batch_unlabelled)[:, :old_size]
                # print(len(pred_current_model))

                def _distillation_loss(out, target, temperature):
                    log_p = torch.log_softmax(out / temperature, dim=1)
                    q = torch.softmax(target / temperature, dim=1)
                    return torch.nn.functional.kl_div(log_p, q, reduction="batchmean")

                unlabelled_loss = self.unlabelled_alpha * _distillation_loss(pred_current_model, pred_old_model, 2.0)
                # print("LWF unlabelled loss: ", unlabelled_loss)
                self.loss += unlabelled_loss
                
        # Labelled data    
        if self.old_model is not None and self.labelled_iterator is not None:
            try:
                batch_labelled = next(self.labelled_iterator)
            except StopIteration:
                try:
                    self.labelled_iterator = iter(self.labelled_dl)
                    batch_labelled = next(self.labelled_iterator)
                except Exception:
                    batch_labelled = None
                    self.labelled_iterator = None

            if batch_labelled is not None:
                batch_labelled = batch_labelled.to(self.device)
                pred_old_model = self.old_model(batch_labelled)
                old_size = pred_old_model.size(1)
                pred_current_model = self.model(batch_labelled)[:, :old_size]
                # print(len(pred_current_model))

                def _distillation_loss(out, target, temperature):
                    log_p = torch.log_softmax(out / temperature, dim=1)
                    q = torch.softmax(target / temperature, dim=1)
                    return torch.nn.functional.kl_div(log_p, q, reduction="batchmean")

                labelled_loss = self.labelled_alpha * _distillation_loss(pred_current_model, pred_old_model, 2.0)
                # print("LWF labelled loss: ", labelled_loss)
                self.loss += labelled_loss
                
        # Pseudo-labels for unlabelled data
        if self.unlabelled_iterator_with_labels is not None:
            try:
                batch_unlabelled = next(self.unlabelled_iterator_with_labels)
            except StopIteration:
                try:
                    self.unlabelled_iterator_with_labels = iter(self.unlabelled_dl_with_labels)
                    batch_unlabelled = next(self.unlabelled_iterator_with_labels)
                except Exception:
                    batch_unlabelled = None
                    self.unlabelled_iterator_with_labels = None

            if batch_unlabelled is not None:
                # batch_unlabelled = batch_unlabelled.to(self.device)
                imgs = batch_unlabelled[0].to(self.device)
                pseudo_labels = batch_unlabelled[1].to(self.device)

                # Calculate cross-entropy loss using pseudo labels
                pred = self.model(imgs)
                # print(len(pred))
                temp = 2.0
                pred = pred / temp
                pseudo_labels_loss = self.pseudo_labels_alpha * torch.nn.functional.cross_entropy(pred, pseudo_labels, ignore_index=-1)
                # print("Pseudo-labels loss: ", pseudo_labels_loss)
                self.loss += pseudo_labels_loss
                
                ### --Add feature extraction
                feature_size = 512
                features_batch_unlabelled = extract_features(self.model, imgs, layer_name='avgpool')
                features_batch_unlabelled = features_batch_unlabelled.reshape(len(features_batch_unlabelled), feature_size)
                features_batch_unlabelled_norm = features_batch_unlabelled / torch.norm(features_batch_unlabelled, dim=1, keepdim=True)
       
                ### --Less Forgetfull learning
                prev_features_batch_unlabelled = extract_features(self.old_model, imgs, layer_name='avgpool')
                prev_features_batch_unlabelled = prev_features_batch_unlabelled.reshape(len(prev_features_batch_unlabelled), feature_size)
                prev_features_batch_unlabelled_norm = prev_features_batch_unlabelled / torch.norm(prev_features_batch_unlabelled, dim=1, keepdim=True)
                
                lfl_loss = torch.nn.functional.mse_loss(features_batch_unlabelled_norm, prev_features_batch_unlabelled_norm) * self.lfl_weight
                self.loss += lfl_loss
                
                
        # Labels for labelled data
        if self.labelled_iterator_with_labels is not None:
            try:
                batch_labelled = next(self.labelled_iterator_with_labels)
            except StopIteration:
                try:
                    self.labelled_iterator_with_labels = iter(self.labelled_dl_with_labels)
                    batch_labelled = next(self.labelled_iterator_with_labels)
                except Exception:
                    batch_labelled = None
                    self.labelled_iterator_with_labels = None

            if batch_labelled is not None:
                # batch_labelled = batch_labelled.to(self.device)
                imgs = batch_labelled[0].to(self.device)
                labels = batch_labelled[1].to(self.device)

                ### --Add feature extraction
                feature_size = 512
                features_batch_labelled = extract_features(self.model, imgs, layer_name='avgpool')
                features_batch_labelled = features_batch_labelled.reshape(len(features_batch_labelled), feature_size)
                features_batch_labelled_norm = features_batch_labelled / torch.norm(features_batch_labelled, dim=1, keepdim=True)
                          
                ### --Less Forgetfull learning
                prev_features_batch_labelled = extract_features(self.old_model, imgs, layer_name='avgpool')
                prev_features_batch_labelled = prev_features_batch_labelled.reshape(len(prev_features_batch_labelled), feature_size)
                prev_features_batch_labelled_norm = prev_features_batch_labelled / torch.norm(prev_features_batch_labelled, dim=1, keepdim=True)
                
                lfl_loss = torch.nn.functional.mse_loss(features_batch_labelled_norm, prev_features_batch_labelled_norm) * self.lfl_weight
                self.loss += lfl_loss

	   
    def _after_training_exp(self, **kwargs):
        # Callback after every training task
        # Do not remove super call as plugins might not work properly otherwise
        super()._after_training_exp(**kwargs)

        # Copy Old Model and Freeze
        self.old_model = copy.deepcopy(self.model)
        # Freeze the old_model
        self.old_model.requires_grad_(False)
        self.old_model.eval()
           
