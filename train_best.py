import argparse
import os
import sys

import torch
import torchvision
from avalanche.models import IncrementalClassifier
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)

from benchmarks.generate_scenario import generate_benchmark
from utils.competition_plugins import (
    GPUMemoryChecker,
    TimeChecker
)

from strategies.my_strategy_best import MyStrategyBest
from utils.generic import set_random_seed, FileOutputDuplicator, evaluate
from utils.short_text_logger import ShortTextLogger

import torch.nn.functional as F

from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import make_classification_dataset

from strategies.early_stopping import EarlyStoppingPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim.lr_scheduler import StepLR


def extract_features(model, input_data, layer_name):
    model.eval()

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
    with torch.no_grad():
        _ = model(input_data)
    
    # Remove the hook
    hook_handle.remove()
    
    return features


def main(args):
    # --- Generate Benchmark
    benchmark = generate_benchmark(args.config_file)

    # --- Setup model and Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    # --- Initialize Model
    set_random_seed()
    model = torchvision.models.resnet18()
    # This classification head increases its size automatically in avalanche with the number of
    # annotated samples. If you modify the network structure adapt accordingly
    model.fc = IncrementalClassifier(512, 2, masking=False)
  
    # --- Logger and metrics
    # Adjust logger to personal taste
    base_results_dir = os.path.join("results", f"{os.path.splitext(args.config_file)[0]}_{args.run_name}")
    os.makedirs(base_results_dir, exist_ok=True)
    preds_file = os.path.join(base_results_dir, f"pred_{args.config_file}")

    sys.stdout = FileOutputDuplicator(sys.stdout, os.path.join(base_results_dir, "log.txt"), "w")
    sys.stderr = FileOutputDuplicator(sys.stderr, os.path.join(base_results_dir, "error.txt"), "w")
    text_logger = ShortTextLogger(file=sys.stdout)
    
    # loggers.append(InteractiveLogger())
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=False, stream=False),
        loss_metrics(minibatch=False, epoch=True, experience=False, stream=False),
        loggers=[text_logger] # TensorboardLogger()
        )

    # --- Competition Plugins -> check
    # DO NOT REMOVE OR CHANGE THESE PLUGINS:
    competition_plugins = [
        GPUMemoryChecker(max_allowed=8000),
        TimeChecker(max_allowed=600)
    ]
    
    patience = 10
    train_epochs = 15 # 20

    optimizer=torch.optim.Adam(model.parameters(), lr=0.0005) # 0.001
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5, verbose=False)
    
    early_stopping_plugin = EarlyStoppingPlugin(patience=patience, val_stream_name='train_stream', margin=0.005, metric_name='Top1_Acc_Epoch', mode='max', verbose=False, train_epochs=train_epochs+1)
    lr_scheduler_plugin = LRSchedulerPlugin(scheduler)

    # --- Your Plugins
    plugins = [
        early_stopping_plugin,
        lr_scheduler_plugin
    ]

    # --- Strategy
    # Implement your own Strategy in MyStrategy and replace this example Approach
    # Uncomment this line to test LwF baseline with unlabelled pool usage
    cl_strategy = MyStrategyBest(model=model,
                                 optimizer=optimizer,
                                 criterion=CrossEntropyLoss(),
                                 train_mb_size=32, # 64
                                 train_epochs=train_epochs+1, 
                                 eval_mb_size=256,
                                 device=device,
                                 plugins=competition_plugins + plugins,
                                 evaluator=eval_plugin)
                                 
    # Initialize dict for prototypes
    buffer_size = 100 # For each class store one sample
    feature_size = 512
    mean_features = {i: torch.zeros(feature_size).to(device) for i in range(buffer_size)}
    
    unl_ds_with_target = None
    features_labelled = None
    features_unlabelled = None
                            
    # --- Sequence of incremental training tasks/experiences
    for exp_idx, (train_exp, unl_ds) in enumerate(zip(benchmark.train_stream, benchmark.unlabelled_stream)):
        # Extract labelled data
        lab_ds = benchmark.train_stream[exp_idx].dataset
        labelled_imgs = [inner_list[0] for inner_list in lab_ds]
        labels = [inner_list[1] for inner_list in lab_ds]
        
        # Extract Unlabelled data
        unlabelled_imgs = [inner_list for inner_list in unl_ds]
        
        # Extract Labelled Features
        tensor_labelled_imgs = torch.stack(labelled_imgs)
        '''features_labelled = extract_features(cl_strategy.model, tensor_labelled_imgs, layer_name='avgpool')
        features_labelled = features_labelled.reshape(len(features_labelled), feature_size)'''
        
        # Extract Unlabelled Features
        tensor_unlabelled_imgs = torch.stack(unlabelled_imgs)
        features_unlabelled = extract_features(cl_strategy.model, tensor_unlabelled_imgs, layer_name='avgpool')
        features_unlabelled = features_unlabelled.reshape(len(features_unlabelled), feature_size)
        # print(features_unlabelled[0].shape)
        # print(len(features_unlabelled))
        
        ### --Assign pseudo labels in unlabeled samples based on cosine similarity
        # Normalize the points to unit length and use cdist
        mean_features_list = list(mean_features.values())
        mean_features_tensor = torch.stack(mean_features_list)
        features_unlabelled_normalized = features_unlabelled / torch.norm(features_unlabelled, dim=1, keepdim=True)
        mean_features_tensor_normalized = mean_features_tensor / torch.norm(mean_features_tensor, dim=1, keepdim=True)

        # Compute pairwise cosine distances between normalized points
        cosine_distances = torch.cdist(features_unlabelled_normalized.to(device), mean_features_tensor_normalized.to(device), p=2)
        cosine_distances[torch.isnan(cosine_distances)] = float('inf')
        # print(cosine_distances.shape) # (1000, 100)
        # print(cosine_distances)
        min_distances, min_indices = torch.min(cosine_distances, dim=1)

        # For scenarios 2 and 3, some samples may not belong to a seen labelled class
        thresh = 0.5
        # Initialize exp_predicted_labels_tensor with -1 (indicating unseen class)
        exp_predicted_labels_tensor = torch.full((min_indices.size(0),), -1, dtype=torch.long).to(device)
        # Check if minimum distances are less than the threshold
        mask = min_distances < thresh
        # Assign the corresponding min_indices to exp_predicted_labels_tensor where mask is True
        exp_predicted_labels_tensor[mask] = min_indices[mask]

        # exp_predicted_labels_tensor = min_indices
        # print(len(exp_predicted_labels_tensor))
        # print(exp_predicted_labels_tensor)
        # print(min_distances)
        
        count = (exp_predicted_labels_tensor != -1).sum().item()
        print("Number of occurrences for pseudo-labels:", count)
        ### --
        
        # Create the Dataset for unlabelled data with pseudo-labels
        torch_data = TensorDataset(tensor_unlabelled_imgs, exp_predicted_labels_tensor)
        tls = [0 for _ in range(len(tensor_unlabelled_imgs))]
        unl_ds_with_target = make_classification_dataset(torch_data, task_labels=tls)
        # print(unl_ds_with_target[0])
        
        # Create the Dataset for labelled data with labels
        labels_tensor = torch.tensor(labels).to(device)
        torch_data = TensorDataset(tensor_labelled_imgs, labels_tensor)
        tls = [0 for _ in range(len(tensor_labelled_imgs))]
        lab_ds_with_target = make_classification_dataset(torch_data, task_labels=tls)
        # print(lab_ds_with_target[0])
        
        # Call training and evaluation         
        cl_strategy.train(train_exp, labelled_ds=labelled_imgs, unlabelled_ds=unl_ds, unlabelled_ds_with_labels=unl_ds_with_target, 
                          labelled_ds_with_labels=lab_ds_with_target, lab_features=None, unl_features=None, prototypes=mean_features, num_workers=args.num_workers)
        evaluate(benchmark.test_stream[0].dataset, cl_strategy.model, device, exp_idx, preds_file)
        
        ####################################################################################################################
        ### --Create class prototypes for each class in current exp, based on both old prototypes and new ones
        # Extract Labelled Features
        # tensor_labelled_imgs = torch.stack(labelled_imgs)
        features_labelled = extract_features(cl_strategy.model, tensor_labelled_imgs, layer_name='avgpool')
        features_labelled = features_labelled.reshape(len(features_labelled), feature_size)
        # print(features_labelled[0].shape)
        # print(len(features_labelled))
        # print(features_labelled[0])
        
        # Initialize a dictionary to hold accumulated features and counts for each class
        class_features = {i: torch.zeros(feature_size).to(device) for i in range(buffer_size)}
        class_counts = {i: 0 for i in range(buffer_size)}
        
        # Iterate over labels and corresponding features
        for label, feature in zip(labels, features_labelled):
            # Accumulate features for each class
            class_features[label] += feature.to(device) # For each class seen accumulate features
            class_counts[label] += 1
        
        # Calculate mean feature for each class
        for label in class_features:
            if class_counts[label] > 0: # For each class seen in this exp calculate mean feature
                if torch.sum(mean_features[label]) == 0: # If mean feature for this class is not stored
                    mean_features[label] = class_features[label] / class_counts[label]
                else: # If mean feature for this class is stored calculate the mean of the two mean features
                    old_mean = mean_features[label]
                    current_mean = class_features[label] / class_counts[label]
                    mean_features[label] = (old_mean + current_mean) / 2
                # print(label)
                # print(class_counts[label])

        # print(mean_features[0].shape)
        # print(mean_features)
        ### --
        ####################################################################################################################
        
    print(f"Predictions saved in {preds_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="scenario_1.pkl")
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)
