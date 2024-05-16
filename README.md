# 5th CLVision Workshop @ CVPR 2024 Challenge
# continual-learning-challenge-2024

5th CLVISION CVPR Challenge: https://sites.google.com/view/clvision2024/challenge

Strategies were be submitted at the CodaLab competition: https://codalab.lisn.upsaclay.fr/competitions/17780

This repository is based on the challenge DevKit: https://github.com/ContinualAI/clvision-challenge-2024

## Challenge Overview
It is fair to assume that data is not cheap to acquire, store and label in real-world machine learning applications. Therefore, it is crucial to develop strategies that are flexible enough to learn from streams of experiences, without forgetting what has been learned previously. Additionally, contextual unlabelled data can also be exploited to integrate additional information into the model.

## Our Approach
In this repository, we extend the baseline strategy, which already employs the well-known Learning without Forgetting (LWF) approach for the unlabeled data stream. We also apply the LWF strategy to the labelled data stream. 
Learning without Forgetting (LWF) enables a model to learn new tasks while retaining knowledge of previous tasks by using knowledge distillation to minimize changes in the model's outputs for old task.

Additionally, we extract features from labelled data stream and generate a prototype (class mean) for each class. For unlabelled data, we assign a pseudo-label to each sample based on the highest similarity between its extracted features and the prototypes. If this similarity value does not exceeds a specific threshold, we do not assign a pseudo-label to the sample, as it may belong to an unseen class or a distractor class (a class not present in the labeled data). This approach allows us to leverage previously seen classes from the labelled data stream when encountering them again in the unlabelled data. The prototypes can be stored in a buffer. We have one prototype for each class, so for 100 classes we need a buffer with 100 samples. The size of each prototype is 512 and with the prototype we also store its corresponding label.

We also apply less forgetful learning (LFL) strategy to both labelled and unlabelled data streams.
Less forgetful learning (LFL) leverages features extracted from both previous and current models to maintain performance and minimize changes in the model's features during the learning process.

Finally, we use learning rate scheduler and early stopping to restore the best model. We limit the number of training epochs to 15 to prevent overfitting to current classes.

## Results
baseline_clvision24: 
Accuracy s1: 0.0954, 
Accuracy s2: 0.1066, 
Accuracy s3: 0.0796, 
Avg. Convergence Rate: 0.007, 
**Average Accuracy: 0.0939**
	    	                	             	             
our_approach: 
Accuracy s1: 0.1660 (+7.06%), 
Accuracy s2: 0.1902 (+8.36%), 
Accuracy s3: 0.1442 (+6.46%), 
Avg. Convergence Rate: 0.007, 
**Average Accuracy: 0.1668 (+7.29%)**

