# 5th CLVision Workshop @ CVPR 2024 Challenge
# continual-learning-challenge-2024

- **Challenge Website**: [CLVision 2024 Challenge](https://sites.google.com/view/clvision2024/challenge) 
- **CodaLab Competition**: [CodaLab Competition Platform](https://codalab.lisn.upsaclay.fr/competitions/17780)   
- **DevKit Repository**: [Challenge DevKit Repository](https://github.com/ContinualAI/clvision-challenge-2024)       
- **Challenge Results**: [Final Challenge Results](https://sites.google.com/view/clvision2024/challenge/challenge-results)   

## Challenge Overview
It is fair to assume that data is not cheap to acquire, store and label in real-world machine learning applications. Therefore, it is crucial to develop strategies that are flexible enough to learn from streams of experiences, without forgetting what has been learned previously. Additionally, contextual unlabelled data can also be exploited to integrate additional information into the model.

## Baseline strategy
A baseline strategy was provided by the organizers of the challenge and applies the well-known Learning without Forgetting (LwF) strategy described in [this paper](https://arxiv.org/abs/1606.09282) to the unlabelled data streams. 

## Team Members:
- Panagiota Moraiti 
- Efstathios Karypidis

## Our Approach
In this repository, we extend the baseline strategy, which already employs the well-known Learning without Forgetting (LWF) approach for the unlabelled data stream. We also apply the LWF strategy to the labelled data stream. 
Learning without Forgetting (LWF) enables a model to learn new tasks while retaining knowledge of previous tasks by using knowledge distillation to minimize changes in the model's outputs for old task.

Additionally, we extract features from labelled data stream and generate a prototype (class mean) for each class. For unlabelled data, we assign a pseudo-label to each sample based on the highest similarity between its extracted features and the prototypes. If this similarity value does not exceeds a specific threshold, we do not assign a pseudo-label to the sample, as it may belong to an unseen class or a distractor class (a class not present in the labeled data). This approach allows us to leverage previously seen classes from the labelled data stream when encountering them again in the unlabelled data. The prototypes can be stored in a buffer. We have one prototype for each class, so for 100 classes we need a buffer with 100 samples. The size of each prototype is 512 and with the prototype we also store its corresponding label.

We also apply less forgetful learning (LFL) strategy to both labelled and unlabelled data streams.
Less forgetful learning (LFL) leverages features extracted from both previous and current models to maintain performance and minimize changes in the model's features during the learning process.

Finally, we use learning rate scheduler and early stopping to restore the best model. We limit the number of training epochs to 15 to prevent overfitting to current classes.

- **Recorded Video of Our Solution**: [Video](https://drive.google.com/file/d/1Z0V8_36qyVzssQAZi36JcE-LokF706G6/view?usp=drive_link)  
- **Presentation Slides**: [Slides](https://drive.google.com/file/d/1Mkacx4i0VreI_Zceod7Jrc4TJJ08VG87/view?usp=drive_link)
- **Paper**: [Technical Report for the 5th CLVision Challenge at CVPR: Addressing the Class-Incremental with Repetition using Unlabeled Data - 4th Place Solution](https://arxiv.org/abs/2503.15697)

## Results	    
| Phase          | Scenario 1              | Scenario 2              | Scenario 3              | Average Accuracy   |
|----------------|-------------------------|-------------------------|-------------------------|--------------------|
| Baseline       | 7.96                    | 10.66                   | 9.54                    | 9.39               |
| Pre-selection  | 14.42 (+6.46%)          | 19.02 (+8.36%)          | 16.60 (+7.06%)          | **16.68 (+7.29%)** |
| Final          | 17.49 (+9.53%)          | 22.44 (+11.78%)         | 23.64 (+14.10%)         | **21.19 (+11.80%)**|

# Citation
If you found this work useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!
```bibtex
@misc{moraiti2025technicalreport5thclvision,
      title={Technical Report for the 5th CLVision Challenge at CVPR: Addressing the Class-Incremental with Repetition using Unlabeled Data -- 4th Place Solution}, 
      author={Panagiota Moraiti and Efstathios Karypidis},
      year={2025},
      eprint={2503.15697},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.15697}, 
}
```

