# Can-Vision-Models-Forget

## Mid evaluation tasks

### PIPELINE 1: initial training
**Given:** Model and Dataset
**Output:** OG Model saved

### PIPELINE 2: create unlearnt model
**Given:** Trained Model and Dataset with forget and retain subsets
**Output:** Unlearnt Model saved

### PIPELINE 3: create quantized models
**Given:** Trained Model or Unlearnt Model
**Output:** Quantized Model saved

### PIPELINE 4: evaluation pipeline
**Given:** Model and Dataset
**Output:** Evaluation metrics

### ToDo (6/3/25):

1. Run the quantzation and evaluation on og and unlearnt models (all combinations) and see if we are getting any valid results.
2. Merge the code and make the entire codebase consistent. for instance, there are too many evaluate functions.
3. Make a single file/notebook where the entire pipeline can be run.
    - input: model names, dataset names, unlearning methods as lists
    - output: json files storing all combination evaluations for all of them.

### Evaluation Metrics

1. Original resent50 Model trained on CIFAR10 - Accuracy: 84.21%, Precision: 0.8511, Recall: 0.8421, F1 Score: 0.8432, Average Loss: 0.8791
2. Quantized fp8 resnet50 Model trained on CIFAR10 - Accuracy: 72.64%, Precision: 0.7607, Recall: 0.7264, F1 Score: 0.7288, Average Loss: 1.5751
3. Quantized int8 resnet50 Model trained on CIFAR10 - Accuracy: 72.92%, Precision: 0.7647, Recall: 0.7292, F1 Score: 0.7316, Average Loss: 1.6108
