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

#### Gold Standard

1. resnet50 model trained on CIFAR10 gold standard forgotten class 0

    - Test Results: {'accuracy': 72.8, 'precision': 0.6661484145483357, 'recall': 0.728, 'f1_score': 0.6908683906946586, 'loss': 2.4389873667608333}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 16.98262667655945}

    - Test Retain Results: {'accuracy': 80.88888888888889, 'precision': 0.8169013096795986, 'recall': 0.8088888888888889, 'f1_score': 0.8095607726569405, 'loss': 0.8147816498514632}

    - The MIA has an accuracy of 0.940 on forgotten vs unseen images.

2. QUANTIZED fp8 resnet50 model trained on CIFAR10 gold standard forgotten class 0

    - Test Results: {'accuracy': 62.83, 'precision': 0.6041849784640573, 'recall': 0.6283, 'f1_score': 0.6004438906354465, 'loss': 2.670265721369393}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 15.607704877853394}

    - Test Retain Results: {'accuracy': 69.81111111111112, 'precision': 0.7266922075634171, 'recall': 0.6981111111111111, 'f1_score': 0.700203596230793, 'loss': 1.2235059864084485}

3. QUANTIZED int8 resnet50 model trained on CIFAR10 gold standard forgotten class 0

    - Test Results: {'accuracy': 62.970000000000006, 'precision': 0.6022535282042252, 'recall': 0.6297, 'f1_score': 0.6010157369415328, 'loss': 2.675068974494934}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 15.731910705566406}

    - Test Retain Results: {'accuracy': 69.96666666666667, 'precision': 0.7253623621143, 'recall': 0.6996666666666667, 'f1_score': 0.7011140151754104, 'loss': 1.2156921583162228}

#### Other models

1. resent50 Model trained on CIFAR10 - Accuracy: 81.24%, Precision: 0.8186, Recall: 0.8124, F1 Score: 0.8109, Average Loss: 0.6305

2. QUANTIZED fp8 resnet50 Model trained on CIFAR10 - Accuracy: 70.15%, Precision: 0.7333, Recall: 0.7015, F1 Score: 0.6984, Average Loss: 1.0665

3. QUANTIZED int8 resnet50 Model trained on CIFAR10 - Accuracy: 70.38%, Precision: 0.7339, Recall: 0.7038, F1 Score: 0.7003, Average Loss: 1.0615

4. UNLEARNT resnet50 model trained on CIFAR10 
    - Test Results: {'accuracy': 72.98, 'precision': 0.6584401750065193, 'recall': 0.7298, 'f1_score': 0.6911992756246189, 'loss': 4.77338171910636}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 39.54209756851196}

    - Test Retain Results: {'accuracy': 81.08888888888889, 'precision': 0.8162964316212271, 'recall': 0.8108888888888889, 'f1_score': 0.8114517270498243, 'loss': 0.8791192509758641}

    - The MIA has an accuracy of 0.926 on forgotten vs unseen images

5. UNLEARNT + QUANTIZED fp8 resnet50 Model trained on CIFAR10

    - Test Results: {'accuracy': 64.24, 'precision': 0.6005127875248849, 'recall': 0.6424, 'f1_score': 0.6102937957155696, 'loss': 5.207106590270996}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 39.122621059417725}

    - Test Retain Results: {'accuracy': 71.37777777777778, 'precision': 0.7353187819210352, 'recall': 0.7137777777777777, 'f1_score': 0.7146639978560649, 'loss': 1.4035379374530954}

6. UNLEARNT + QUANTIZED int8 resnet50 model trained on CIFAR10 

    - Test Results: {'accuracy': 64.25999999999999, 'precision': 0.6017066186845085, 'recall': 0.6426, 'f1_score': 0.6107173739452252, 'loss': 5.305596496485457}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 40.02132511138916}

    - Test Retain Results: {'accuracy': 71.39999999999999, 'precision': 0.7363433632246896, 'recall': 0.714, 'f1_score': 0.7150294010352782, 'loss': 1.4128570842071317}
