### Evaluation Metrics

#### Gold Standard

1. resnet50 model trained on CIFAR10 gold standard forgotten class 0

    - Test Results: {'accuracy': 72.8, 'precision': 0.6661484145483357, 'recall': 0.728, 'f1_score': 0.6908683906946586, 'loss': 2.431051226514578}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 16.981738892812576}

    - Test Retain Results: {'accuracy': 80.88888888888889, 'precision': 0.8169013096795986, 'recall': 0.8088888888888889, 'f1_score': 0.8095607726569405, 'loss': 0.814532986766002}

    - The MIA has an accuracy of 0.946 on forgotten vs unseen images

2. QUANTIZED fp8 resnet50 model trained on CIFAR10 gold standard forgotten class 0

    - Test Results: {'accuracy': 62.760000000000005, 'precision': 0.602880876780584, 'recall': 0.6276, 'f1_score': 0.5998396877922235, 'loss': 2.6645474528074264}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 15.605862768869551}

    - Test Retain Results: {'accuracy': 69.73333333333333, 'precision': 0.7252459748875426, 'recall': 0.6973333333333334, 'f1_score': 0.6995202394433859, 'loss': 1.2270813956459716}

    - The MIA has an accuracy of 0.944 on forgotten vs unseen images

3. QUANTIZED int8 resnet50 model trained on CIFAR10 gold standard forgotten class 0

    - Test Results: {'accuracy': 63.029999999999994, 'precision': 0.6025996824657538, 'recall': 0.6303, 'f1_score': 0.6014312161298658, 'loss': 2.671693898534775}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 15.73249670058962}

    - Test Retain Results: {'accuracy': 70.03333333333333, 'precision': 0.7257580540753483, 'recall': 0.7003333333333334, 'f1_score': 0.701675998233705, 'loss': 1.220651829883218}

#### Other models

1. resent50 Model trained on CIFAR10 - Accuracy: 81.24%, Precision: 0.8186, Recall: 0.8124, F1 Score: 0.8109, Average Loss: 0.6305

2. QUANTIZED fp8 resnet50 Model trained on CIFAR10 - Accuracy: 70.15%, Precision: 0.7333, Recall: 0.7015, F1 Score: 0.6984, Average Loss: 1.0665

3. QUANTIZED int8 resnet50 Model trained on CIFAR10 - Accuracy: 70.38%, Precision: 0.7339, Recall: 0.7038, F1 Score: 0.7003, Average Loss: 1.0615

4. UNLEARNT resnet50 model trained on CIFAR10 
    - Test Results: {'accuracy': 72.99, 'precision': 0.6585349802615801, 'recall': 0.7299, 'f1_score': 0.6912992167944038, 'loss': 4.747792054724694}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 39.548707386804004}

    - Test Retain Results: {'accuracy': 81.10000000000001, 'precision': 0.8164140869551604, 'recall': 0.811, 'f1_score': 0.8115683005098032, 'loss': 0.8810338847505272}

    - The MIA has an accuracy of 0.931 on forgotten vs unseen images

5. UNLEARNT + QUANTIZED fp8 resnet50 Model trained on CIFAR10

    - Test Results: {'accuracy': 64.16, 'precision': 0.5997927986159782, 'recall': 0.6416, 'f1_score': 0.6093938835966215, 'loss': 5.175039009928703}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 39.134745401049415}

    - Test Retain Results: {'accuracy': 71.28888888888889, 'precision': 0.7345982180156408, 'recall': 0.7128888888888889, 'f1_score': 0.7136276348212536, 'loss': 1.4036227469704714}

    - The MIA has an accuracy of 0.933 on forgotten vs unseen images

6. UNLEARNT + QUANTIZED int8 resnet50 model trained on CIFAR10 

    - Test Results: {'accuracy': 64.29, 'precision': 0.6017310510918735, 'recall': 0.6429, 'f1_score': 0.6109979916840687, 'loss': 5.274725973653793}

    - Test Forget Results: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'loss': 40.05655161539713}

    - Test Retain Results: {'accuracy': 71.43333333333334, 'precision': 0.7362397324408531, 'recall': 0.7143333333333334, 'f1_score': 0.7153105082486577, 'loss': 1.412348008372194}

    - The MIA has an accuracy of 0.934 on forgotten vs unseen images
