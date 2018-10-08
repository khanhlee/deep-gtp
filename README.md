# deep-gtp

DeepGTP - A deep convolutional neural network for identifying GTP binding sites in Rab GTPases

Deep learning has been increasingly and widely used to solve numerous problems in various fields with state-of-the-art performance. It can also be applied in bioinformatics to reduce the requirement for feature extraction and reach high performance. This study attempts to use deep learning to predict GTP binding sites in Rab proteins, which is one of the most vital molecular functions in life science. A functional loss of GTP binding sites in Rab proteins has been implicated in a variety of human diseases (choroideremia, intellectual disability, cancer, Parkinson’s disease). Therefore, creating a precise model to identify their functions is a crucial problem for understanding these diseases, and designing the drug targets. Our deep learning model with two-dimensional convolutional neural network and position specific scoring matrices profiles could identify GTP binding residues with achieved sensitivity of 92.3%, specificity of 99.8%, accuracy of 99.5%, and MCC of 0.92 for independent data set. Compared with other published works, this approach achieved a significant improvement. Throughout the proposed study, we provide an effective model for predicting GTP binding sites in Rab proteins and a basis for further research that can apply deep learning in bioinformatics, especially in nucleotide binding site prediction.

File:

- train_model_cv: train model with 5-fold cross-validation
- build_model.py: build model using 2D CNN on given dataset
- motor_load_model.py: load JSON model
