## Project TAROT: The AI-driven Renal Outcome Tracking

### Aim:
- To predict chronic kidney disease (CKD) progression with artificial intelligence (AI) models

### How to use:
This project aims to predict the cumulative incidence function (CIF) to start renal replacement therapy (RRT) / eGFR less then 10ml/min/1.73m^2 or all cause mortality within 5 years using ensembled deep learning models.

In this project the outcome are competitive, where 0 are censored, 1 is either eGFR < 10ml/min/1.73m^2 or starting RRT and 2 is the all cause mortality. The prediction was made based on competitive outcome survival analysis with deepsurv and deephit models.

- The [dataloader2.py](code/dataloader2.py) contained all the code used for data cleaning, scaling and imputation, the function create_pipeline will create a scikit-learn data pipeline for data preprocessing
- The [databalancer2.py](code/databalancer2.py) contained all the code used to rebalance the dataset for model training
- The [netweaver2.py](code/netweaver2.py) contained the code for pytorch neural network creation
- The [datatrainer2.py](code/datatrainer2.py) contained the code to feed in training and validation data for model training
- The [modeleval.py](code/modeleval.py) contained the code to evaluate the robustness of the prediction model


The prediction (output) will be in the format of numpy array with shape (2, 6, number of data entries), where output[0] will be the array of CIF of outcome 1 (low eGFR/starting RRT) and output[1] will be the CIFarray of outcome 2 (all cause mortality). The prediction time point will be 0, 1, 2, 3, 4 and 5 years.

An example jupyter notebook of using the models for training and prediction is in [meta_learner.py](code/meta_learner.ipynb)

### Reference:
1. Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In: Advances in Neural Information Processing Systems 32 [Internet]. Curran Associates, Inc.; 2019. p. 8024–35. Available from: http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
2. Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology. 2018;18(1): 24. https://doi.org/10.1186/s12874-018-0482-1.
3. DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks | Proceedings of the AAAI Conference on Artificial Intelligence. https://ojs.aaai.org/index.php/AAAI/article/view/11842 [Accessed 21st November 2024].
4. Kvamme H. havakv/pycox. 2024. https://github.com/havakv/pycox [Accessed 21st November 2024].





