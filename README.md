# MLfinalProject


**Files in this Project:**

    dataProc - This is a preliminary analysis of the data, including histograms and simple curve fitting.
    binTreeMain - An emsemble of binary decision trees implementing bagging and Adaboost.
    NeuralNet - A simple neural network with options of sigmoid, tanh, and ReLu for the activation function and that uses RMSProp


```
.
├── 1_rawData
│   ├── cb-devices-main(1).csv
│   ├── <9 More Files>
|
├── 2_dataManipulation
│   ├── dataProc.ipynb
│   ├── dataProc.py
│   ├── dataProc_2.ipynb
│   ├── oversamp.py
│   ├── undersamp.py
│   └── xValidation.py
|
├── 3_processedData
│   ├── 012D.csv
│   ├── 01BA.csv
│   ├── 481F.csv
│   ├── 9328.csv
│   ├── aggregatedAndProcessed.csv
│   ├── aggregatedAndProcessedNN.csv
│   ├── mappings.json
│   └── mappingsNN.json
|
├── 4_learningData
│   ├── DStest.csv
│   ├── DStrain.csv
│   ├── NMtest.csv
│   ├── NMtrain.csv
│   ├── ROStest.csv
│   ├── ROStrain.csv
│   ├── RUStest.csv
│   ├── RUStrain.csv
│   ├── SMOTEtest.csv
│   ├── SMOTEtrain.csv
│   ├── TLtest.csv
│   ├── TLtrain.csv
│   └── stratFolds
│       ├── 0DStestFold.csv
│       └── <41 More Files>
|
├── 5_models
│   ├── binTree.py
│   ├── binTreeMain.py
│   ├── ensembles.py
│   ├── scikit_models.py
│   ├── singleLayerNN.py
│   ├── utils.py
│   └── xgB.py
|
├── 6_output
│   ├── NNoutputNM.txt
│   ├── NNoutputROS.txt
│   ├── NNoutputRUS.txt
│   ├── NNoutputSMOTE.txt
│   ├── NNoutputTL.txt
│   ├── Scikit
│   │   ├── Scikit_outputDS.txt
│   │   ├── Scikit_outputNM.txt
│   │   ├── Scikit_outputROS.txt
│   │   ├── Scikit_outputRUS.txt
│   │   ├── Scikit_outputSMOTE.txt
│   │   ├── Scikit_outputTL.txt
│   │   └── rocCurves
│   │       ├── DS_SK_DTs.png
│   │       ├── DS_SK_NNs.png
│   │       ├── NM_SK_DTs.png
│   │       ├── NM_SK_NNs.png
│   │       ├── ROS_SK_DTs.png
│   │       ├── ROS_SK_NNs.png
│   │       ├── RUS_SK_DTs.png
│   │       ├── RUS_SK_NNs.png
│   │       ├── SMOTE_SK_DTs.png
│   │       ├── SMOTE_SK_NNs.png
│   │       ├── TL_SK_DTs.png
│   │       └── TL_SK_NNs.png
│   ├── outputBNDS.txt
│   ├── outputDS.txt
│   ├── outputFolds
│   │   ├── output0DS.txt
│   |   └── <20 More Files>
│   ├── outputNM.txt
│   ├── outputROS.txt
│   ├── outputRUS.txt
│   ├── outputSMOTE.txt
│   ├── outputTL.txt
│   └── rocCurves
│       ├── DSbagging.png
│       ├── DSboosting.png
│       ├── NMbagging.png
│       ├── NMboosting.png
│       ├── ROSbagging.png
│       ├── ROSboosting.png
│       ├── RUSbagging.png
│       ├── RUSboosting.png
│       ├── SMOTEbagging.png
│       ├── SMOTEboosting.png
│       ├── TLbagging.png
│       └── TLboosting.png
|
├── 7_outputEval
│   ├── graphs.py
│   ├── heatmap.py
│   └── rocExp.py
├── README.md
└── researchGuide.pdf

13 directories, 167 files
```


**Sources:**

    Here are all the articles/materials used

[1] B. Kumar, “Imbalanced Classification: Handling Imbalanced Data using Python,” Analytics Vidhya, 24-Jul-2020. [Online]. Available: https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/. [Accessed: 16-Apr-2021].
[2] R. Karim, “10 Gradient Descent Optimisation Algorithms,” Medium, 04-May-2020. [Online]. Available: https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9. [Accessed: 16-Apr-2021]. 
[3] G. Lemaitre, F. Nogueira, and C.K. Aridas (2014-2020) imbalanced-learn [Python Package] scikit-learn-contrib/imbalanced-learn: A Python Package to Tackle the Curse of Imbalanced Datasets in Machine Learning
[4] G. Lemaitre, F. Nogueira, and C.K. Aridas, “NearMiss,” Github, 2017. [Online] Available:  NearMiss — Version 0.8.0  [Accessed: 25-Apr-2021].
[5] J. Brownlee, “Stacking Ensemble Machine Learning With Python,” Machine Learning Mastery, 10-Apr-2020. [Online]. Available: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/ 
[6] J. Brownlee, “How to Configure XGBoost for Imbalanced Classification,” Machine Learning Mastery, Feb. 04, 2020. https://machinelearningmastery.com/xgboost-for-imbalanced-classification/.
[7] P. Płoński, “Xgboost Feature Importance Computed in 3 Ways with Python,” MLJAR Automated Machine Learning, Aug. 17, 2020. https://mljar.com/blog/feature-importance-xgboost/ (accessed May 11, 2021).
[8] S. SHARMA, “Activation Functions in Neural Networks,” Medium, Sep. 06, 2017. https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6.
[9] J. Brownlee, “A Gentle Introduction to k-fold Cross-Validation,” Machine Learning Mastery, May 21, 2018. https://machinelearningmastery.com/k-fold-cross-validation/.
[10] R. Khan, “Nothing but NumPy: Understanding & Creating Binary Classification Neural Networks with…,” Medium, Dec. 08, 2020. https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c (accessed May 11, 2021).
