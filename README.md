# MLfinalProject


**Files in this Project:**

    dataProc - This is a preliminary analysis of the data, including histograms and simple curve fitting.
    perceptron - Creates a perceptron model using batch gradient descent with RMSprop.
    binTreeMain - An emsemble of binary decision trees implementing bagging and Adaboost.
    SingleLayerNN - A simple neural network with options of sigmoid, tanh, and ReLu for the activation function and that uses RMSProp


```
.
├── 1_rawData
│   ├── cb-devices-main(1).csv
│   ├── cb-devices-main(10).csv
│   ├── cb-devices-main(2).csv
│   ├── cb-devices-main(3).csv
│   ├── cb-devices-main(4).csv
│   ├── cb-devices-main(5).csv
│   ├── cb-devices-main(6).csv
│   ├── cb-devices-main(7).csv
│   ├── cb-devices-main(8).csv
│   └── cb-devices-main(9).csv
├── 2_dataManipulation
│   ├── dataProc.ipynb
│   ├── dataProc.py
│   ├── dataProc_2.ipynb
│   ├── oversamp.py
│   ├── undersamp.py
│   └── xValidation.py
├── 3_processedData
│   ├── 012D.csv
│   ├── 01BA.csv
│   ├── 481F.csv
│   ├── 9328.csv
│   ├── aggregatedAndProcessed.csv
│   ├── aggregatedAndProcessedNN.csv
│   ├── mappings.json
│   └── mappingsNN.json
├── 4_learningData
│   ├── DStest.csv
│   ├── DStestNN.csv
│   ├── DStrain.csv
│   ├── DStrainNN.csv
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
│       ├── 0DStrainFold.csv
│       ├── 0ROStestFold.csv
│       ├── 0ROStrainFold.csv
│       ├── 0RUStestFold.csv
│       ├── 0RUStrainFold.csv
│       ├── 1DStestFold.csv
│       ├── 1DStrainFold.csv
│       ├── 1ROStestFold.csv
│       ├── 1ROStrainFold.csv
│       ├── 1RUStestFold.csv
│       ├── 1RUStrainFold.csv
│       ├── 2DStestFold.csv
│       ├── 2DStrainFold.csv
│       ├── 2ROStestFold.csv
│       ├── 2ROStrainFold.csv
│       ├── 2RUStestFold.csv
│       ├── 2RUStrainFold.csv
│       ├── 3DStestFold.csv
│       ├── 3DStrainFold.csv
│       ├── 3ROStestFold.csv
│       ├── 3ROStrainFold.csv
│       ├── 3RUStestFold.csv
│       ├── 3RUStrainFold.csv
│       ├── 4DStestFold.csv
│       ├── 4DStrainFold.csv
│       ├── 4ROStestFold.csv
│       ├── 4ROStrainFold.csv
│       ├── 4RUStestFold.csv
│       ├── 4RUStrainFold.csv
│       ├── 5DStestFold.csv
│       ├── 5DStrainFold.csv
│       ├── 5ROStestFold.csv
│       ├── 5ROStrainFold.csv
│       ├── 5RUStestFold.csv
│       ├── 5RUStrainFold.csv
│       ├── 6DStestFold.csv
│       ├── 6DStrainFold.csv
│       ├── 6ROStestFold.csv
│       ├── 6ROStrainFold.csv
│       ├── 6RUStestFold.csv
│       ├── 6RUStrainFold.csv
│       ├── 7DStestFold.csv
│       ├── 7DStrainFold.csv
│       ├── 7ROStestFold.csv
│       ├── 7ROStrainFold.csv
│       ├── 7RUStestFold.csv
│       └── 7RUStrainFold.csv
├── 5_models
│   ├── __pycache__
│   │   ├── binTree.cpython-39.pyc
│   │   ├── ensembles.cpython-39.pyc
│   │   └── utils.cpython-39.pyc
│   ├── binTree.py
│   ├── binTreeMain.py
│   ├── ensembles.py
│   ├── scikit_models.py
│   ├── singleLayerNN.py
│   ├── utils.py
│   └── xgB.py
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
│   │   ├── output0ROS.txt
│   │   ├── output0RUS.txt
│   │   ├── output1DS.txt
│   │   ├── output1ROS.txt
│   │   ├── output1RUS.txt
│   │   ├── output2DS.txt
│   │   ├── output2ROS.txt
│   │   ├── output2RUS.txt
│   │   ├── output3DS.txt
│   │   ├── output3ROS.txt
│   │   ├── output3RUS.txt
│   │   ├── output4DS.txt
│   │   ├── output4ROS.txt
│   │   ├── output4RUS.txt
│   │   ├── output5DS.txt
│   │   ├── output5ROS.txt
│   │   ├── output5RUS.txt
│   │   ├── output6DS.txt
│   │   ├── output6ROS.txt
│   │   ├── output6RUS.txt
│   │   ├── output7DS.txt
│   │   ├── output7ROS.txt
│   │   └── output7RUS.txt
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

    https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
      Used to gain information about RMSprop addition to Gradient descent.

    https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
      Used to explain some resampling techniques to help with class imbalance.
