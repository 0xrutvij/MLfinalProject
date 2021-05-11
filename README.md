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

    https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
      Used to gain information about RMSprop addition to Gradient descent.

    https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
      Used to explain some resampling techniques to help with class imbalance.
