# MLfinalProject

##Main Files in this Project

dataProc - This is a preliminary analysis of the data, including histograms and simple curve fitting.
binTreeMain - An emsemble of binary decision trees implementing bagging and Adaboost.
NeuralNet - A simple neural network with options of sigmoid, tanh, and ReLu for the activation function and that uses RMSProp

- Executing the data processing module
  ```cd 2_dataManipulation && python3 dataProc.py```
  This accesses data in the 1_rawData folder and stores it in 3_processedData
  
- Executing the sampling and 8-fold stratified xValidation modules
  `python3 oversamp.py && python3 undersamp.py && python3 xValidation.py`
  These access data in the 3_processedData folder and store it in the 4_learningData folder.
  
- Executing the custom ensembles and the decision trees.
  `cd 5_models && python3 binTreeMain.py`
  This reads in the various sampled and stratified datasets and generates output (text and images for ROC curves) 
  saving them in the 6_output folder
  
- Executing the custom Neural Network
  `python3 NeuralNet.py`
  This reads in the various sampled and stratified datasets and generates output (text and images for ROC curves) 
  saving them in the 6_output folder




```
.
├── 1_rawData
│   ├── cb-devices-main(1).csv
│   ├── cb-devices-main(10).csv
|   └── <8 more files>
|
├── 2_dataManipulation
│   ├── dataProc.py
│   ├── oversamp.py
│   ├── undersamp.py
│   ├── xValidation.py
|   └── <2 more files>
|
├── 3_processedData
│   ├── aggregatedAndProcessed.csv
│   ├── aggregatedAndProcessedNN.csv
│   ├── mappings.json
│   └── <5 more files>
|
├── 4_learningData
│   ├── DStest.csv
│   ├── DStrain.csv
│   ├── <8 more files>
│   ├── TLtest.csv
│   ├── TLtrain.csv
│   └── stratFolds
│       ├── 0DStestFold.csv
│       └── <41 more files>
|
├── 5_models
│   ├── NeuralNet.py
│   ├── binTree.py
│   ├── binTreeMain.py
│   ├── ensembles.py
│   └── <3 more files>
|
├── 6_output
│   ├── Scikit
│   │   ├── Scikit_outputDS.txt
│   │   ├── <10 more files>
│   │   └── rocCurves
│   │       └── <12 more files>
|   |
│   ├── ensemblesOutput
│   │   └── <6 more files>
|   |
│   ├── nnOutput
│   │   └── <6 more files>
|   |
│   ├── outputStratifiedFolds
│   │   └── <21 more files>
|   |
│   └── rocCurves
│       └── <6 more files>
|   
├── 7_outputEval
│   └── <3 more files>
|
├── README.md
└── researchGuide.pdf

15 directories, 172 files
```
