# HS

This project contains code for the execution of tasks related to the Hacking Women's stroke study.

Project layout:

HS:
  - feature_selection
      - amt_features.py
      - execute_fs.py
      - fs_algorithms.py
      - run.py
      
  - hyperparameter_tuning
      - hpt.py
      - run.py
      
  - learn
      - binary.py
      - survival.py
      - models.py
      - execution.py
      - run.py
      
  - util_
      - util.py
      - in_out.py
      - support.py
      
For each of the scripts described below, the user is expected to have a csv file containing feature vectors with a unique identifier and a target at the end of the vector. In case of non-survival models, the target should either be a 1 or a 0. In case of survival models, the target is expected to be a list containing an event indicator (stroke/ no stroke) and the observed time (time in days since start of recording). E.g. : [event_indicator, observed time].

The scripts in the feature selection directory contain the feature selection algorithms and a script to generate a trade-off curve between the amount of features and Area Under the Curve (AUC) or the Concordance Inde (IC). To generate the curve, run the run.py script after filling in the necessary parameters.

In the hyperparameter tuning directory, the hyperparameters that you want to tune for each model can be defined. After defining the hyperparameters, run.py can be executed to perform hyperparameter tuning. Parameters have to be filled in before running.

Executing run.py from the learn directory will result in the creation of models and their results. Which models and other parameters have to be defined in the run.py file.  
