Naive B cell project 
Introduction:
In this project we working with Naive B cells data set. 
We will try to create a model that will predict the subject's health status based
 on the features of his repertoire sequences.

Data Pre-Proccecing:
The data was obtained from a plate of naive B cells derived from blood 
samples of 98 patients, 51 of whom were diagnosed with celiac disease. 
(It is known that there are specific biomarkers that allow us to specifically 
sequence naive B cells with error rates ranging from 10-20%). We get FASTA files,
 which we tranfer in IGBLAST and MAKE_DB algorithms to align the sequences to the
 known human reference. At the end of the MAKE DB process, we have a table that 
the raws represent patient's repertoire sequences, and the different columns 
describe different characteristics of the sequence.

Project goal:
In this work we will investigate a range of state-of-the-art machine learning 
methods for classification, including random forests, logistic regression and 
support vector regression. These models will be applied to predict the 
clinically-relevant subject states based on observed repertoire features. 
Cross-validation will be used to prevent over-fitting, and feature selection will 
allow for identification of the most relevant properties.

We hope that the tools we develop will pave the way for Ig repertoire data to be
 used for diagnostic and prognostic purposes, and allow for tailored treatment
 for each individual based on their immune system characteristics. 
A satisfying classification model will approve a future low cost test for early
 detection of celiac patients, save unnecessary medical procedures and might
 reveal unique signatures for potential antibodies development.

Input:
patient’s BCR repertoire
Output:
A classified clinical status for the patient.

Features:
D: Gene & Family usage 
V : Gene & Family usage 
J : Gene usage 
V-J : all possible combination between V and J genes 
JUNCTION LENGTH: column for each type of length 
V-J-JL: all possible combination between V,J and Junction length 
DseqL: D sequence length 
FUNC: functionality 
CDR3 properties: 
CDR3_AA_LENGTH: number of amino acids. 
CDR3_AA_GRAVY : grand average of hydrophobicity(GRAVY) index. 
CDR3_AA_BULK : average bulkiness of amino acids. 
CDR3_AA_ALIPHATIC: aliphatic index. 
CDR3_AA_CHARGE: net charge. 
CDR3_AA_BASIC: fraction of informative position that are Arg,His or Lys. 
CDR3_AA_ACIDIC: fraction of informative position that are Asp or Glu. 
CDR3_AA_AROMATIC: fraction of informative positions that are His,Phe,Trp or Tyr. 
(For more information about each feature - See the "Celiac.ppt" file)

Model: 
We utilized a machine learning approach, in order to predicts the clinical group 
based on a combination of the given features. This approach can be utilized not
 only as a prediction model, but it may also be used as a tool to identify 
significant features that did not rise in the single-feature analysis.

For feature selection and prediction model for B cell repertoires, 
we calculated frequency per-sample for each repertoire of sequences.
To avoid false repetroires which may occur due to grouping of several erroneous
 sequences with correct ones, we removed sequences with rare features expression
 that appeared in at low frequencies and/or in less than others. 
In order to be sure that we worked with Naive B cells we filtered 
all sequences with no appropriate isotype, means that all the sequences 
we work with are from IGM/IGD isotype(These isotypes characterize the naive cells).

We applied Random Forests model to extract the best 20 features,
 followed by utilization of different models on the selected 
features to generate the prediction model. The process of sampling 
and training was repeated K times, to ensure that the model was not 
biased towards specific samples.

Model Steps:
Step 1: Data Overview 
1.2: Feature Engineering  
2.1.2: First visualization 
2.2: Kolmogorov Smirnov test 
2.2.1: Extract features based on KS test 
Step 3: Feature Selection - 2 methods 
3.1: Feature Selection- Our New Method - based on KS test 
3.2: Extract important features used Random Forest 
3.3: Comparison between the chosen best informative features by KS vs Random-Forest  
3.4.1: Continued analyzed based on Visualization 
Step 4: Learning Algorithms 
4.2: Random Forest 
4.2.1: Tuned Hyperparameter - Random Forest 
4.2.2: Prediction and Evaluations - Random Forest 
4.3: SVC 
4.3.1: Tuned Hyperparameter - SVC 
4.3.2: Prediction and evaluations - SVC 
4.4: Logistic Regression 
4.4.1: Tuned hyperparameters - LR 
4.4.2: Prediction and evaluation - LR 
4.5: Gaussian Naive Bayes  
4.6: Compare Machine Learning Algorithms and improve by cross validation 
Step 5: Control 
5.4 Control Results 
Step 6: New machine learning function  
6.3: New ML - Control 
Step 7: Conclusion and Results 
The next steps 

