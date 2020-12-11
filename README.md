# Goals
HII regions (regions of interstellar atomic hydrogen that is ionized) 
are important objects for astrophysics. 
In particular, studies of HII regions can help us better understand 
star formation, the evolution of chemical elements abundance in galaxies, 
as distance indicators, etc. Those objects could be distinguished using 
bright emission lines, such as H_alpha, H_beta, O[III],
which leads to photometric excess in g and r filters.
But using linear relationships for HII classification is complicated by the similarity
of HII regions with other objects, for instance, blue stars. The goal of 
this project is to build a model for discovering complex non-linear
relationships between input parameters for HII regions classification.


# Dataset
The base for this investigation is a sample of approx. 43000 HII regions,
selected from SDSS DR7 database from objects for which spectra are available (first sample).
To represent the "other" objects, the classification data from SDSS DR12
was downloaded. Those objects were classified automatically based on available spectra
of objects (second sample). The second sample includes 3 classes of objects: Galaxy, Star, QSO 
(Quasi-Stellar Object). As HII regions were selected from the same database, the intercept
of two samples was removed from the second sample. 
The final dataset includes approx. 445000 objects.

<img src="https://github.com/lap1dem/hii-classification/blob/master/figures/piechart.png?raw=true" height="250" align="center">


# Model
A neural network was built using `tensorflow`. The input layer takes six parameters - 
measured magnitudes in photometric filters u, b, g, r, i, z. The output layer represents 
four target classes - HII, Other Galaxies, QSO, Stars. The model also includes 3 hidden
layers with 12, 24 and 48 neurons respectively (see scheme below). 
The cross entropy and Adam optimizer were used to calculate and minimize the errors in the training.

<img src="https://github.com/lap1dem/hii-classification/blob/master/figures/scheme.png?raw=true" height="350" align="center">



# Results
The model was trained and tested with 94.24% and 94.47% total accuracy respectively.
But the idea is to build a classifier for HII regions, so the precision of HII regions prediction
would be more appropriate, which is 77.8% (see confusion matrix below). Comparing
to other works in this field, the result is satisfying but could be improved.


<img src="https://github.com/lap1dem/hii-classification/blob/master/figures/confusion.png?raw=true" height="350" align="center">



# Further steps
* Dataset review.
* Hyperparameters optimization for training the most precise model.
* Comparing results to other ML techniques, such as SVM or Random Forest.