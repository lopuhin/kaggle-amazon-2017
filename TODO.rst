TODO:

- add from scratch models

submissions:

- default submit
- try "--weighted 1"
- try "--average-thresholds 1"
- try bayes submission
- albu merge

nn training:
- try RandomSizedCrop with larger min area (~0.25)
- try to finetune with SGD
- add VGG and Inception (check input size for Inception)
- add models trained from scratch to the ensemble
- stratified split
- vary batch size
- vary input size
- train several models jointly
- mixture of experts?

prediction/f2 optimization:
- 5 stage scheme from 913 paper
- optimize F2 loss (try training it from scratch?)
- optimize expected F2 score for sigmoid predictions - bad?
- xgboost on fc inputs
- xgboost on predictions
