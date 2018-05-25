# tf_EEGNet
It's a tensorflow implemention for EEGNet

for more inoformation see https://github.com/vlawhern/arl-eegmodels

# Experment
EEGNet just cannont converges on BCI_competion 2a and P300 even in the train set.

And it would predit the same labels for all samples.

The BCI_competion 2a is label-balance, while the P300 have a proportion of 9.23 for 1 aginst 0. After the unbalance_weights was added, the prediction will not be the same but fixed.
