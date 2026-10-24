# Keras Tuner results

trial  dropout  unfreeze_layers       lr  val_auc
   09      0.4               40 0.000044 0.895157
   01      0.2               40 0.000051 0.890693
   03      0.2               40 0.000172 0.890218
   06      0.2               10 0.000718 0.873599
   05      0.1               20 0.000028 0.863723
   07      0.3               10 0.000068 0.831244
   04      0.5                0 0.000855 0.795916
   08      0.1                0 0.000304 0.667521
   02      0.3                0 0.000099 0.571320
   00      0.4                0 0.000066 0.492783

## Spearman correlation with val_auc
lr                -0.309
dropout           -0.173
unfreeze_layers    0.934

## val_auc by number of unfrozen layers
                   mean     max  count
unfreeze_layers                       
0                0.6319  0.7959      4
10               0.8524  0.8736      2
20               0.8637  0.8637      1
40               0.8920  0.8952      3
