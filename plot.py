
# Author: yoshi
# Date: 8/14/2019

## Project: gcn pathway
## Script for plot accuracy and cost

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import joblib

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

# Load data
data = joblib.load('.jbl')
cv0 = data[0]

# Plot cost
plt.plot(cv0['training_cost'], 'k-', lw=2, color='green', label='Train Set Cost')
plt.plot(cv0['validation_cost'], 'r-', lw=2, color='magenta', label='Validation Set Cost')
plt.title("Loss per Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.savefig("./test_loss.png")
plt.clf()

# Plot accuracy
plt.plot(cv0['training_acc'], 'k-', lw=2, color='green', label='Train Set Accuracy')
plt.plot(cv0['validation_acc'], 'r-', lw=2, color='magenta', label='Validation Set Accuracy')
plt.title("Train and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.savefig("./test_acc.png")
plt.clf()


