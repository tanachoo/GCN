
# Author: yoshi
# Date: 8/14/2019
# Updated: 8/16/2019

## Project: gcn pathway
## Script for plot accuracy and cost

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import joblib
import argparse

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

def make_plot(filename, output, cv):
    # Load data
    print(f'\nLoad: {filename}\n'
          f'cv fold: {cv}')
    result_data = joblib.load(filename)
    data = result_data[cv]

    # Plot cost
    plt.plot(data['training_cost'], 'k-', lw=2, color='green', label='Train Set Cost')
    plt.plot(data['validation_cost'], 'r-', lw=2, color='magenta', label='Validation Set Cost')
    plt.title("Loss per Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    #plt.savefig(output + '_cv' + str(cv) + '_loss.png')
    outputfile_loss = output + '_cv' + str(cv) + '_loss_plot.png'
    plt.savefig(outputfile_loss)
    print(f'Save cost plot: {outputfile_loss}')
    # plt.savefig("./test_loss.png")
    plt.clf()

    # Plot accuracy
    plt.plot(data['training_acc'], 'k-', lw=2, color='green', label='Train Set Accuracy')
    plt.plot(data['validation_acc'], 'r-', lw=2, color='magenta', label='Validation Set Accuracy')
    plt.title("Train and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    #plt.savefig(output + '_cv' + str(cv) + '_acc.png')
    outputfile_acc = output + '_cv' + str(cv) + '_acc_plot.png'
    plt.savefig(outputfile_acc)
    print(f'Save accuracy plot: {outputfile_acc}')
    # plt.savefig("./test_acc.png")
    plt.clf()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='set input jbl result file')
    parser.add_argument('--output', type=str, help='set output file (i.e. target0_gcn2)')
    parser.add_argument('--cv', default=0, type=int, help='cross validation: select 0,1,2,3,4')
    args = parser.parse_args()

    make_plot(args.input, args.output, args.cv)

    print(f'Completed cost and accuracy plot.\n')

