
import sys
import numpy as np
import os
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
import seaborn as sns


valid_seqlocs = np.load('../data/puffin/evaluation/valid_seqlocs.npy')
allcage = np.load('../data/puffin/evaluation/allcage_valid.npy', mmap_mode='r')



def plot_pred_chr8(allpred, checkpoint):
    print(f'Saving {checkpoint}/pred_cage_chr8.pdf')
    plt.figure(figsize=(10,1),dpi=1000)
    plt.plot(allpred[0,:145138636-50000], linewidth=0.2, color='#2077b4')
    sns.despine()
    plt.savefig(f'{checkpoint}/pred_cage_chr8.pdf',format='pdf', bbox_inches = 'tight')

def plot_pred_chr9(allpred, checkpoint):
    print(f'Saving {checkpoint}/pred_cage_chr9.pdf')
    plt.figure(figsize=(10,1),dpi=1000)
    plt.plot(allpred[0,145138636-50000:], linewidth=0.2, color='#2077b4')
    sns.despine()
    plt.savefig(f'{checkpoint}/pred_cage_chr9.pdf',format='pdf', bbox_inches = 'tight')


def calculate_correlation(allpred, allcage, valid_seqlocs):
    cor1 = pearsonr(np.log10(10**np.concatenate([allpred[0,valid_seqlocs],allpred[-1,valid_seqlocs]])-1+0.1), 
             np.log10(10**np.concatenate([allcage[0,valid_seqlocs],allcage[-1,valid_seqlocs]])-1+0.1))[0]
    print('FANTOM CAGE: ', cor1)
    cor2 = pearsonr(np.log10(10**np.concatenate([allpred[1,valid_seqlocs],allpred[-2,valid_seqlocs]])-1+0.1), 
                np.log10(10**np.concatenate([allcage[1,valid_seqlocs],allcage[-2,valid_seqlocs]])-1+0.1))[0]
    print('ENCODE CAGE: ', cor2)
    cor3 = pearsonr(np.log10(10**np.concatenate([allpred[2,valid_seqlocs],allpred[-3,valid_seqlocs]])-1+0.1), 
                np.log10(10**np.concatenate([allcage[2,valid_seqlocs],allcage[-3,valid_seqlocs]])-1+0.1))[0]
    print('ENCODE RAMPAGE: ', cor3)
    cor4 = pearsonr(np.concatenate([allpred[3,valid_seqlocs]/np.log(10),allpred[-4,valid_seqlocs]/np.log(10)]), 
                np.concatenate([allcage[3,valid_seqlocs]/np.log(10),allcage[-4,valid_seqlocs]/np.log(10)]))[0]
    print('GRO-cap: ', cor4)
    cor5 = pearsonr(np.concatenate([allpred[4,valid_seqlocs]/np.log(10),allpred[-5,valid_seqlocs]/np.log(10)]), 
                np.concatenate([allcage[4,valid_seqlocs]/np.log(10),allcage[-5,valid_seqlocs]/np.log(10)]))[0]
    

# python eval_tisp.py ../outputs/TISP/[timestamp]/checkpoints

if __name__ == "__main__":
    checkpoint = sys.argv[1]

    allpred = np.load(f'{checkpoint}/allpred.npy', mmap_mode='r')
    assert allpred.shape == allcage.shape

    calculate_correlation(allpred, allcage, valid_seqlocs)
    plot_pred_chr8(allpred, checkpoint)
    plot_pred_chr9(allpred, checkpoint)

