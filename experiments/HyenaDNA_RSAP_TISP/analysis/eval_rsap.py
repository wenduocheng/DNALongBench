import sys
import numpy as np
from scipy.stats import pearsonr


# python eval_rsap.py ../outputs/RSAP_mouse/[timestamp]/checkpoints/last.result.npy mouse

if __name__ == "__main__":
    checkpoint = sys.argv[1]
    gene_type = sys.argv[2]

    if gene_type == "mouse":
        alltarget = np.load("../data/Enformer/mouse_test.npy", mmap_mode='r')
    elif gene_type == "human":
        alltarget = np.load("../data/Enformer/human_test.npy", mmap_mode='r')

    file = f"{checkpoint}/last.result.npy"
    allpred = np.load(file, mmap_mode='r')

    n_gene, n_bin, n_track = allpred.shape

    correlations = np.zeros((n_gene,))
    # Iterate over each predicted track
    for gene in range(n_gene):
        # Flatten the predictions and actuals for the current track across all bins and test samples
        pred_flat = allpred[gene, :, :].flatten()
        actual_flat = alltarget[gene, :, :].flatten()
        
        # Compute Pearson correlation for the current track
        correlation, _ = pearsonr(pred_flat, actual_flat)
        
        # Store the correlation
        correlations[gene] = correlation
        print(gene, correlation)

    # Display the Pearson correlations
    print("Pearson correlations for each predicted gene:")
    print(correlations)

    # Optionally, compute and display the mean correlation as a summary statistic
    mean_correlation = np.mean(correlations)
    print(f'Mean Pearson correlation across all genes: {mean_correlation}')