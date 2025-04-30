import os
import sys
import numpy as np
import pandas as pd
import selene_sdk
from scipy.signal import convolve


import pyBigWig
import tabix
from selene_sdk.targets import Target
import numpy as np

root = "../data/puffin/"
save_dir = "../data/puffin/evaluation"
genome = selene_sdk.sequences.Genome(
                    input_path=root+"Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    blacklist_regions= 'hg38'
                )


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None, 
        replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

            
        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist)  for blacklist in self.blacklists]
            self.initialized=True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise
        
        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s)-start,0): int(e)-start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = wigmat[replacement_indices, np.fmax(int(s)-start,0): int(e)-start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat



def preprocess():
    print("Preprocessing TISP test data")
    tfeature = GenomicSignalFeatures([root+"agg.plus.bw.bedgraph.bw",
    root+"agg.encodecage.plus.v2.bedgraph.bw",
    root+"agg.encoderampage.plus.v2.bedgraph.bw",
    root+"agg.plus.grocap.bedgraph.sorted.merged.bw",
    root+"agg.plus.allprocap.bedgraph.sorted.merged.bw",
    root+"agg.minus.allprocap.bedgraph.sorted.merged.bw",
    root+"agg.minus.grocap.bedgraph.sorted.merged.bw",
    root+"agg.encoderampage.minus.v2.bedgraph.bw",
    root+"agg.encodecage.minus.v2.bedgraph.bw",
    root+"agg.minus.bw.bedgraph.bw"],
                                ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
    ,'encoderampage_minus', 'encodecage_minus',
    'cage_minus'],
                                (100000,),
                                [root+"fantom.blacklist8.plus.bed.gz",root+"fantom.blacklist8.minus.bed.gz"],
                                [0,9], [1,8], [0.61357, 0.61357])



    allseq1 = genome.get_encoding_from_coords('chr8',0, 145138636)
    allcage1 = tfeature.get_feature_data('chr8',0, 145138636)
    allcage1 = allcage1[:,25000:-25000]
    allseq1 = allseq1[25000:-25000,:]
    print('allcage1',allcage1.shape)
    print('allseq1',allseq1.shape)

    save_path = os.path.join(save_dir, 'allcage.npy')
    np.save(save_path, allcage1)
    print(f"Done saving {save_path}")

    save_path = os.path.join(save_dir, 'allseq.npy')
    np.save(save_path, allseq1)
    print(f"Done saving {save_path}")


    allseq2 = genome.get_encoding_from_coords('chr9',0, 138394717)
    allcage2 = tfeature.get_feature_data('chr9',0, 138394717)
    allcage2 = allcage2[:,25000:-25000]
    allseq2 = allseq2[25000:-25000,:]
    print('allcage2',allcage2.shape)
    print('allseq2',allseq2.shape)

    save_path = os.path.join(save_dir, 'allcage2.npy')
    np.save(save_path, allcage2)
    print(f"Done saving {save_path}")
    save_path = os.path.join(save_dir, 'allseq2.npy')
    np.save(save_path, allseq2)
    print(f"Done saving {save_path}")


    allseq = np.concatenate([allseq1, allseq2], axis=0)


    allseq_n = (allseq == 0.25).all(axis=-1)
    allseq_n_1k = convolve(allseq_n, np.ones(1001), mode='same')
    valid_seqlocs = allseq_n_1k<0.1

    print('valid_seqlocs.shape',valid_seqlocs.shape)

    save_path = os.path.join(save_dir, 'valid_seqlocs.npy')
    np.save(save_path, valid_seqlocs)
    print(f"Done saving {save_path}")


    allcage = np.concatenate([allcage1, allcage2], axis=1)
    allcage[:,~valid_seqlocs]=0
    print('allcage.shape',allcage.shape)

    save_dir = '../data/puffin/evaluation/'
    save_path = os.path.join(save_dir, 'allcage_valid.npy')
    np.save(save_path, allcage)
    print(f"Done saving {save_path}")


def process(path):
    allpred_raw = np.load(path, mmap_mode='r')
    print("allpred_raw.shape", allpred_raw.shape)   # (11338, 100000, 10)
    if (allpred_raw[0]<0).any():
        print("Use exp to convert logit to probability")
        allpred_raw = np.exp(allpred_raw)

    pointer = 0
    allpred1 = np.zeros((10,145138636))
    for i in np.arange(0, 145138636, 50000)[:-2]:
        pred1 = allpred_raw[pointer].transpose(1,0)   # (10, 100_000)
        pred2 = allpred_raw[pointer+1].transpose(1,0)
        if np.isnan(pred1).any():
            print(i, "pred1", pred1.min(), pred1.max())
            pred1 = np.nan_to_num(pred1)
        if np.isnan(pred2).any():
            print(i, "pred2", pred2.min(), pred2.max())
            pred2 = np.nan_to_num(pred2)
            # pred1[np.isnan(pred2)] = 0
        allpred1[:,i+25000:i+75000] = pred1[:,25000:75000]*0.5 + pred2[:,25000:75000]*0.5
        pointer += 2
    i = 145138636-100000
    pred1 = allpred_raw[pointer].transpose(1,0)
    pred2 = allpred_raw[pointer+1].transpose(1,0)
    if np.isnan(pred1).any():
        print(i, "pred1", pred1.min(), pred1.max())
        pred1 = np.nan_to_num(pred1)
    if np.isnan(pred2).any():
        print(i, "pred2", pred2.min(), pred2.max())
        pred2 = np.nan_to_num(pred2)
    allpred1[:,i+25000:i+75000] = pred1[:,25000:75000]*0.5 + pred2[:,25000:75000]*0.5
    allpred1 = allpred1[:,25000:-25000]
    pointer += 2

    print("allpred1", pointer, pointer // 2)

    allpred2 = np.zeros((10,138394717))
    for i in np.arange(0, 138394717, 50000)[:-2]:
        pred1 = allpred_raw[pointer].transpose(1,0)   # (10, 100_000)
        pred2 = allpred_raw[pointer+1].transpose(1,0)
        if np.isnan(pred1).any():
            print(i, "pred1", pred1.min(), pred1.max())
            pred1 = np.nan_to_num(pred1)
        if np.isnan(pred2).any():
            print(i, "pred2", pred2.min(), pred2.max())
            pred2 = np.nan_to_num(pred2)
        allpred2[:,i+25000:i+75000] = pred1[:,25000:75000]*0.5 + pred2[:,25000:75000]*0.5
        pointer += 2
    i = 138394717-100000
    pred1 = allpred_raw[pointer].transpose(1,0)
    pred2 = allpred_raw[pointer+1].transpose(1,0)
    if np.isnan(pred1).any():
        print(i, "pred1", pred1.min(), pred1.max())
        pred1 = np.nan_to_num(pred1)
    if np.isnan(pred2).any():
        print(i, "pred2", pred2.min(), pred2.max())
        pred2 = np.nan_to_num(pred2)
    allpred2[:,i+25000:i+75000] = pred1[:,25000:75000]*0.5 + pred2[:,25000:75000]*0.5
    allpred2 = allpred2[:,25000:-25000]
    pointer += 2

    print("allpred2", pointer, pointer // 2)

    print("allpred1.shape", allpred1.shape, "allpred2.shape",allpred2.shape)
    allpred = np.concatenate([allpred1, allpred2], axis=1)
    print('allpred.shape',allpred.shape)

    valid_seqlocs = np.load('../data/puffin/evaluation/valid_seqlocs.npy')
    print('valid_seqlocs.shape',valid_seqlocs.shape)

    allpred[:,~valid_seqlocs]=0
    save_path = path.replace('last.result.npy', 'allpred.npy')
    np.save(save_path, allpred)
    print(f"Done saving {save_path}")


# python process_tisp.py ../outputs/TISP/[timestamp]/checkpoints

if __name__ == "__main__":
    model_path = sys.argv[1]

    if not os.path.exists(f"../data/puffin/evaluation/valid_seqlocs.npy"):
        preprocess()

    process(f"{model_path}/last.result.npy") 
