# bert-word-embeddings.py

"""
this script:
- instantiates a pre-trained BERT (base model, uncased)
- using some hard-coded test sentences, it preprocess sentences
- feeds preprocessed tokens them thru BERT to get all parameters out (num_sentences x num_words x 13 layers x 768 features
- the last 4 layers are averaged for each token (word) to be a single 768 length feature vector per word.
    - option to use concatenation of the last 4.
- remove punctuation, distinguished sentence start/end symbols, punctuation
- use PCA: dimension reduce from 768 to about 16 dimensions (manually choosing most influential principle components
- cluster - HDBSCAN and SpectralClustering are tested, but HDBSCAN is best. (Manually tune min_cluster_size paramerter)
- prints of clusters as lists of words, and cluster distributions found in each sentence.

Citations:
### BERT as a service from https://colab.research.google.com/drive/1yFphU6PW9Uo6lmDly_ud9a6c4RCYlwdX#scrollTo=Mq2PKplWfbFv
### Cite :Chris McCormick and Nick Ryan. (2019, May 14). BERT Word Embeddings Tutorial. Retrieved from http://www.mccormickml.com

## also read: https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b

## WordPiece papers: Schuster & Nakima 2012, and Wu et al 2016.

## citations for word embeddings:
## Devlin et al. 2019 (original BERT paper) tested CoNLL data for Entity Extraction task with different word embeddings (w/o and w/ fine tuning the model).
## they found that the concatenation of the last 4 layers gave best results and that averaging the last 4 layers (unsure of how they weighted them) gave close to as good
## Miaschi & Dell'Orletta 2020 found that BERT dramatically outperforms Word2Vec on 68 tasks using word embeddings of the using the last layer (-1) and the -8 layer.
## They also note that for morphosyntactic tasks, word-level embeddings are better than sentence level aggregations.

## HDBSCAN citations i have in the work w/ Deborah--> Dropbox/archer/hdbscan? maybe.
"""


# import os
import pandas as pd
import numpy as np
from code.utils import *
# import nltk
from nltk.corpus import stopwords
from string import punctuation
# from nltk import sent_tokenize, word_tokenize, pos_tag
# from string import punctuation
from transformers import pipeline
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering



# Load pre-trained model tokenizer (vocabulary)
def preprocess(sentences):
    """
    Input: set of sentences (list of strings, on string per sentence).
    Description: This will tokenize all of the sentences (applies .lower, and keeps punctuation)
                Map the tokens to thier word IDs in WordPiece and pads out 0s
                creates an array of sentence indices as needed.
    Output: tokens = list of word tokens (strings)
            tokens_tensor = torch tensor of wordpiece indices (one row per sentence, one column per word, padded to be same length per sentence)
            segment_ids = torch tensor, one row per sentence w/ sentence index repeated
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens = [] ## to be populated and returned
    m = 0 ## running count of max leng of a sentence.
    for s in sentences:
        s = "[CLS] " + s + " [SEP]" ## add BERT-required delimiters
        t = tokenizer.tokenize(s) ## does the tokenization, .lower, and keeps punctuation (what i think we want for BERT uncased)
        tokens.append(t) ## add the tokens for later analysis
        m = max(m, len(t)) ## update max length.

    ## run thru again and get the WordPiece indices for each word and the needed sentence indicator array:
    input_ids = []
    segment_ids = []
    for i, t in enumerate(tokens):
        ids = tokenizer.convert_tokens_to_ids(t)
        ids += [0]* (m - len(ids))
        assert len(ids) == m
        input_ids.append(ids)
        segment_ids.append([i]*m)

    tokens_tensor = torch.tensor(input_ids) ## torch tensor, one row per sentence, one column per word (padded), words represented by indices from WordPiece embeddings
    segments_tensor = torch.tensor(segment_ids) ## torch tensor, one row per sentence, sentence index in every spot.
    return tokens, tokens_tensor, segments_tensor


def get_hidden_states_tensor(tokens_tensor, segments_tensor):
    """
    Input: tokens_tensor, segments_tensor (see output of preprocessing function)
    Description: Instantiates the BERT base (12x768) uncased model and feeds this data thru the model.
                takes the hidden states (all parameters) and makes them a torch tensor,
                reorganizes the tensor dimensions to be more intuitive.
    Output: token_embeddings: torch tensor with 4 dimensions, num_sentences x max_num_tokens, 13 layers (12 + inputs), 768 features
    """
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True) ## second argument will have it output all hidden states.
    model.eval() ## feed forward mode

    # Run the text through BERT, and collect all of the hidden states produced from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)
        hidden_states = outputs.hidden_states

    ## note that BERT is 12 layer NN, and we added an input layer, so hidden_states should have 13 layers x num sentences x num words x 768 features
    token_embeddings = torch.stack(hidden_states, dim=0) ## hidden_states is a tuple of torch tensors. make it simply a tensor
    token_embeddings.size() ## 13 layers x num_sentences x  max_num_tokens x 768 features
    token_embeddings = token_embeddings.permute(1,2,0,3) ## reorders:   num_sentences xmax_num_tokens x 13 layers x 768 features
    return token_embeddings


def make_word_vectors_from_last_4(token_embeddings, type = "ave"):
    """
    Input: token_embeddings (output of get_hidden_states_tensor)
    Descriptioni: For each word it grabs the last 4 hidden layers and either averages (type = "ave" default) them (768 length vector per word)
    or concatenates (type = "cancat") them (3072 length vector per word).

    output: Returns an array that is 3 dimensional (num_sentences x num_tokens x length of feature vector)
    """
    assert type in ["ave", "concat"]

    t = np.array(token_embeddings)
    n_s, n_t, n_l, n_f = t.shape ## number of sentences, number of tokens, 13 layers, 768 features
    if type == "ave":
        a = np.zeros((n_s, n_t, n_f))
        for i in range(n_s):
            for j in range(n_t):
                a[i,j] = np.average(t[i,j,-4:, :], axis = 0)

    if type == "concat":
        a = np.zeros((n_s, n_t, 4*n_f))
        a.shape
        for i in range(n_s):
            for j in range(n_t):
                a[i,j] = t[i,j,-4:, :].flatten()
    return a


def remove_stop_words(word_vecs, keep_cls = True):
    """
    Input:  word_vecs = tensor = an array that is 3 dimensional (num_sentences x num_tokens x length of feature vector)
            keep_cls = bool indicating if we want the feature vector for the [CLS] (sentence start indicator token.) this token's feature vector is supposed to be a good vector for the sentence.
    output: indices = dict of form {sentence index: list of remaining word indices }
            wv_dict = dict of form {sentence index: 2d array of remaining words (num_remaining_words x num_features)}
    """
    stop_words = set(stopwords.words('english')).union({char for char in punctuation})
    stop_words= stop_words.union({"[SEP]"})
    if not keep_cls:
        stop_words = stop_words.union({"[CLS]"})

    indices= {} ## for sentence i: list of token indices j that are not stopwords
    wv_dict = {} ## for sentence i, array of feature vectors of those tokens that are not stopwords
    for i,s in enumerate(tokens): ## sentence
        indices[i] = [j for (j,t) in enumerate(s) if t not in stop_words]
        wv_dict[i] = np.vstack([word_vecs[i,j,:] for j in indices[i]])
    return indices, wv_dict


def pca_plot(wv_dict):
    """
    Plots explained_variance_ratio_ for the array of all words in wv_dict
    Used to define n_components for PCA dimension reduction
    """
    all_words = np.vstack([wv_dict[i] for i in wv_dict]) ## c is the array of feature vectors of all the non-stopwords
    all_words.shape
    pca = PCA().fit(all_words)

    plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)

    plt.xlabel('time (s)')
    plt.xlabel("component")
    plt.ylabel("explained_var / total var")
    plt.show()


def pca_dim_reduction(wv_dict, n_components):
    """
    Input: wv_dict = dict from output of remove_stop_words
            n_components = positive int. use pca_plot to get it
    Output: wv_dict2 = same dict, but dimension reduced feature vectors.
    """
    all_words = np.vstack([wv_dict[i] for i in wv_dict]) ## c is the array of feature vectors of all the non-stopwords
    all_words.shape
    pca = PCA(n_components = n_components).fit(all_words)
    wv_dict2 = {}
    for i,a in wv_dict.items():
        wv_dict2[i]= pca.transform(a)
    return wv_dict2


def smash_and_make_lookup_dicts(wv_dict2):
    """
    Creates X = array of all word vectors remaining.
    Creates mappings to and from wv_dict2 and X.
    Output:
        - X = array of shape n_total_words_in_wv_dicts2 x num_features_per_word
        - lookup = {} ## {(sentence i, word i_ ): overall index j_
        - reverse_lookup = {} ## overall index j_ : (sentence i, word i_)
    """
    X = np.vstack([wv_dict2[i] for i in wv_dict2])
    ## now make lookups:
    j = 0
    k = 0
    c = 0
    lookup = {} ## {(sentence i, word i_ ): overall index j_
    reverse_lookup = {} ## overall index j_ : (sentence i, word i_)
    for i,a in wv_dict2.items():
        j,k = k, k + a.shape[0] ## overall word indices for words in sentence i
        for i_,j_ in enumerate(range(j,k)):
            lookup[(i,i_)] = j_  ## sentence i's word i_ maps to overall word j_
            reverse_lookup[j_] = (i,i_) ## overall word j_ maps to sentence i word i_
    return X, lookup, reverse_lookup


def retrieve_word(j_, reverse_lookup, indices, tokens):
    """
    Input:  j_ = int (overall word index),
            reverse_lookup, indices, tokens (all three are previous outputs )
    Output: (i,j,token) triple, i = the sentence index, the original word index j, and the actual token (string).
    """
    #example: j_ = 30 #overall_index. now what is the original word?
    i,i_ = reverse_lookup[j_]
    j = indices[i][i_]
    return i, j, tokens[i][j]


def run_hdbscan_clustering(X, reverse_lookup, indices, tokens, min_cluster_size=2):
    """
    Input:  X = smashed array of all remaining words

            min_cluster_size = int gives parameter for HDBSCAN

    Description: runs HDBSCAN on X, and finds distribution of each cluster in each sentence.
    Output: clusterer (HDBSCAN class that  is fit to the data, Note: clusterer.labels_ is the list of labels/cluster assignments  for each row in X  )
            s_labels = list of labels (cluster numbers) found by the algorithm
            clusters = {cluster label: list of tokens in that cluster}
            topic_dists = {sentence index: list or percents of each topic in that sentence}
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    clusterer.fit(X)
    clusterer.labels_
    s_labels = sorted(set(clusterer.labels_))
    s_labels ## set of labels. label -1 indicates "no cluster"
    clusters = {l: list(map( lambda j_: retrieve_word(j_, reverse_lookup, indices, tokens), np.where(clusterer.labels_== l)[0])) for l in s_labels }

    ## topic distributions per sentence:
    j = 0
    k = 0
    topic_dists = {}
    for i, a in wv_dict2.items():
        j,k = k, k + a.shape[0] ## sentence indices updated for this sentence
        l = list(clusterer.labels_[j:k])
        topic_dists[i] = {x: l.count(x)/len(l) for x in sorted(set(l))}
        # topic_dists[i] = list(map(lambda x: l.count(x)/len(l), s_labels))
    return clusterer, s_labels, clusters, topic_dists


def run_spectral_clustering(X, reverse_lookup, indices, tokens, n_clusters=5):
    """
    Input:  X = smashed array of all remaining words
            other inputs are outpus of previous fun
            n_clusters = int for number of clusters to pass to Spectral Clustering

    Description: runs SpectralClustering on X, and finds distribution of each cluster in each sentence.
    Output: clusterer (SCc lass that  is fit to the data, Note: clusterer.labels_ is the list of labels/cluster assignments  for each row in X  )
            s_labels = list of labels (cluster numbers) found by the algorithm
            clusters = {cluster label: list of tokens in that cluster}
            topic_dists = {sentence index: list or percents of each topic in that sentence}
    """

    clusterer = SpectralClustering(n_clusters=6, assign_labels="discretize", random_state=0).fit(X)

    clusterer.fit(X)
    clusterer.labels_
    s_labels = sorted(set(clusterer.labels_))
    s_labels ## set of labels. label -1 indicates "no cluster"
    clusters = {l: list(map( lambda j_: retrieve_word(j_, reverse_lookup, indices, tokens), np.where(clusterer.labels_== l)[0])) for l in s_labels }

    ## topic distributions per sentence:
    j = 0
    k = 0
    topic_dists = {}
    for i, a in wv_dict2.items():
        j,k = k, k + a.shape[0] ## sentence indices updated for this sentence
        l = list(clusterer.labels_[j:k])
        topic_dists[i] = {x: l.count(x)/len(l) for x in sorted(set(l))}
        # topic_dists[i] = list(map(lambda x: l.count(x)/len(l), s_labels))
    return clusterer, s_labels, clusters, topic_dists


sentences = [
    "This really long sentence is interestingly boring, which seems oxymoronical, but its not!",
    "This sentence is terrible and horrible and negative!",
    "This sentence is not terrible or horrible or negative.",
    "This is a neutral sentence.",
    "I am testing this terrible, not great sentiment analyzer.",
    "I am testing this great, not terrible sentiment analyzer.",
    'I am testing this seemingly awesome, super cool package!',
    "I went to the store and they were out of potatoes.",
    "I went to the store.",
    "i think this could be useful.",
    "We are very happy to show you the transformers library.",
    "We hope you don't hate it.",
    "we hope you don't hate it",
    "We hope you don't hate it too much!",
    "The tool looks useful.",
    "I think the threat intelligence piece would be very nice.",
    "The UI seems hard to use.",
    "The UI seems really hard to learn to use effectively.",
    ]



sentences2 = [
    "After the bank robber, Rob, robbed the bank, he swam across the river and buried the treasure on the opposite river bank.",
    "I like fishing in rivers.",
    "Chase is a huge bank.",
    "What a fancy dinner that was!",
    "I fancy a that tin of beans for dinner.",
    "Can you pass me a can of soup?",
    "I cannot eat canned food.",
    "The threat intel plugin looks very useful.",
    "They provide a threat intel, but we would need the NIOC's threat feed.",
    "I think the metrics and statistics provided look useful.",
    "Some of the statistics provided, like counts of alert, counts of tickets, would help the SOC lead but not the analyst.",
    "The interface seemed very complicated.",
    "I think the workflows and the user interface for that part would help a lot.",
    "Workflows and playbooks were not really covered.",
    "The UI makes it worth it!"
    ]

tokens, tokens_tensor, segments_tensor = preprocess(sentences2) ## preprocess
token_embeddings = get_hidden_states_tensor(tokens_tensor, segments_tensor) ## run forward thru BERT
token_embeddings.shape ### these are the 13 layers x 768 features per layer x sentences x words.
word_vecs = make_word_vectors_from_last_4(token_embeddings, type = "concat") ## array sentences x words x features.
word_vecs.shape
indices, wv_dict = remove_stop_words(word_vecs, keep_cls = False) ## remove stopwords, keep indices of remaining words so we can look up their original tokens.
pca_plot(wv_dict) ## PCA plot so we can see how to dimension reduce.
n_components = 16 ### manually set the dimension reduction
wv_dict2 = pca_dim_reduction(wv_dict, n_components) ## dict of for {index i: array of shape num_tokenes_in_sent_i x n_components}
X, lookup, reverse_lookup = smash_and_make_lookup_dicts(wv_dict2) ## X is the array of all word vectors (not indexed by sentence x word any more)
clusterer, s_labels, clusters, topic_dists = run_hdbscan_clustering(X, reverse_lookup, indices, tokens, min_cluster_size=4)
# clusterer, s_labels, clusters, topic_dists = run_spectral_clustering(X, reverse_lookup, indices, tokens, n_clusters= 5)



print("Test sentences: ")
for i,s in enumerate(sentences2):
    print(f"\t{i}:{s}")

print("Word clusters found: ")
for i,l in clusters.items():
    print (f'\t{i}: {l}')

print("Cluster distribution per sentence: ")
for i, d in topic_dists.items():
    print(f'\t{i}: {d} \n\t{tokens[i]} ')
