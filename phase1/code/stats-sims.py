# stats-sims.py

"""
This script is designed to examine if we could get statistically significant results based on number of users
for phase 1.


It uses two distributions to generate ratings data in {1, 2, .., 5}
The goal here is to make realistic scores that should clearly allow us to see the second tool is better.
Using distributions above we generate user ratings for tool 1 and tool 2 for m = 5, 10, .., 45, users.
NOTE: phase 1 will have each user answer 7 questions per tool video, we're only generating 1 score per user per tool here.
Assuming we boil their 7 scores into a single score, that is what we are simulating.

First we examine the number of users (m) needed to tell with confidence that tool 2 is better than tool 1
    - tool 1 ratings have mean, variance: (2.9285714285714284, 0.9234693877551035)
    - tool 2 ratings have mean, variance: (3.647058823529412, 1.1695501730103768)
    - Results are stored in /results/stats-sim-results/p1-p2-simulations
        - ttest.json: results give the percent of hypothesis tests that confirm tool 1 not equal to tool 1  with a two-sided Welch's t-test for different numbers of users (m = 5, 10, ..40, 100)
        - binary-{z, chi2, binomail}-{threshold}.json: After converting the 1-5 scores into binary using two thresholds (3.5 meaning:1,2,3 --> 0, 4,5 --> 1 and 2.5 meaning:1,2 --> 0, 3,4,5 --> 1 ), using
            - a z-score on the difference of means (observed p1 - observed p2 / shat)
            - a chi-sq. test on difference in proportions p1, p2
            - a custom binomial test (null hypothesis is that tool 2 data is from same binomial as tool 1's--> pvalue 1, null hypoth that tool 1's data is sampled from tool 2's --> pvalu 2, average them. )
    - higher percentages are better for all the results in this folder. e.g., the result 25:.67 implies that with 25 users that test provided at least 95% confidence that tool 2 was not equal to tool 1.

Second we examine the number of users needed to tell with confidence that tool2 and tool 1 are the same.
    - tool 2 and tool 1 ratings have the same mean, variance (3.647058823529412, 1.1695501730103768)
    - results are stored in /results/stats-sim-results/p2-p2-simulations
    - same tests/thresholds as above used, mapping to same file names
    - lower percentages are better in this folder. e.g. the result 25:.08 means that with 25 users and that that test/scenario, we were 95% confident tool 1 and tool 2 were different (even though they were the same) in 0.08 of the runs.

"""

import os, sys
import pandas as pd
import numpy as np
from scipy.stats import binom_test, ttest_ind, norm, chi2_contingency

import numpy.random
from numpy.random import choice
from .utils import *

results_path = './phase1/results'
os.path.isdir(results_path)
sim_path = os.path.join(results_path, 'stats-sim-results/')
if not os.path.isdir(sim_path): os.mkdir(os.path.join(results_path, 'stats-sim-results/'))

## used for seeing mu, variance of the underlying distributions i create later...
def get_mu_var(l):
    assert np.sum(l) == 1
    mu = sum([(i+1)*l[i] for i in range(5)])
    var = sum([ ((i+1)**2)*l[i] for i in range(5)] ) - mu**2
    return mu, var


## t-test code:
def run_tt_test(p1,p2, n1, n2, n = 100, equal_var = True, verbose = False, seed=10):
    """
    Description:
        - generates data for n1 (n2) user ratings for tool 1 (2) based on probabilities p1 (p2) resp.
        - Runs the Welch's (var not equal) two-sided t-test n times and stores/returns the p-values from each.
    Input:
        - p1, p2 = length 5 lists giving probabilities of seeing each rating [1,2,3,4,5] for tools 1, 2, resp.
        - n1, n2 = ints, number of ratings to generate  (number of users rating each tool)
        - n = number of times to run the experiment
        - equal_var = True/False --> If population variance for sample set 1 == that of sample set 2, use True (changes the Deg. of Freedom)
    Output:
        - results = { i: pval } for i = 0,... n-1
    """
    numpy.random.seed(seed)
    results= {}
    for i in range(n):
        ## make overall ratings per tool:
        df1 = pd.DataFrame(zip( [f'user-{i}' for i in range(n1)],  choice(range(1,6), n1, p = p1 )) , columns = ["user", "tool"])
        df2 = pd.DataFrame(zip( [f'user-{i}' for i in range(n2)],  choice(range(1,6), n2, p = p2 )) , columns = ["user", "tool"])

        ## two-sided t-test. Null Hypoth: equal means, with assumption that they have different variances (Welch's test)
        t, pval = ttest_ind(df1["tool"], df2["tool"], equal_var = equal_var, alternative = 'two-sided') ## have to remove nan's manually before inserting data
        results[i] = pval
        if verbose: print (f'alt = {alternative}, t = {t}, pval = {pval}')

    return results


## monte carlo for ttest: wrap it in a loop:
def run_simulation_tt_test(p1, p2, n = 1000, equal_var = True, save_folder = False, seed = 10):
    """loops the test function for different numbers of users,
    stores and returns results
    if save_folderevaluates to true, results are stored in {save_folder}/.json"""
    d = {}
    for m in [5, 10, 15, 25, 30, 35, 40, 100]:
        results = run_tt_test(p1, p2, m, m, n = n, equal_var = equal_var, seed = seed)
        p = len([v for v in results.values() if v <= .05])/len(results)
        d[m] = p
    if save_folder: jsonify(d, os.path.join(save_folder, "ttest.json"))
    return d


#### let's convert data to binary (e.g. 1,2,3 --> 0, 4,5--> 1) and try z-test, chi2, binomial test:
def run_binary_tests(p1, p2, n1, n2, n = 100, threshold = 3.5,  seed=10, verbose = False):
    """
    Description:
        - generates data for n1 (n2) user ratings for tool 1 (2) based on probabilities p1 (p2) resp.
        - changes the 1,2,3--> 0 and the 4,5--> 1
        - Runs a z-test (normal distribution for difference of means) and stores p-value
        - Runs a chi squared test w/ Yates correction and stores p-value
        - Runs two binomial tests (first hypothesis is tool 2 is sampled from tool 1's probability, second is vice-versa) and stores average of p-values.
    Input:
        - p1, p2 = length 5 lists giving probabilities of seeing each rating [1,2,3,4,5] for tools 1, 2, resp.
        - n1, n2 = ints, number of ratings to generate  (number of users rating each tool)
        - n = number of times to run the experiment
        - threshold: if rating is <= threshold, it's changed to a 0, if rating is >= threshold it's changed to a 1
    Output:
        - results_z = { i: pval } for i = 0,... n-1
        - results_c = { i: pval } for i = 0,... n-1
        - results_b = { i: ave(pval1, pval2) } for i = 0,... n-1
    """
    numpy.random.seed(seed)
    results_z = {}
    results_c = {}
    results_b = {}
    for i in range(n):
        ## make 1-5 rating data:
        df1 = pd.DataFrame(zip( [f'user-{i}' for i in range(n1)],  choice(range(1,6), n1, p = p1 )) , columns = ["user", "tool"])
        df2 = pd.DataFrame(zip( [f'user-{i}' for i in range(n2)],  choice(range(1,6), n2, p = p2 )) , columns = ["user", "tool"])
        ## use threshold to make into binary:
        df3 = df1.replace([x for x in range(1,6) if x <= threshold], 0).replace([x for x in range(1,6) if x > threshold], 1)
        df4 = df2.replace([x for x in range(1,6) if x <= threshold], 0).replace([x for x in range(1,6) if x > threshold], 1)

        # ## do z-score results:
        muhat = (df3.mean() - df4.mean())
        phat = ( df3.tool.sum() + df4.tool.sum() ) /( df3.shape[0] + df4.shape[0] )
        z = muhat/(phat* (1-phat) * (1/df3.shape[0] + 1/df4.shape[0]))
        pval = norm.sf(abs(z))*2
        results_z[i] = pval

        ## do chi-sq. test results:
        chi_tab = np.array([[df3.tool.sum(), df3.shape[0] - df3.tool.sum()], [df4.tool.sum(), df4.shape[0] - df4.tool.sum()]])
        chi_tab
        if (chi_tab[0] == chi_tab[1]).all(): p = 1.0
        else:
            try: chi2, p, dof, ex = chi2_contingency(chi_tab, correction=True)
            except: p = False
        results_c[i] = p

        ## do binomial test:
        p1hat = df3.tool.sum()/ df3.shape[0]
        p2hat = df4.tool.sum()/ df4.shape[0]
        if p1hat <= p2hat:
            pval1 = binom_test(df4.tool.sum(), df4.shape[0], p = p1hat, alternative = "greater")
            pval2 = binom_test(df3.tool.sum(), df3.shape[0], p = p2hat, alternative = "less")
        else:
            pval1 = binom_test(df4.tool.sum(), df4.shape[0], p = p1hat, alternative = "less")
            pval2 = binom_test(df3.tool.sum(), df3.shape[0], p = p2hat, alternative = "greater")
        results_b[i] = np.mean([pval1, pval2])

    return results_z, results_c, results_b


# z,c,b = run_binomial_tests(p1, p2, 5, 5, n=10)

def run_simulation_binary_tests(p1, p2, n = 100, threshold = 3.5,  seed=10, save_folder = False):
    """loops the bin test function for different numbers of users,
    stores and returns results
    if save_folder == True, stores results there with three .jsons, threshold is recorded in filename. """
    dz = {}
    dc = {}
    db = {}
    for m in [10, 15, 25, 30, 35, 40, 100]:
        results_z, results_c, results_b = run_binary_tests(p1, p2, m, m, n = 1000, threshold = threshold, seed = seed)
        pz = len([v for v in results_z.values() if v <= .05])/len(results_z)
        pc = len([v for v in results_c.values() if v <= .05])/len(results_c)
        pb = len([v for v in results_b.values() if v <= .05])/len(results_b)
        dz[m] = pz
        dc[m] = pc
        db[m] = pb

    if save_folder:
        jsonify(dz, os.path.join(save_folder,f'binary-z-score-{threshold}.json'))
        jsonify(dc, os.path.join(save_folder,f'binary-chi2-{threshold}.json'))
        jsonify(db, os.path.join(save_folder,f'binary-binomial-{threshold}.json'))

    return dz, dc, db

# dz, dc, db = run_simulation_bin_tests(p1, p2, n = 10, threshold = 3.5, savepath = False )
# dz
# dc
# db


def make_df(d,dz,dc,db, dz2, dc2, db2):
    columns = [x for x in sorted(dc.keys()) if x >5 and x<100]

    predf = {
        "$t$-test": [round(d[x],2) for x in columns],
        "$z$-test": [( round(dz[x],2), round(dz2[x],2) ) for x in columns],
        "$\chi^2$-test": [(round(dc[x],2), round(dc2[x],2)) for x in columns],
        "Bin.-test": [(round(db[x],2), round(db2[x],2)) for x in columns],
        }

    return  pd.DataFrame.from_dict(predf, orient = 'index', columns = columns)


if __name__ == '__main__':
    ## Make distributinos for generating data for two tools' videos:
    p1 = np.array([1,3,7,2,1])
    p1 = p1 / np.sum(p1)
    print(f"Distribution 1 for creating ratings data has (mu, var) = {get_mu_var(p1)}"  )

    p2 = np.array([1,1,5,6,4])
    p2 = p2 / np.sum(p2)
    print(f"\nDistribution 2 for creating ratings data has (mu, var) = {get_mu_var(p2)}"  )

    ## tests with ratings distribution for tool 1 having mean < ratings for tools 2:
    print(f"\n----      -----\nRunning tests with tool 1 ratings sampled from distribution 1, tool 2 ratings sampled from distribution 2\n\t (Note: recovering that tool 1 < tool 2 with 95% confidence in stats test is goal here)")

    folder_path = os.path.join(sim_path, "p1-p2-simulations/")
    if not os.path.isdir(folder_path): os.mkdir(folder_path)
    print(f"\n\tresults will be stored in {folder_path}")
    print(f"\n\tNOTE: all results in {folder_path} are the percent of times in the simulation the hypothesis test confirmed the right result. \n\tHigher is better in this folder's results")

    ## two-sided t test w/ different population variances:
    d = run_simulation_tt_test(p1, p2, n = 1000, equal_var = False, save_folder = folder_path, seed = 10)
    print(f"\n\t --- Results of Two-Sided T-Test with different population variances (Welch's test) is --- \n\t\t {d}")

    ## run binary tests with threshold = 3.5
    import time
    start = time.time()
    dz, dc, db = run_simulation_binary_tests(p1, p2, n = 100, threshold = 2.5,  seed=10, save_folder = folder_path)
    end = time.time()
    print (f"it took {(end-start)/60.} min")
    print(f"\n\t --- Results of binary test with threshold 2.5 (so 1,2,3--> 0, and 4,5--> 1) using --- \n\t\t z-test is\n\t\t{dz}\n\t\t chi2-test is\n\t\t{dc}\n\t\t binomial-test is\n\t\t{db}")


    ## run binary tests with threshold = 2.5
    dz2, dc2, db2 = run_simulation_binary_tests(p1, p2, n = 100, threshold = 3.5,  seed=10, save_folder = folder_path)
    print(f"\n\tResults of binary test with threshold 3.5 (so 1,2--> 0, and 3,4,5--> 1) using \n\t z-test is\n\t\t\t{dz}\n\t chi2-test is\n\t\t\t{dc}\n\t binomial-test is\n\t\t\t{db}")

    # df_1 = make_df(d,dz,dc,db, dz2, dc2, db2)
    #
    # print( df_1.to_latex() )



    ###------####
    ## tests with ratings distribution for tool 1 = tool 2
    print(f"---     -----\n\nNow running tests with tool 1 and tool 2 ratings using distribution 2...")
    folder_path = os.path.join(sim_path, "p2-p2-simulations/")
    if not os.path.isdir(folder_path): os.mkdir(folder_path)
    print(f"\n\tResults will be stored in {folder_path}")
    print(f"\n\tNOTE: all results in {folder_path} are the percent of times in the simulation the hypothesis test confirmed the wrong result. \n\tLower is better in this folder's results")

    d = run_simulation_tt_test(p2, p2, n = 1000, equal_var = True, save_folder = folder_path, seed = 10)
    print(f"\n\tResults of Two-Sided T-Test with same population variances is \n\t\t {d}")

    ## run binary tests with threshold = 3.5
    dz, dc, db = run_simulation_binary_tests(p2, p2, n = 100, threshold = 2.5,  seed=10, save_folder = folder_path)
    print(f"\n\tResults of binary test with threshold 2.5 (so 1,2,3--> 0, and 4,5--> 1) using \n\t\t z-test is\n\t\t\t{dz}\n\t\t chi2-test is\n\t\t\t{dc}\n\t\t binomial-test is\n\t\t\t{db}")

    ## run binary tests with threshold = 2.5
    dz2, dc2, db2 = run_simulation_binary_tests(p2, p2, n = 100, threshold = 3.5,  seed=10, save_folder = folder_path)
    print(f"Results of binary test with threshold 3.5 (so 1,2--> 0, and 3,4,5--> 1) using \n\t\t z-test is\n\t\t\t{dz}\n\t\t chi2-test is\n\t\t\t{dc}\n\t\t binomial-test is\n\t\t\t{db}")

    # df_2 = make_df(d,dz,dc,db, dz2, dc2, db2)
    #
    # print( df_2.to_latex() )
