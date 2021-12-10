"""
This is multiprocessed (parallelized) so (I think) it has to be run from command line inside the `if __name__ == '__main__':` block b/c python won't let subproccesses spawn subproccesses

For multiprocessing it will look up the number of cores on your computer and use n-1 of them.

this code has a single class, AK_predictor()
it takes inputs:
    df_u_raw = dataframe with users data --> read in from raw data .csv
    df_r_raw = dataframe with ratings data --> read in from raw data .csv
    N = positive int, number of samples for montecarlo sim when computing distances over unknown ratings. default is N = 10000

It takes a  couple minutes to initiate b/c it builds all needed similarirty matrices
(recommend pickling it after instatiation)
To train it use the self.param_grid_search(n_folds = 20, results_folder = results_folder) method. it'll grid search all possible inputs and write outputs to results_folder.
    - this also takes a minute or two.
    - this instantiates self.results, and self.optimal_params()
(recommend pickling it after training)

To fill in the missing values of the ratings data use self.fill_r, and pass it parameter values (ideally self.optimal_params)
    - This will instantiate the attribute self.df_r_filled (data frame of ratings + predicted ratings instead of nans)
    - it will write this to a csv in the passed results_folder.

Basic useage, see the main block at bottom.

---
more info on what it does:
it makes df_r form df_r_raw (raw ratings data) by converting the comments to a float using sentiment analysis tools (average of RoBERTa and VADER)


- sim_user_rankings: It make a matrix of similarities of the users based on the users rankings of aspects  (df_u) see citation for Kendall Tau Correlation, bottom of doc block
- sim_user_ratings: it makes a matrix of similarites of the users based on their ratings (df_r). it integrates over unknown (missing) ratings using a uniform prior distribution
- sim_item_ratings: it makes a matrix of similarites of the items based on their ratings (df_r). it integrates over unknown (missing) ratings using a uniform prior distribution

Similarities of users (items) from ratings are simply exp(-|| r(i,:,:) - r(j,: : )||_p )
    - we have 8 distances for these similarities given by p in [0, 1, 2, np.inf] x naive_dist in [True, False]
    - naive_dist = True means ignores unknown ratings. e.g., if two users have the same ratings on only a few items, it gives them distance 0
    - naive_dist = False uses a uniform prior on ratings {1, .., 5} for unknown ratings and computes the expected difference in ratings when computing similarities.
Note that the first similarity on users from aspect ranks is Kendall Tau Correlation

Fix a p and a naive_dist value (so a distance on rating vectors)
From a similarity on users, we define an unseen rating (aspect k) from user u, item j as follows,
    r(u,j,k): sum_{v: r(v,j,k) exists} r(v,j,k) sim(u,v) / z_u , w/ z_u = sum_same set sim(u,v)
Analogously for similarities on items.

Now armed with these similaritis and abilities to predict rankings, we defines unknown ratings as r(u,i): = convex sum of three predictions for r(u,i):
    -  user_similarity_by_rank -- weight is parameter a in [0,1]
    -  user_similarity_by_ratings -- weight is parameter b in [a, 1]
    -  item_similarity_by_ratings -- weight is 1-a-b

the param_grid_search() method searches over all values of a, b, p, naive_dist and for each one it computes the 20-fold cross-validation average error (macro-averaged)
the best tuple are our optimal parameters.


Reference for the similarity by ranking the aspects:
^ Kendall Tau Reference: W.R. Knight, “A Computer Method for Calculating Kendall’s Tau with Ungrouped Data”, Journal of the American Statistical Association, Vol. 61, No. 314, Part 1, pp. 436-439, 1966.


- RAB 2021-04-29
"""

# imports
import pandas as pd
import ast, sys
from .utils import *
import numpy as np
from numpy.linalg import norm
from scipy.stats import kendalltau
from itertools import product
from collections import defaultdict

from .sentiment import roBERTa_polarity, vader_polarity
os.environ["TOKENIZERS_PARALLELISM"] = "false" ## this turns off annoying warnings that get printed 1000 times...

from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer



class AK_predictor:
    """
    Inputs: df_u_raw = dataframe with users data
            df_r_raw = dataframe with ratings data
            N = positive int, number of samples for montecarlo sim when computing distances over unknown ratings. default is N = 10000
    Attributes:
        - df_u_raw, df_r_raw = original inputs with column's added
        - df_u, df_r = augmented dataframes with the needed subset of colums for users (includes aspect rankings) and ratings (users x items x aspects + overall ratings)
        - _user_id_index = {u: i for i,u in enumerate(self.df_r.user_id.unique())} ## this is the map between {user_id: r's row index}--> same for similarity matricies that are indexed from 0 to num_users.
        - _tool_id_index = {v: j for j,v in enumerate(self.df_r.tool_id.unique())} ## this is the map between {tool_id: r's column index}--> same for similiarity matricies that are indexed from 0 to num_items.
        - df_r_filled = same as df_r but has all nan values filled in. this is blank until self.fill_r() is run.
        - num_users, num_tools = number of users, tools
        - r0 = numpy array with ratings, nan values for missing ratings. of shape num_users x num_tools x 8 ratings.
            r0[user, tool, 0] = overall ratings, r0[user, tool, 1:] = aspect ratings
        - nan_dists = dict giving probability distributions of distances between ratings a and b where at least one of a,b are unseen (nan values). It uses a uniform prior on unseen ratings in the {1, .., 5}
        - sim_u_rank = numpy array of shape (num_users x num_users), triangular, 1's on diag. gives the Kendall tau similiarity (0 to 1 scale) of users based on their aspect rankings in the user table df_u_raw0
        - sim_u_ratings = dict of form { (p, bool): similarity matrix of users based on ratings with Lp distance, and naive_dist = bool} (p in [0, 1, 2, np.inf])
        - sim_i_ratings = dict of form { (p, bool): similarity matrix of users based on ratings with Lp distance, and naive_dist = bool} (p in [0, 1, 2, np.inf])
        - p_optimal  = None until grid search is run to train it
    Methods: see each method's code block
    """

    def __init__(self, df_u_raw, df_r_raw, N = 10000):
        print("Beginning initialization...")
        self.df_u_raw = df_u_raw
        self.df_r_raw = df_r_raw
        self.N = N

        self.optimal_params = {} ## not instantiated here--found when running self.param_grid_search
        self.results = {} ## not instantiated here--adds a key (n_folds), and values (results for all combinations of parameters)

        self.df_u, self.df_r = self._make_dfs_from_raw()
        self.num_users = len(self.df_r.user_id.unique())
        self.num_tools = len(self.df_r.tool_id.unique())

        ## making r, numpy array of shape (num_users, num_tools, num_aspects + 1 (overall)),
        self._user_id_index = {u: i for i,u in enumerate(self.df_r.user_id.unique())} ## this is the map between {user_id: r's row index}--> same for similarity matricies that are indexed from 0 to num_users.
        self._tool_id_index = {v: j for j,v in enumerate(self.df_r.tool_id.unique())} ## this is the map between {tool_id: r's column index}--> same for similiarity matricies that are indexed from 0 to num_items.
        self.r0 = self._make_r0() ## numpy array of ratings, r0[user, tool, 0] = overall rating
        self.df_r_filled = None ## place holder to be filled by running self.fill_df_r

        ## similarity and user conditional probabilities based on how they ranked the aspect questions:
        self.sim_u_rank = self._make_user_similarity_by_rank()
        self.nan_dists = self._make_nan_dist_probs() ## needed for distance computations

        ## for multiprocessing need to define the pool object:
        processes = cpu_count()-1
        pool = Pool(processes=processes)

        ## making similarity matricies from ratings for differing p-norms, and naive/bayesian distance methods...
        args = [ (p, naive_dist) for p in [ 0, 1, 2, np.inf]
                for naive_dist in [ True, False] ]
        start = timer()
        print(f'\t\tInstantiating similarity matrix attributes from ratings.  This is multiprocessed using {processes} cores. \n\t\t This step takes a minute or two ...')
        sim_u_results = pool.starmap( self.make_sim_u_ratings, args )
        sim_i_results = pool.starmap( self.make_sim_i_ratings, args )
        end = timer()
        print(f'\t\t Done, elapsed time: {end - start}')
        ## put the results into the desired attributes:
        self.sim_u_ratings = { args[i] : sim_u_results[i] for i in range(len(args))}
        self.sim_i_ratings = { args[i] : sim_i_results[i] for i in range(len(args))}

        print("\tDone initializing!")

    @staticmethod
    def sentiment_score(text_string, verbose = True): ## need to double check this works with real data--> are we inputting the right kind of text/preprocessing is ok, etc.
        """
        Takes average of Vader and roBERTa_polarity (in [-1,1]), maps that to [0,5], then uses ceiling to map to {1,2,...,5}
        """
        try:
            if np.isnan(text_string): return np.nan
        except: pass
        try:
            text_string = text_string.strip()
            if not text_string:
                return np.nan
        except: pass
        if not text_string:
            return np.nan
        x = roBERTa_polarity(text_string)
        y = vader_polarity(text_string)

        if verbose and (np.abs(x-y) > .5):
            print(f'assign score manually, roBERTa_polarity = {x} vader_polarity = {y}, text is: \n\t{text_string}')
        sentiment = (x + y)/2 ## In [-1,1]
        sentiment = np.ceil( 5 * (sentiment + 1)/2 ) ## into [0,1], into [0,5], into {1,2,..,5}
        if sentiment == 0: sentiment = 1
        if sentiment == 6: sentiment = 5

        return sentiment


    def _make_dfs_from_raw(self, verbose = False):
        """
        adds cols to self.df_r_raw with numeric sentiment analysis of comments
        adds cols to sefl.df_u_raw with rankings converted from string(tuple) to jsut a tuple.

        returns corresponding dataframes with only cols needed. :
            df_r = ratings dataframe with aspect responses sorted, and free response converted to a float
            df_u = user dataframe with rankings converted to tuples (not strings)
        """

        for col in ["aspect_rank", "tool_rank"]: ## convert strings of tuples into just tuples:
            self.df_u_raw[col] = self.df_u_raw.apply(lambda row: ast.literal_eval(row[col + "ing"]), axis = 1)
        ## make attribute df_u
        df_u = self.df_u_raw[['user_id', 'familiarity', 'occupation', 'experience', 'name', 'aspect_rank', 'tool_rank']]

        self.df_r_raw["aspect_id_6"] = self.df_r_raw.apply(lambda row: self.sentiment_score(row["Free Response"], verbose = verbose), axis = 1) ## add column w/ numeric sentiment score
        ## make attribute df_r
        print(self.df_r_raw.index)
        df_r = self.df_r_raw[["user_id", "tool_id", "Overall Score"] + [f'aspect_id_{k}' for k in range(7)]] ## keep only the cols we want in df_r
        return df_u, df_r


    def _make_r0(self):
        """
        uses only self.num_users, .num_tools, and .df_r, dataframe with user_id, tool_id, and all questions answered on a 1-5 scale
                **Important that df_r has a row for each user x each tool (w/ nan rating values if that user didn't rank that tool)**
        Output: r = n,m,num_questions array of results. nan values in missing ratings.
                r[user, tool, 0] = overall ratings,
                r[user, tool, 1:] = aspect ratings
        """
        m = self.num_users
        n = self.num_tools
        r = np.empty((m,n,8)) ## ratings matrix, with 0th rating the overall rating (Q3), and ratings 1-7 the aspect ratings
        r[:] = np.nan

        for index, row in self.df_r.iterrows():
            i = self._user_id_index[row.user_id] ## converts from user_id to index
            j = self._tool_id_index[row.tool_id] ## converts from tool_id to index
            cols = list(self.df_r.columns)
            cols.remove("user_id")
            cols.remove("tool_id")
            for k, c in enumerate(cols):
                r[i,j,k] = row[c]
        return r


    def _make_user_similarity_by_rank(self):
        """
        Only uses  self.df_u = dataframe with user data. specifically it requires a column named "aspect_rank"
            Kendall Tao Correlation used.
            kt(x,y) gives values in [-1,1]. 1 if x=y and -1 if x = inverse(y). here we normalize it to be between 0 and 1

        Output: sim_rank - array, mxm, symmetric, 1's on diagonals,  gives similarity of users based on their rankings of the aspects
        """
        m = self.num_users
        sim_rank = np.ones((m,m))
        inverse_lookup = {i:u for u,i in self._user_id_index.items()} ## need to map the indices i = 0,.., num_users to the u = user_ids
        l = sorted(inverse_lookup.keys())
        for ind, i in enumerate(l[:-1]): ## run thru i in l except the last element
            x = self.df_u[self.df_u.user_id == inverse_lookup[i]].iloc[0]["aspect_rank"]
            for j in l[ind+1:]: ## run thru j in l with j>i
                y = self.df_u[self.df_u.user_id == inverse_lookup[j]].iloc[0]["aspect_rank"]
                sim_rank[i,j] = (1 + kendalltau(x,y)[0] )/2
                sim_rank[j,i] = sim_rank[i,j]
        return sim_rank


    @staticmethod
    def distance_naive(x_,y_, p = 2):
        """
        Inputs: x_, y_ two numpy arrays of the same shape
                p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
        Computes ||x_-y_||_p / len(x_) (as though x_ and y_ are vectors) but ignores any components of these vectors that have a nan value.
        if all values are nan it returns 1.
        Output: float in [0,1]
        """
        def _remove_nans(x,y):
            """
            Input:  x,y: arrays of same shapes
            Output: x1, y1: 1-d arrays with the corresponding entries of x,y that are both not nan.
            (ignore any component where x or y is nan, then flatten what's left into a vector and return)
            """
            x = np.array(x)
            y = np.array(y)
            assert x.shape == y.shape

            return x[~np.isnan(x+y)], y[~np.isnan(x+y)]

        x = x_.copy().flatten()
        y = y_.copy().flatten()

        denom = len(x)
        (x1, y1) = _remove_nans(x,y)
        # if len(x1)>0: return norm(x1-y1, p ) / len(x1)
        if len(x1)>0: return norm(x1-y1, p ) / denom
        else: return 1 ## if no mutual ratings, return max distance.


    @staticmethod
    def _make_nan_dist_probs():
        """
        Returns dict of form {(a,b):  {  k: prob(|a-b| = k)} } where either a or b are nan
        """
        nan_dists = { ("nan", "nan") : {0 : (5/25), 1: (8/25),  2: (6/25), 3: (4/25), 4: (2/25)}}
        for a in range(1,6):
            nan_dists[(a,"nan")] = defaultdict(float)
            nan_dists[("nan", a)] = nan_dists[(a,"nan")] # symmetric
            for i in range(1,6):
                nan_dists[(a,"nan")][np.abs(a-i)] += 1/5

        return nan_dists


    def distance(self, x_, y_ ,  p = 2, N = None):
        """
        Input:  x_, y_: arrays of same shapes
                p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
                N = how many samples to use for the monte carlo simulation, if nothing passed, uses self.N
        Output: ||x-y||_p / len(x) (should be in the interval [0,4]), but we take the expected value over any unseen components  (nan values) w/ uniform prior (x[i] \in {1,2,3,4,5} each w/ probability 1/5).
            This is intractable (e.g., if there are 20 missing values and each has 5 different values it can take, there are 5^20 vectors to compute distances on)
            We use a monte carlo simulation
        """
        if not N:
            N = self.N

        x = x_.copy().flatten()
        y = y_.copy().flatten()

        I = np.where(np.isnan(x+y))[0] ## components where x[i] or y[i] is nan
        if len(I)==0: ## no nan values!
            return norm( x - y, p ) /len(x)

        # if not self.nan_dists:
            # self.nan_dists = _make_nan_dist_probs()

        w = (x-y).reshape(1, x.shape[0] ) ## column vector with x-y
        W = np.concatenate([w]*N, axis = 0) ## N rows, each are w.
        for i in I: # sample N values for w[i]
            if np.isnan(x[i]): a = "nan"
            else: a = x[i]
            if np.isnan(y[i]): b = "nan"
            else: b = y[i]

            vals, probs = list(zip(*self.nan_dists[ (a,b) ].items()))
            counts = np.random.multinomial(N, probs)
            samples = []
            for j in range(len(vals)): samples += [vals[j]] * counts[j]
            np.random.shuffle(samples)
            W[:,i] = samples ## replaces nan values with the N random samples for w[i] in W

        return sum([norm( W[i,:], p)/N  for i in range(N)])/len(x)


    def make_sim_u_ratings(self, p = 2, naive_dist = False, N = None, r = None):
        """
        Input:  r = np.array of shape num_users, num_tools, num_ratings (0'th rating index for overall ratings. 1: indices for aspects)
                    usually r = self.r0
                    if r is not passed (r = None), then it uses self.r0
                p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
                naive_dist = bool. True uses self.distance_naive(), False uses self.distance()
                N = positive int, if nothing passed, uses self.N
                    only used if naive_dist = False, in which case it is the monte carlo simulation number.

        Note, this is all working in the matrix indices space, so indexes go from 0 to num_users. self._user_id_index  maps from user_id to the indices in these matrices.

        Output: sim_u_ratings = np.array of shape num_users x num_users, giving the similarity (np.exp(-dist(u,v))) of users (positive definite, 1's on diag)
        """
        # if not naive_dist and not nan_dists: nan_dists = _make_nan_dist_probs()
        if not r:
            r = self.r0

        if not N:
            N = self.N

        m = self.num_users
        sim_u_ratings = np.ones((m,m))
        for i in range(m-1):
            for j in range(i+1,m):
                if naive_dist: sim_u_ratings[i,j] =  np.exp(-self.distance_naive(r[i,:,:], r[j, :, :], p = p)) ##1/(1+self.distance_naive(r[i,:,:], r[j, :, :], p = p))
                else:  sim_u_ratings[i,j] = np.exp(- self.distance(r[i,:,:], r[j, :, :], p = p, N = N))
                sim_u_ratings[j,i] = sim_u_ratings[i,j]
        return sim_u_ratings


    def make_sim_i_ratings(self, p = 2, naive_dist = False, N = None,  r = None):
        """
        Input:  r = np.array of shape num_users, num_tools, num_ratings (0'th rating index for overall ratings. 1: indices for aspects)
                    usually r = self.r0
                    if r is not passed (r = None), then it uses self.r0
                p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
                naive_dist = bool. True uses self.distance_naive(), False uses self.distance()
                N = positive int, if nothing passed, uses self.N
                    only used if naive_dist = False, in which case it is the monte carlo simulation number.

            Note, this is all working in the matrix indices space, so indexes go from 0 to  num_items.  self._tool_id_index maps from tool_id to the indices in these matrices.


        Output: sim_i_ratings = np.array of shape num_tools^2, giving the similarity (np.exp(-dist(i,j))) of items (positive definite, 1's on diag)
        """
        # if not naive_dist and not nan_dists: nan_dists = _make_nan_dist_probs()
        if not r:
            r = self.r0
        if not N:
            N = self.N

        n = self.num_tools
        sim_i_ratings = np.ones((n,n))
        for i in range(n-1):
            for j in range(i+1,n):
                if naive_dist: sim_i_ratings[i,j] = np.exp(-self.distance_naive(r[ :, i, :], r[ :, j, :], p = p)) ## 1/(1 + self.distance_naive(r[ :, i, :], r[ :, j, :], p = p))  #
                else: sim_i_ratings[i,j] = np.exp(-self.distance(r[ :, i, :], r[ :, j, :], p = p, N = N))
                sim_i_ratings[j,i] = sim_i_ratings[i,j]
        return sim_i_ratings


    @staticmethod
    def _predict_nans_from_sim(r, s, user = True, predict_overall_rating = False ):
        """
        Description:    Uses smilarity matrix and user must specify w/ bool `user` whether this similarity matrix is for users (mxm) or items  (nxn) )
                        to predict the unknown (nan values) in r. To infer each unobserved rating (nan values in r) it uses the weighted sum of the other users'/items' (as indicated by bool input "user")
                        ratings for the same user,item, aspect. The weights are conditional probabilities proportional to the user/item similarities.

                    Note, this is all working in the matrix indices space, so indexes go from 0 to num_users or num_items. the self._user_id_index and self._tool_id_index maps from user_id and tool_id to the indices in these matrices.

        Input:  r = ratings matrix to use
                s = numpy array, similarity matrix
                user = True if prob is from a user similarity matrix,
                     = False if prob is from an item similarity matrix.
                predict_overall_rating = bool, if True it predicts r[:,:,0] -- the overall ratings for each user x item. If false it leaves r[:,:,0] untouched.
        Output: r_new = matrix that is identical to r on the observed entries but updated to include inferred ratings for the unobserved (nan) ratings
                        based on the similarity matrix passed.
        """
        r_new = r.copy() ## to be populated with predictions
        nans = list(zip(*np.where(np.isnan(r)))) ## nans = indices where nan values exist (only replace the ratings that were not observed == nans of original r matrix)
        m, n, n_a  = r.shape ## number of users, num items, num aspects
        for i,j,k in nans:
            if user: ## prob matrix given is for users, not items
                x = r[:,j,k] ## all users' ratings for tool j, aspect k
                if np.all(np.isnan(x)): x = np.ones_like(x)*3 ## if no other items have been rated by this tool, the all other itemms get average rating
                q = s[i,:] ## sim of all other users to user i

            else: ## prob matrix given is for items
                x = r[i,:,k] ## this users rating of the same aspect across all tools
                if np.all(np.isnan(x)): x = np.ones_like(x)*3 ## if no other user has rated this tool, the all get average rating
                q = s[j,:] ## sim of all other users to tool j

            z = np.sum(q[~np.isnan(x)]) ## denominator is sum of s over all other users/items that have rated this item/ been rated by this user.

            if (predict_overall_rating and k == 0) or k>0: ## only update r_new if k >0 (it is an aspect rating, not an overall rating), or k = 0 (it's an overall rating) and predict_overall_rating == True
                r_new[i,j,k] = np.nansum(x*q)/z ## nansum only sums those x[i]*q[i] that are not nans.

        return r_new


    def predict_nans(self,  p = 2, naive_dist = False, a = 1/3, b = 1/3, predict_overall_rating = False, r = None ):
        """
        Input:  p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
                naive_dist = bool. True uses distance_naive(), False uses distance()
                a, b = non-negative floats such that a + b \in [0,1]. they give weights for predictions by the three similarities (user by aspect rank, user by ratings, item by ratings )
                predict_overall_rating = bool, if True it predicts r[:,:,0] -- the overall ratings for each user x item. If false it leaves r[:,:,0] untouched.
                r = numpy array of ratings. If None is passed (default) this uses r0.
                    (The use case for this argument is cross validation. copy r0 to r,  make a few known values nans in r, then pass r so it predicts those known ones and we can test accuracy)
        Description: fills in the nan values of r (r0 if r not passed) using expected value according to three different similarities
                    (user by aspect rank, user by ratings, item by ratings ),
                    Returns the weighted average of the three guesses.
                    weights given by a, b, 1-a-b, repectively, for these three values.
        Output: r_new = a*r_u1 + b * r_u2 + (1-a-b)* r_i
        """
        if isinstance(r, type(None)): r = self.r0 ## if you don't pass an r, just use self.r0.
        r_u1 = self._predict_nans_from_sim( r, self.sim_u_rank, user = True, predict_overall_rating = predict_overall_rating)
        r_u2 = self._predict_nans_from_sim( r, self.sim_u_ratings[ p, naive_dist ], user = True, predict_overall_rating = predict_overall_rating)
        r_i = self._predict_nans_from_sim(r, self.sim_i_ratings[ p, naive_dist ], user = False, predict_overall_rating = predict_overall_rating)
        return  a*r_u1 + b * r_u2 + (1-a-b)* r_i


    def fill_df_r(self,  p = 2, naive_dist = False, a = 1/3, b = 1/3, results_folder = None):
        """
        Makes and saves if results_folder passed self.df_r_filled--the ratings table all filled in using the parameters passed!
        """

        r = self.predict_nans(p, naive_dist, a, b, predict_overall_rating = True) ## has all the ratings filled in!
        df_r_filled = self.df_r.copy() ## copy to be instantiated!

        ## we need the column names for the aspect and overall ratings:
        cols = df_r_filled.columns.drop("user_id")
        cols =cols.drop("tool_id")

        def _nan_fill(row, r, cols):
            """
            creates a new row that's the same as the passed row, but has all nans filled with corresponding values from the r array.
            """
            u = int(row["user_id"])
            v = int(row["tool_id"])
            row_new = row.copy()
            for k, c in enumerate(cols):
                if np.isnan(row[c]):
                    row_new[c] = r[self._user_id_index[u], self._tool_id_index[v], k ] ## note here is we look up the user/tool_id in the dictionary that maps it to the index used in the matrices.
            return row_new

        self.df_r_filled = df_r_filled.apply(lambda row: _nan_fill(row, r, cols), axis = 1)

        if results_folder:
            path = os.path.join(results_folder, "df_r_filled.csv")
            print(f"Saving df_r_filled to {path}")
            self.df_r_filled.to_csv(path)

        return


    def compute_fold_error(self, fold, p, naive_dist, a, b):
        """
        Input:  fold = a subset of indices of r0 for which values exist (to be heldout, predicted, then used for computing prediction error of this fold)
                p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
                naive_dist = bool. True uses distance_naive(), False uses distance()
                a, b = non-negative floats such that a + b \in [0,1]. they give weights for predictions by the three similarities (user by aspect rank, user by ratings, item by ratings )

        Description: the kernel function in cross_validation(). Note that when performing cross validation we use the overall ratings and aspect ratings

        Output: mean squared error of predictions in this fold (sum |y_i - y_hat_i|^2 / len(fold))
        """
        r = self.r0.copy()
        r[tuple(zip(*fold))] = [np.nan]*len(fold) ## make nan values to fill in then check accuracy.
        r_new = self.predict_nans(p = p, naive_dist = naive_dist, a = a, b = b, predict_overall_rating = True, r = r) ## note this fills in the original nans nad the artificial ones, but we'll only check the error on the artificial ones.
        return (norm(r_new[tuple(zip(*fold))] - self.r0[tuple(zip(*fold))], 2)**2) / len(fold)


    def cross_validation(self, p = 2, naive_dist = False, a = 1/3, b = 1/3, n_folds = 20, multiprocess = False, verbose = False):
        """
        Input:  p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
                naive_dist = bool. True uses distance_naive(), False uses distance()
                a, b = non-negative floats such that a + b \in [0,1]. they give weights for predictions by the three similarities (user by aspect rank, user by ratings, item by ratings )
                n_folds = number of folds to use.
                multiprocess = bool. If true, it distributes the compute_fold_error() across all but 1 core. if false it doe not paralleize this set of computations

        Description: divides the known values of self.r0 into n_folds sets (folds) randomly.
                    Then for each fold it uses the rest of the values to predict those values and computes the fold's mean sq. error for those predictions.
                    then averages the mse values for the folds.

        Output: float, the average of each fold's mean sq. error is returned.
        """
        nans = list(zip(*np.where(np.isnan( self.r0)))) ## ignore these for this evaluation since we don't know ground truth
        vals = list(zip(*np.where(~np.isnan(self.r0)))) ## these are all known, so we'll guess them, then check our erros on them in folds.
        np.random.shuffle(vals)
        c = int(len(vals)/n_folds)

        if not multiprocess:
            if verbose: print(f"Beginning cross_validation of AK with {n_folds} folds, without multiprocessing...")
            folds = tuple(vals[i:i+c] for i in range(0, len(vals), c))
            fold_errors = []
            for fold in folds:
                fold_errors.append( self.compute_fold_error( fold, p = p, naive_dist = naive_dist, a = a, b = b) )

        if multiprocess:
            if verbose: print(f"Beginning cross_validation of AK with {n_folds} folds, with multiprocessing...")
            args = tuple([ vals[i:i+c] , p, naive_dist, a, b] for i in range(0, len(vals), c))

            processes = cpu_count() - 1
            print(f"This is multiprocessed using {processes} cores. \n\t\t This step takes a minute or two ...")
            pool = Pool(processes=processes)
            start = timer()
            fold_errors = pool.starmap( self.compute_fold_error, args)
            end = timer()
            if verbose: print(f'\tcross_validation of  AK multiprocess elapsed time: {end - start}')

        mse = np.sum(fold_errors)/len(fold_errors)
        if verbose: print(f"\tFinished cross_validation() with MSE = {mse}")
        return mse


    def param_grid_search(self, n_folds = 20, results_folder = None):
        """
        Input:  p = lp norm parameter (p = 0, count non-zero entries, p = 1,2 are usual lp nroms, p = np.inf for infinity norm)
                naive_dist = bool. True uses distance_naive(), False uses distance()
                n_folds = number of folds to use for cross validation
                results_folder = string path variable to folder to save results csv and json

        Description
        Output:
        """
        ## define args for using a mapping a single function to them:
        args = tuple( (p, naive_dist, a, b, n_folds)
                    for p in [0,1,2,np.inf]
                    for naive_dist in [True, False]
                    for a in np.arange(0,1.05, .1)
                    for b in np.arange(0,1-a,.1)
                )

        processes = cpu_count()-1 ## define how many cores to use
        pool = Pool(processes=processes) ## this is the multiprocessing object
        start = timer()
        print(f'Starting grid search using {processes} cores. \nBeginning {len(args)} argument tuples x {n_folds}-fold cross validation... ')
        res = pool.starmap(self.cross_validation, args) ## maps the function to the arguments list and puts * in front of each
        end = timer()
        print(f'\tgrid search elapsed time: {end - start}')
        ## take list of results and put them back into form needed:
        results = [{"p": args[i][0], "naive_dist": args[i][1], "a": args[i][2], "b": args[i][3], "ave-error": res[i] } for i in range(len(args))]

        df_results = pd.DataFrame(results) ## make a dataframe
        results = sorted(results, key = lambda x: x["ave-error"] ) ## sort so we know the optimal params.

        self.optimal_params = results[0] ## store optimal params as attribute
        self.results[n_folds] = results ## store all results for n_folds as attribute

        print(f"Optimal Parameters found! self.optimal_params = \n{results[0]}")

        if results_folder:
            jsonpath = os.path.join(results_folder, f"{n_folds}-fold-results.json")
            csvpath = os.path.join(results_folder, f"{n_folds}-fold-results.csv")
            try:
                jsonify(results, jsonpath)
                df_results.to_csv(csvpath)
                print(f"Saved results to\n\t{jsonpath}\n\tand\n\t{csvpath}")
                return results, df_results
            except:
                pass
            if not os.path.isdir(folder):
                try:
                    os.mkdir(folder)
                    jsonify(results, jsonpath)
                    df_results.to_csv(csvpath)
                    print(f"Saved results to\n\t{jsonpath}\n\tand\n\t{csvpath}")
                    return results, df_results
                except:
                    print(f"results_folder argument is not valid! \n\t  argument results_folder = {results_folder}")
                    pass
        return results




if __name__ == '__main__':
    ## read in data:
    # folder = os.path.join('data', "toy_data", "realistic", "with_null", "parsed")
    # df_r_raw0 = pd.read_csv(os.path.join(folder, "set1_ratings_table.csv"))
    # df_u_raw0 = pd.read_csv(os.path.join(folder, "set1_user_table.csv"))
    ## read in data:
    folder = os.path.join('data', "toy_data", "clear_winners", "parsed")
    df_r_raw0 = pd.read_csv(os.path.join(folder, "clear_winners_ratings.csv"))
    df_u_raw0 = pd.read_csv(os.path.join(folder, "clear_winners_users.csv"))

    ## set paths for where to save stuff:
    results_folder = os.path.join("results", "bobby-toy-results-clear-winners")
    if not os.path.isdir(results_folder): os.mkdir(results_folder)
    picklepath = os.path.join(results_folder, "test-AK.pkl")

    ## instantiate class, or unpickle.
    AK = AK_predictor(df_u_raw0, df_r_raw0)
    # AK = unpickle(picklepath)

    ## save it b/c that takes a while!
    print(f"pickling AK into {picklepath}")
    picklify(AK, picklepath)

    ## train it and store all results:
    results, df_results = AK.param_grid_search(n_folds = 20, results_folder = results_folder)

    print(f"pickling updated AK into {picklepath}") ## save it b/c that takes a while:
    picklify(AK, picklepath)

    print(f"-- Now predicting unknown ratings using the optimal parameters -- ")
    AK.fill_df_r(p = AK.optimal_params['p'],
                naive_dist = AK.optimal_params['naive_dist'],
                a = AK.optimal_params["a"],
                b =AK.optimal_params["b"],
                results_folder = results_folder)





    ## some practice w/ these new methods/attributes:
    # print(f"printing AK.cond_prob_u_ratings[ p = 0, naive_dist = False]\n{AK.cond_prob_u_ratings[ 0, True]}")
    # r = AK.predict_nans(p = 0, naive_dist = True)
    # print(f"predicted r:\n{r}")
    # ave_error = AK.cross_validation(p = 0, naive_dist = True, n_folds = 10, multiprocess = False, verbose = True)
