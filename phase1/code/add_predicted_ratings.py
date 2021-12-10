# Code source: Jaques Grobler
# License: BSD 3 clause

from sklearn.metrics  import r2_score, mean_squared_error
from sklearn import datasets, linear_model, tree, model_selection
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import math
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR
from pathlib import Path
import warnings

import ast, sys
from .utils import *

def make_predictions(AK, results_dir):
    warnings.filterwarnings(action='ignore', category=UserWarning)

    AK_part = AK.df_r_filled.loc[:,('user_id', 'tool_id', 's_overall_score', 'b_overall_score')]
    AK_part.to_csv(Path(results_dir, "AK.csv"))

    int_ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    # ratings = pd.read_csv('../results/real/4_29_20/Stage2/df_r_filled.csv')
    ratings = AK.df_r_filled.copy()
    ratings = ratings.loc[:,('user_id','tool_id','Overall Score','aspect_id_0','aspect_id_1','aspect_id_2','aspect_id_3','aspect_id_4','aspect_id_5','aspect_id_6')]
    ratings = ratings.rename(columns={'Overall Score': 'overall_score'})
    ratings['b_overall_score'] = ratings['overall_score']
    ratings['s_overall_score'] = ratings['overall_score']

    ratings.loc[(~ratings['s_overall_score'].isin(int_ratings)), 's_overall_score'] = None
    vals = ratings[ratings['s_overall_score'].notnull()]
    vals_x = vals.loc[:,('aspect_id_0','aspect_id_1','aspect_id_2','aspect_id_3','aspect_id_4','aspect_id_5')]
    vals_y = vals[['s_overall_score']]
    # train_x, test_x, train_y, test_y = train_test_split(vals_x, vals_y)


    unknown = ratings.copy()
    unknown = unknown[unknown['s_overall_score'].isnull()]
    unknown_x = unknown[['aspect_id_0','aspect_id_1','aspect_id_2','aspect_id_3','aspect_id_4','aspect_id_5']]

    reg_list = [linear_model.LinearRegression(), CatBoostRegressor(verbose=False),
                linear_model.SGDRegressor(), KernelRidge(), linear_model.ElasticNet(), linear_model.BayesianRidge(),
                GradientBoostingRegressor(), SVR(), RandomForestRegressor()]

    reg_str = ['linear_reg', 'catboost', 'sgd', 'kernel_ridge', 'elastic_net', 'bayesian_ridge', 'gradient_boost', 'svr', 'random_forest']

    model_path = Path(results_dir, "model_results.csv")

    ## do MSE of Adom. & Kwon:
    akmse = AK.cross_validation(
        p=AK.optimal_params["p"],
        naive_dist=AK.optimal_params["naive_dist"],
        a=AK.optimal_params["a"],
        b=AK.optimal_params["b"],
        n_folds = 5,
        multiprocess = True,
        verbose = True)

    lowest_mse = akmse ### instantiate lowest score yet.
    best_model = "AK" ### instantiate best model yet.
    all_models = [["AK", akmse] ] ### instantiate all models list

    for x, reg in enumerate(reg_list):

        ## xvalidation and store ave mse:
        cross = cross_val_score(reg, vals_x, vals_y, scoring='neg_mean_squared_error')
        cross_avg = abs(sum(cross) / len(cross))
        # model_info = [reg_str[x], r2_score(test_y, preds), mean_squared_error(test_y, preds), math.sqrt(mean_squared_error(test_y, preds)), cross_avg]
        model_info = [reg_str[x],  cross_avg]

        all_models.append(model_info)
        if cross_avg < lowest_mse:
            lowest_mse = cross_avg
            best_model = reg_str[x]

        ## fit and predict unknown overall scores for all regressors and write a csv.
        reg.fit(vals_x, vals_y.values.ravel())
        preds = reg.predict(unknown_x)
        f_name = reg_str[x] + '.csv'
        unknown.loc[:,'s_overall_score'] = preds
        all_scores = pd.concat([unknown, vals])
        all_scores = all_scores.loc[:,('user_id', 'tool_id', 's_overall_score', 'b_overall_score')]
        csv_path = Path(results_dir, f_name)
        all_scores.to_csv(csv_path)

        print('Finished {} predictions'.format(reg_str[x]))
    
    models = pd.DataFrame(all_models, columns=['model', 'cross_val_mse'])
    models.to_csv(model_path)
    print('Best model is {0} with average mean squared error of {1}'.format(best_model, lowest_mse))
    return best_model + '.csv'


if __name__ == '__main__':
    dataset = "5_19_21"
    dataset_dir = f"./results/real/{dataset}"
    try: os.path.isdir(dataset_dir)
    except: dataset_dir = "../results/real/{dataset}"

    AK = unpickle(os.path.join( dataset_dir , "Stage2", "AK.pkl") )
    results_dir = os.path.join(dataset_dir, "Stage3")

    make_predictions(AK, results_dir)
