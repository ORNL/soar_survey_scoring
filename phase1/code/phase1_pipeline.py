import anyconfig
import click
import pandas as pd
import ast, sys, os
from .utils import *

from .utils import *
from .parse_phase1 import parse as _parse
from .ak_inference import AK_predictor
from .add_predicted_ratings import make_predictions
from pathlib import Path
from .Stage4 import sams_stuff
from .stats_visualization import ashleys_stuff

_data_dir_help = (
    "Path to the directory containing all of our raw datasets. IE '../data/real/"
)
_results_dir_help = "Path to the directory containing all of our processed datasets. IE '../results/real/"
_dataset_help = "Subdirectory from within data_dir and results_dir to read/save our raw/processed data."
_additional_config = (
    "An optional extra yaml/json configuration file for further configuring each stage of the pipeline. Could be useful for running one-off testing without overwriting important files."
)

_default_stage_config = {
    "aspect_ranking_in": "UserCapabilitiesRanked.csv",
    "demographics_in": "UserDemographics.csv",
    "ratings_in": "UserResponses.csv",
    "users_out": "Users.csv",
    "tools_out": "Tools.csv",
    "ratings_out": "Ratings.csv",
    "aspect_ids_out": "Aspects.csv",
    "stage1_subdir": "Stage1",
    "stage2_subdir": "Stage2",
    "stage3_subdir": "Stage3",
    "stage4_subdir": "Stage4",
    "stage5_subdir": "Stage5",
    "demographics_blacklist": "UserDemographics_blacklist.csv",
    "aspect_ranking_blacklist": "UserCapabilitiesRanked_blacklist.csv",
    "ratings_blacklist": "UserResponses_blacklist.csv",
}


def parse(data_dir, results_dir, dataset, **kwargs):
    stage1_kwarg_keys = {
        "aspect_ranking_in",
        "demographics_in",
        "ratings_in",
        "users_out",
        "tools_out",
        "ratings_out",
        "aspect_ids_out",
        "demographics_blacklist",
        "aspect_ranking_blacklist",
        "ratings_blacklist",
    }

    # make the output dir if it doesn't exist
    stage1_output_dir = Path(results_dir, dataset, kwargs["stage1_subdir"])
    if not stage1_output_dir.exists():
        os.makedirs(stage1_output_dir)

    # setup kwargs to be passed to _parse by completing the paths
    for k in kwargs:
        if k in stage1_kwarg_keys and k.endswith("_in"):
            kwargs[k] = Path(data_dir, dataset, kwargs[k])
        elif k in stage1_kwarg_keys and k.endswith("_out"):
            kwargs[k] = Path(results_dir, dataset, kwargs["stage1_subdir"], kwargs[k])
        elif k in stage1_kwarg_keys and k.endswith("_blacklist"):
            kwargs[k] = Path(data_dir, kwargs[k])
            if not os.path.exists(kwargs[k]):
                kwargs[k] = None

    return _parse(**kwargs)


def predict_missing(ratings_df, demographics_df, results_dir, dataset, stage_config):
    df_r_raw0 = ratings_df.reset_index()
    df_u_raw0 = demographics_df.reset_index()
    df_u_raw0[["aspect_ranking", "tool_ranking"]] = df_u_raw0[
        ["aspect_ranking", "tool_ranking"]
    ].applymap(lambda x: str(x))
    testname = dataset
    # folder = os.path.join('data', "real",testname, "Parsed")
    folder = Path(results_dir, testname, stage_config["stage1_subdir"])
    if not folder.exists():
        os.makedirs(folder)
    # df_r_raw0 = pd.read_csv(os.path.join(folder, "Ratings.csv"))
    # df_u_raw0 = pd.read_csv(os.path.join(folder, "Users.csv"))

    ## set paths for where to save stuff:
    results_folder = Path(results_dir, dataset, stage_config["stage2_subdir"])
    if not results_folder.exists():
        os.makedirs(results_folder)
    picklepath = Path(results_folder, "AK.pkl")

    ## instantiate class, or unpickle.
    AK = AK_predictor(df_u_raw0, df_r_raw0)
    # AK = unpickle(picklepath)

    ## train it and store all results:
    results, df_results = AK.param_grid_search(
        n_folds=20, results_folder=results_folder
    )

    print(f"pickling updated AK into {picklepath}")  ## save it b/c that takes a while:
    picklify(AK, picklepath)

    print(f"-- Now predicting unknown ratings using the optimal parameters -- ")
    AK.fill_df_r(
        p=AK.optimal_params["p"],
        naive_dist=AK.optimal_params["naive_dist"],
        a=AK.optimal_params["a"],
        b=AK.optimal_params["b"],
        results_folder=results_folder,
    )
    picklify(AK, picklepath)
    return AK


@click.command()
@click.option("-d", "--data-dir", type=str, required=True, help=_data_dir_help)
@click.option("-r", "--results-dir", type=str, required=True, help=_results_dir_help)
@click.option("-s", "--dataset", type=str, required=True, help=_dataset_help)
@click.option("-c", "--additional-config", type=str, help=_additional_config)
#@click.option("--resume-stage", type=int, help="Resume from a stage if for some reason one fails.")
def cli(data_dir, results_dir, dataset, additional_config):
    """
    Run the phase 1 scoring pipeline. Raw results are read from
    'data_dir/dataset' and written to 'results_dir/dataset'.

    Example instatiation:
    python3 phase1_pipeline.py -d ../data/real/ -r ../results/real/ -s 4_29_20
    """

    ## CONFIG SETUP
    stage_config = _default_stage_config

    # Load additional configuration and merge with defaults
    if additional_config is None:
        additional_config = {}
    else:
        additional_config = anyconfig.load(additional_config)
    stage_config.update(additional_config)

    ## PARSING STAGE

    ## tool_table: tool name -> id conversion
    ## aspect_ids: aspect_name -> question number -> question text -> aspect id
    ## demographics_df: user name -> id -> rankings -> other demographic info
    ## ratings_df: meat and potatoes
    tool_table, aspect_ids, demographics_df, ratings_df = parse(
        data_dir, results_dir, dataset, **stage_config
    )

    ## END PARSING CODE

    ## MISSING DATA PREDICTION
    AK = predict_missing(
        ratings_df, demographics_df, results_dir, dataset, stage_config
    )


    ## Savannah's imports here to run w/ filled_ratings_df

    stage3_res = Path(results_dir, dataset, stage_config["stage3_subdir"])
    if not stage3_res.exists():
        os.makedirs(stage3_res)

    best_model = make_predictions(AK, stage3_res)
    best_model_csv = Path(stage3_res, best_model)
    print(best_model_csv)

    ## Sam's graph/ranking stuff here to run w/ all the things!
    # sam you don't use this
    stage4_res = Path(results_dir, dataset, stage_config["stage4_subdir"])
    if not stage4_res.exists():
        os.makedirs(stage4_res)

    sams_stuff(data_dir, dataset, results_dir, best_model_csv)

    #ashleys_stuff(data_dir, dataset, results_dir)

if __name__ == "__main__":
    cli()
