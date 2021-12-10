import click
import logging
import numpy as np
import pandas as pd


def not_at_least_one_filled(row):
    return any(all(row[column] for column in group) for group in at_least_one_filled)


def row_in_df(row, df):
    for df_row in df.iterrows():
        _break = False
        for i in df_row[1].iteritems():
            if row[i[0]] != i[1]:
                _break = True
                break
        if _break:
            continue
        else:
            return True
    return False


def apply_blacklist(df, blacklist):
    # df = df.astype('int64', errors='ignore')
    # blacklist = blacklist.astype('int64', errors='ignore')
    # df.drop_duplicates(keep="first", inplace=True)
    # blacklist.drop_duplicates(keep="first", inplace=True)
    # blacklist["dup"] = blacklist.duplicates(keep=False)
    # df["dup"] = df.duplicates(keep=False)
    return df.loc[df.apply(lambda x: not row_in_df(x, blacklist), axis=1)]
    # tmp = pd.concat([df, blacklist])
    # tmp = tmp.applymap(lambda x: np.int64(x) if type(x) not in [str] and not np.isnan(x)  else x)
    # tmp["dup"] = tmp.duplicated(keep=False)
    # tmp.to_csv("dup.csv")
    # print(f"Len1: {len(df.index)} Len2: {len(blacklist.index)} New length: {len(tmp.index)}")
    # return tmp.drop_duplicates(keep=False)


def preprocess_df(df, column_types):
    for col in df:
        df[col] = df[col].apply(
            lambda x: column_types[col][1]
            if x == "can't tell"
            or (type(x) in [float, np.float64, np.float32] and np.isnan(x))
            else x
        )
        df[col] = df[col].astype(column_types[col][0])
    return df


ratings_column_types = {
    "Title": ["object", ""],
    "Survey": ["object", ""],
    "Tool": ["object", ""],
    "Q1Answer": ["object", ""],
    "Q2Answer": ["object", ""],
    "Q3Answer": ["int64", -1],
    "Q4Answer": ["int64", -1],
    "Q5Answer": ["int64", -1],
    "Q6Answer": ["int64", -1],
    "Q7Answer": ["int64", -1],
    "Q8Answer": ["int64", -1],
    "ToolRank": ["int64", -1],
    "Q9Answer": ["int64", -1],
    "Q10Answer": ["object", ""],
    "SurveyUser": ["object", ""],
    "SurveyStatus": ["object", ""],
}

at_least_one_filled = [
    [
        "Question4",
        "Question5",
        "Question6",
        "Question7",
        "Question8",
        "Question9",
    ]
]

demographics_column_types = {
    "Title": ["object", ""],
    "Survey": ["object", ""],
    "Q1Answer": ["int64", -1],
    "Q2Answer": ["object", ""],
    "Q3Answer": ["float64", -1.0],
    "SurveyUser": ["object", ""],
    "SurveyStatus": ["object", ""],
}

aspect_column_types = {
    "CapabilityLabel": ["object", ""],
    "CapabilityValue": ["object", ""],
    "Survey": ["object", ""],
    "SurveyUser": ["object", ""],
    "CapabilityRank": ["int64", -1],
}

aspect_to_answer = {
    "Playbooks": "Q6Answer",
    "rank / score": "Q4Answer",
    "automate common": "Q7Answer",
    "prepopulate": "Q9Answer",
    "ingest": "Q5Answer",
    "geographic": "Q8Answer",
}


@click.command()
@click.option("-d", "--demographics-in", required=True, type=str)
@click.option("-r", "--ratings-in", required=True, type=str)
@click.option("-a", "--aspect-ranking-in", required=True, type=str)
@click.option("-u", "--users-out", required=True, type=str)
@click.option("-o", "--ratings-out", required=True, type=str)
@click.option("-k", "--tools-out", required=True, type=str)
@click.option("-s", "--aspect-ids-out", required=True, type=str)
@click.option("--demographics-blacklist", type=str)
@click.option("--aspect-ranking-blacklist", type=str)
@click.option("--ratings-blacklist", type=str)
def main(
    demographics_in,
    ratings_in,
    aspect_ranking_in,
    users_out,
    ratings_out,
    tools_out,
    aspect_ids_out,
    demographics_blacklist,
    aspect_ranking_blacklist,
    ratings_blacklist,
    **kwargs,
):
    parse(
        demographics_in,
        ratings_in,
        aspect_ranking_in,
        users_out,
        ratings_out,
        tools_out,
        aspect_ids_out,
        demographics_blacklist,
        aspect_ranking_blacklist,
        ratings_blacklist,
        **kwargs,
    )


def parse(
    demographics_in,
    ratings_in,
    aspect_ranking_in,
    users_out,
    ratings_out,
    tools_out,
    aspect_ids_out,
    demographics_blacklist=None,
    aspect_ranking_blacklist=None,
    ratings_blacklist=None,
    **kwargs,
):
    print("Reading CSVs...")
    demographics_df = pd.read_csv(demographics_in)
    aspect_rankings_df = pd.read_csv(aspect_ranking_in)
    ratings_df = pd.read_csv(ratings_in)
    demographics_df = preprocess_df(demographics_df, demographics_column_types)
    aspect_rankings_df = preprocess_df(aspect_rankings_df, aspect_column_types)
    ratings_df = preprocess_df(ratings_df, ratings_column_types)
    print(ratings_df.index)
    print(aspect_rankings_df.index)
    print(demographics_df.index)
    if demographics_blacklist is not None and demographics_blacklist != "":
        blacklist = preprocess_df(
            pd.read_csv(demographics_blacklist), demographics_column_types
        )
        demographics_df = apply_blacklist(demographics_df, blacklist).reset_index()
    if aspect_ranking_blacklist is not None and aspect_ranking_blacklist != "":
        blacklist = preprocess_df(
            pd.read_csv(aspect_ranking_blacklist), aspect_column_types
        )
        aspect_rankings_df = apply_blacklist(aspect_rankings_df, blacklist).reset_index()
    if ratings_blacklist is not None and ratings_blacklist != "":
        blacklist = preprocess_df(pd.read_csv(ratings_blacklist), ratings_column_types)
        # blacklist = blacklist.applymap(lambda x: np.NaN if x == "can't tell" else x)
        ratings_df = apply_blacklist(ratings_df, blacklist).reset_index()
    print(ratings_df.index)
    print(aspect_rankings_df.index)
    print(demographics_df.index)
    s = {}
    ranking_repeats = {}
    repeats = False
    ranking_repeated = False

    print("Checking for dirty data...")
    for row in ratings_df.iterrows():
        row = row[1]
        if (row["Tool"], row["SurveyUser"]) in s:
            s[(row["Tool"], row["SurveyUser"])].append(dict(row))
        else:
            s[row["Tool"], row["SurveyUser"]] = [dict(row)]
        if (row["SurveyUser"], row["ToolRank"]) in ranking_repeats:
            ranking_repeated = True
            ranking_repeats[(row["SurveyUser"], row["ToolRank"])].append(
                (row["SurveyUser"], row["Tool"], row["ToolRank"])
            )
        else:
            ranking_repeats[(row["SurveyUser"], row["ToolRank"])] = [
                (row["SurveyUser"], row["Tool"], row["ToolRank"])
            ]
    if repeats:
        for rows in s.values():
            if len(rows) > 1:
                for v in rows:
                    print(v)
                print()
        raise ValueError(
            "Repeated Tool User pairs detected. See printout above and add to blacklist accordingly."
        )
    if ranking_repeated:
        for rows in ranking_repeats.values():
            if len(rows) > 1:
                for v in rows:
                    print(v)
                print()
        raise ValueError(
            "Repeated Tool Ranks detected. See printout above and add to blacklist accordingly."
        )

    print("Formatting data...")
    aspect_ids = aspect_rankings_df[0:6].drop(
        columns=["CapabilityValue", "Survey", "SurveyUser", "CapabilityRank"]
    )
    aspect_ids["QXAnswer"] = ""
    for phrase, answer in aspect_to_answer.items():
        aspect_ids.at[
            aspect_ids.index[aspect_ids["CapabilityLabel"].str.contains(phrase)][0],
            "QXAnswer",
        ] = answer
    demographics_df = demographics_df.drop(columns=["Title", "Survey", "SurveyStatus"])
    demographics_df = demographics_df.rename(
        columns={
            "Q1Answer": "familiarity",
            "Q2Answer": "occupation",
            "Q3Answer": "experience",
            "SurveyUser": "name",
        }
    )
    demographics_df["aspect_ranking"] = [(-1, -1, -1, -1, -1, -1)] * len(
        demographics_df
    )
    for user in demographics_df["name"].unique():
        l = [0, 0, 0, 0, 0, 0]
        for index, row in aspect_rankings_df[aspect_rankings_df["SurveyUser"] == user].iterrows():
            id = aspect_ids[aspect_ids["CapabilityLabel"] == row["CapabilityLabel"]].index[0]
            l[row["CapabilityRank"] - 1] = int(id)
        demographics_df.at[demographics_df.index[demographics_df["name"] == user][0], "aspect_ranking"] = tuple(l)
    demographics_df.index.name = "user_id"
    tool_table = pd.DataFrame(ratings_df["Tool"].unique()).rename(
        columns={0: "tool_name"}
    )
    tool_table.index.name = "tool_id"
    ratings_df["Tool"] = ratings_df["Tool"].apply(
        lambda x: tool_table.index[tool_table["tool_name"] == x][0]
    )
    # demographics_df.index[demographics_df['name'] == "Brian Bell"][0]
    # ratings_df = ratings_df.loc[ratings_df['SurveyUser'].apply(lambda x: x in demographics_df['name'].unique())]
    ratings_df["SurveyUser"] = ratings_df["SurveyUser"].apply(
        lambda x: demographics_df.index[demographics_df["name"] == x][0]
    )
    ratings_df = ratings_df.drop(
        columns=["Survey", "Title", "SurveyStatus", "Q1Answer", "Q2Answer"]
    )
    ratings_df = ratings_df.rename(
        columns={
            "Tool": "tool_id",
            "SurveyUser": "user_id",
            "Q3Answer": "Overall Score",
        }
    )
    tool_ids = ratings_df["tool_id"].unique()
    user_ids = ratings_df["user_id"].unique()
    ratings_df = ratings_df.set_index(["user_id", "tool_id"])
    ratings_df = ratings_df.reindex(
        index=pd.MultiIndex.from_product(
            [list(set(user_ids.tolist())), list(set(tool_ids.tolist()))],
            names=["user_id", "tool_id"],
        )
    )
    demographics_df["tool_ranking"] = [(-1, -1)] * len(demographics_df)
    for user_id in user_ids:
        user_ratings_df = ratings_df.loc[user_id]
        # l = len(user_ratings_df['ToolRank'].unique()) - 1 if np.isnan(user_ratings_df['ToolRank'].unique()).any() else len(user_ratings_df['ToolRank'].unique())
        s = list(user_ratings_df["ToolRank"].unique())
        for i in range(len(s)):
            if np.isnan(s[i]):
                s.pop(i)
                break
        l = int(max(s))
        rankings = [-1] * l
        for tool_id in user_ratings_df.index:
            rank = user_ratings_df.loc[tool_id]["ToolRank"]
            if not np.isnan(rank):
                rankings[int(rank) - 1] = tool_id
        i = 0
        while i != len(rankings):
            if rankings[i] == -1:
                rankings.pop(i)
                i -= 1
            i += 1
        demographics_df.at[user_id, "tool_ranking"] = tuple(rankings)
    ratings_df = ratings_df.drop(columns=["ToolRank"])
    ratings_df = ratings_df.applymap(lambda x: np.nan if x == -1 or x == "" else x)
    rename_aspects = {}
    for q_num in range(4, 10):
        q = "Q" + str(q_num) + "Answer"
        aspect_id = aspect_ids.index[aspect_ids["QXAnswer"] == q][0]
        rename_aspects[q] = "aspect_id_" + str(aspect_id)
    rename_aspects["Q10Answer"] = "Free Response"
    ratings_df = ratings_df.rename(columns=rename_aspects)

    print("Dropping empty users...")
    aspect_columns = []
    dropped_users = []
    for c in ratings_df.columns:
        if "aspect" in c:
            aspect_columns.append(c)
    for user in ratings_df.index.unique(level=0):
        if all(
            ratings_df.loc[user][column].apply(np.isnan).all()
            for column in aspect_columns
        ):
            dropped_users.append(user)
    if len(dropped_users) > 0:
        print(f"Users {dropped_users} have no filled in responses - dropping...")
        ratings_df = ratings_df.drop(dropped_users, level=0)
        demographics_df = demographics_df.drop(dropped_users)
        # sort by index then reindex to get user_ids in sequential order starting from 0
    else:
        print("Not dropping any users...")

    # for user in range(13):
    #    print(ratings_df.iloc[user])
    #    print()

    ratings_df.reset_index(inplace=True)
    demographics_df.reset_index(inplace=True)
    for user in dropped_users:
        ratings_df["user_id"] = ratings_df["user_id"].apply(
            lambda x: x - 1 if x > user else x
        )
        demographics_df["user_id"] = demographics_df["user_id"].apply(
            lambda x: x - 1 if x > user else x
        )

    print("Writing CSVs...")
    ratings_df.to_csv(ratings_out)
    demographics_df.to_csv(users_out)
    aspect_ids.to_csv(aspect_ids_out)
    tool_table.to_csv(tools_out)
    print("Stage 1 Done.")
    return tool_table, aspect_ids, demographics_df, ratings_df


if __name__ == "__main__":
    # ratings_df = pd.read_csv(sys.argv[1])
    # ratings_blacklist_df = pd.read_csv(sys.argv[2])
    # ratings_df = preprocess_df(ratings_df, ratings_column_types)
    # ratings_blacklist_df = preprocess_df(ratings_blacklist_df, ratings_column_types)
    # apply_blacklist(ratings_df, ratings_blacklist_df).to_csv("test.csv")
    main()
