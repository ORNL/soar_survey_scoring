#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:48:03 2021

@author: 2he
"""

"""
Created on Mon May  3 21:33:05 2021

@author: 2he
"""
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from pathlib import Path
from .parse_phase1 import preprocess_df, ratings_column_types

# date = '5_19_21'

######################
######Stage 1 Analysis
######################
def sams_stuff(data, date, results_dir, best_model_csv):
    user_res = Path(data, date, 'UserResponses.csv')
    df_import = pd.read_csv(user_res)
    #toolnames = pd.read_csv(r"../results/real/"+date+"/Stage1/Tools.csv")
    toolnames = pd.read_csv(Path(results_dir, date, "Stage1/Tools.csv"))
    toolname_dict = toolnames.tool_name.to_dict()
    # df_import = pd.read_csv(r'../data/real/'+date +'/UserResponses.csv')
    raw_df = df_import[["Tool", "Q3Answer"]]
    raw_df = preprocess_df(raw_df, ratings_column_types)
    print("--Raw Overall Score Metrics--")
    print(raw_df)
    ##Overall Score
    #plt.savefig(r"../results/real/"+date+"/Stage4/Overallresults_anon.png", bbox_inches = 'tight')
    #plt.savefig(Path(results_dir, date, "Stage4/Overallresults_anon.png"), bbox_inches='tight')
    plt.figure()
    sns.boxplot(x="Tool", y = "Q3Answer", data = raw_df, linewidth=0.5, whis=np.inf, color = ".7", meanline=True, showmeans = True, meanprops={"color":"black"})
    sns.swarmplot(x="Tool", y = "Q3Answer", data = raw_df)
    plt.ylabel("Overall Score")
    plt.ylim(0.5, 5.5)
    plt.xticks(rotation=30) 
    plt.xlabel("Tool ID", fontsize=8)
    #plt.savefig(r"../results/real/"+date+"/Stage4/Overallresults.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/OverallRawResults.png"), bbox_inches='tight')
    raw_df.groupby('Tool')['Q3Answer'].mean().to_csv(Path(results_dir, date, "Stage4/Rawmean.csv")) 


    #####Run Page rank on given rankings
    ###This chunk loads raw user data
    rankdf = df_import[["Tool", "ToolRank", "SurveyUser"]]
    len(rankdf.SurveyUser.unique())
    Users = rankdf.SurveyUser.unique()

    ###Turns rankings into tuples
    pairs = pd.DataFrame({'from':[0], 'to': [0], 'User': [0]})


    ###Loops through user's to create pairs
    for i in range(len(Users)):
        indrank = rankdf[rankdf['SurveyUser']  == Users[i]]
        indrank = indrank.sort_values(by = ['ToolRank']).reset_index(inplace=False)
        del indrank["index"]
        edges = len(indrank)-1
        for x in range(edges): 
            tools = len(indrank)-1-x
            for j in range(tools):
                p = pd.DataFrame({'from': [indrank.Tool[len(indrank)-1-x]], 'to': [indrank.Tool[j]], 'User': Users[i]})
                pairs = pairs.append([p])

    #pairs = pairs.iloc[1:]
    pairs['weight'] = 1*0.75
    del pairs['User']
    pairs = pairs.iloc[1:]

    #Weight pairs and sum up negative weights
    weightedpairs = pairs.groupby(['from', 'to'])['weight'].sum().reset_index()
    negweightedpairs = weightedpairs
    negweightedpairs = negweightedpairs.rename(columns = {'to':'from', 'from':'to'}).reset_index()
    negweightedpairs.weight = negweightedpairs.weight*(-1)
    weightedpairs = weightedpairs.append([negweightedpairs])
    df_weight = weightedpairs.groupby(['from', 'to'])['weight'].sum().reset_index()

    #Final df for graphing
    df_s1 = df_weight[df_weight.weight >0].reset_index()

    #Build adjacency matrix
    G=nx.from_pandas_edgelist(df_s1, 'from', 'to', edge_attr='weight', create_using=nx.DiGraph())

    #nx.draw_shell(G, with_labels=True, node_size=1500, node_color='skyblue', alpha=0.3, arrows=True,
    #              weight=nx.get_edge_attributes(G,'weight').values(), arrowstyle='fancy')
    nx.draw_shell(G, with_labels=True, node_size=1500, node_color='skyblue', alpha=0.3, arrows=True, arrowstyle='fancy')
    widths = nx.get_edge_attributes(G, 'weight')
    nodelist = G.nodes()

    ####Ways to make graph look better, I still don't love the look
    plt.figure(figsize=(11,11))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G,pos,
                           nodelist=nodelist,
                           node_size=500,
                           node_color='blue',
                           alpha=0.65)
    nx.draw_networkx_edges(G,pos,
                           edgelist = widths.keys(),
                           width=list(widths.values()),
                           edge_color='grey',
                           #arrowstyle='fancy',
                           alpha=0.75)
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0]+0.01*(1)*np.sign(coords[0]), coords[1]+0.1*(1)*np.sign(coords[1]))

    nx.draw_networkx_labels(G, pos=pos_attrs,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='black',
                            font_size = 16)
    #plt.show(block=False)

    #Calculate page rank
    print("--Saving network and initial page rank analysis--")
    # path = Path(results_dir, date, "Stage4/RawUserRanking.png")
    # print(f"first path is {path}")
    plt.savefig(Path(results_dir, date, "Stage4/RawUserRanking.png"), format="PNG")
    #plt.savefig(r"../results/real/"+date+"/Stage4/RawUserRanking.png", format="PNG")
    #Calculate and save page rank
    #pd.DataFrame.from_dict(nx.pagerank(G, alpha = 0.85, weight = 'weight'), orient="index").to_csv(r"../results/real/"+date+"/Stage4/rawrankingPR.csv")
    csv_path = Path(results_dir, date, "Stage4/rawrankingPR.csv")
    # print(f"csv path is {csv_path}")
    pd.DataFrame.from_dict(nx.pagerank(G, alpha=0.85, weight='weight'), orient="index").to_csv(Path(results_dir, date, "Stage4/rawrankingPR.csv"))

    ######################
    ######Stage 2 Analysis
    ######################
    #df_s2 = pd.read_csv(r"../results/real/"+date+"/Stage2/df_r_filled.csv")
    df_s2 = pd.read_csv(Path(results_dir, date, "Stage2/df_r_filled.csv"))
    df_s2 = df_s2.merge(toolnames, on="tool_id")
    del df_s2["Unnamed: 0"]
    df_s2.columns = ['user_id', 'tool_id', 'OS', '0', '1', '2', '3', '4', '5', 'Sentiment', "tool_name"]
    
    score = pd.read_csv(best_model_csv)
#   score = score.merge(toolnames, on="tool_id")
    
    df_s2 = df_s2.merge(score, on = ["user_id", 'tool_id'])
   # print(df_s2)
    
    del df_s2["OS"]
    del df_s2["Unnamed: 0"]
    del df_s2["b_overall_score"]
    # print(df_s2)
    df_s2.columns = ['user_id', 'tool_id', '0', '1', '2', '3', '4', '5', 'Sentiment', "tool_name", "OS"]

    average = df_s2.mean(axis=0)
    
    Sentiment = df_s2[["user_id", "tool_id", "Sentiment", "tool_name"]]
    Overall = df_s2[["user_id", "tool_id", "OS", "tool_name"]]
    
    #del df["Overall Score"]
    del df_s2["Sentiment"]
    del df_s2["tool_name"]
    
    df1 = pd.melt(df_s2, id_vars = ['user_id', 'tool_id'], var_name = 'Rating')
    ##Create per tool scatter plots
    df1['Type1'] = df1.value-df1.value.astype(int) == 0
    df1= df1.assign(Type=df1.Type1.map({True: "User Def", False: "Predicted"}))
    
    
    ##Compare Given, Predicted data, overall informations
    print("--Plotting User Results--")
    for i in range(len(df1.tool_id.unique())):
        uniquetool = df1[df1['tool_id'] == i]
        uniquetool.sort_values(by=['Type'], ascending=False)
        alltools = df1
        plt.figure(i, figsize = (5,3))
        plt.subplot(1, 2, 1)
        #sns.boxplot(x="Rating", y="value", data=alltools, whis=np.inf, color = ".7")
        #sns.violinplot(x="Rating", y="value", data=uniquetool, inner=None, color=".8")
        sns.violinplot(x = "Rating", y = "value", data = uniquetool, hue= "Type", hue_order=["User Def", "Predicted"], palette="Set2", split=True, scale="count", inner="quartile")
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Rating", fontsize=12)
        plt.ylim(0.5, 5.5)
        plt.title("User Defined Results vs Our Predicted Data", fontsize=14)
        plt.legend(title = "Type", loc='lower right')
        a=str(i)
        plt.subplot(1, 2, 2)
        sns.boxplot(x="Rating", y="value", data=alltools, whis=np.inf, color = ".7")
        #sns.violinplot(x="Rating", y="value", data=uniquetool, inner=None, color=".8")
        sns.swarmplot(x = "Rating", y = "value", data = uniquetool, hue= "Type", hue_order=["User Def", "Predicted"], palette="Set2")
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Rating", fontsize=12)
        plt.ylim(0.5, 5.5)
        plt.title("Individual Tool (Dots) vs All Tools (Box)", fontsize=14)
        plt.legend().remove()
        plt.subplots_adjust(top=0.87, bottom=0.2, left=0, right=2, hspace=0.2,
                        wspace=0.2)
        plt.suptitle(f"Tool {i}",x=1, fontsize = 20)
        #plt.savefig(r"../results/real/"+date+"/Stage4/Tool"+a+"results_anon.png", bbox_inches = 'tight')
        plt.savefig(Path(results_dir, date, f"Stage4/Tool{a}results_anon.png"), bbox_inches='tight')
        b = toolnames[toolnames.tool_id == i].tool_name.values[0]
        plt.suptitle(f"Tool {b}",x=1, y=1.05, fontsize = 20)
        #plt.savefig(r"../results/real/"+date+"/Stage4/Tool"+a+"results.png", bbox_inches = 'tight')
        plt.savefig(Path(results_dir, date, f"Stage4/Tool{a}results.png"), bbox_inches='tight')
    
    ##Sentiment scores    
    plt.figure()
    sns.boxplot(x="tool_id", y = "Sentiment", data = Sentiment, linewidth=0.5, whis=np.inf, color = ".7", meanline=True, showmeans = True, meanprops={"color":"black"})
    sns.swarmplot(x="tool_id", y = "Sentiment", data = Sentiment)
    plt.ylim(0.5, 5.5)
    plt.xlabel("Tool ID", fontsize=8)
    #plt.savefig(r"../results/real/"+date+"/Stage4/sentimentresults_anon.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/sentimentresults_anon.png"), bbox_inches='tight')
    plt.figure()
    sns.boxplot(x="tool_name", y = "Sentiment", data = Sentiment, linewidth=0.5, whis=np.inf, color = ".7", meanline=True, showmeans = True, meanprops={"color":"black"})
    sns.swarmplot(x="tool_name", y = "Sentiment", data = Sentiment)
    plt.ylim(0.5, 5.5)
    plt.xticks(rotation=30) 
    plt.xlabel("Tool Name", fontsize=8)
    #plt.savefig(r"../results/real/"+date+"/Stage4/sentimentresults.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/sentimentresults.png"), bbox_inches='tight')


    ##Overall Score
    plt.figure()
    sns.boxplot(x="tool_id", y = "OS", data = Overall, linewidth=0.5, whis=np.inf, color = ".7", meanline=True, showmeans = True, meanprops={"color":"black"})
    sns.swarmplot(x="tool_id", y = "OS", data = Overall)
    plt.ylabel("Overall Score")
    plt.ylim(0.5, 5.5)
    plt.xlabel("Tool ID", fontsize=8)
    #plt.savefig(r"../results/real/"+date+"/Stage4/Overallresults_anon.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/Overallresults_anon.png"), bbox_inches='tight')
    plt.figure()
    sns.boxplot(x="tool_name", y = "OS", data = Overall, linewidth=0.5, whis=np.inf, color = ".7", meanline=True, showmeans = True, meanprops={"color":"black"})
    sns.swarmplot(x="tool_name", y = "OS", data = Overall)
    plt.ylabel("Overall Score")
    plt.ylim(0.5, 5.5)
    plt.xticks(rotation=30) 
    plt.xlabel("Tool ID", fontsize=8)
    #plt.savefig(r"../results/real/"+date+"/Stage4/Overallresults.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/Overallresults.png"), bbox_inches='tight')

    ######################
    ######Stage 3 Analysis
    ######################

    #df_s3 = pd.read_csv(r'../results/real/'+date+'/Stage3/model_results.csv')
    #df_s3 = pd.read_csv(Path(results_dir, date, '/Stage3/model_results.csv'))

    #model = df_s3[df_s3.cross_val_mse == min(df_s3.cross_val_mse)].model
    #model = model.iloc[0]

    ##Read best ML scores
    # score = pd.read_csv(r'../results/real/'+date+'/Stage3/' + model+ '.csv')
    score = pd.read_csv(best_model_csv)
    score = score.merge(toolnames, on="tool_id")
    
    ##Plot box plots
    plt.figure()
    sns.boxplot(x="tool_id", y = "s_overall_score", data = score, linewidth=0.5, whis=np.inf, color = ".7", meanline=True, showmeans = True, meanprops={"color":"black"})
    sns.swarmplot(x="tool_id", y = "s_overall_score", data = score)
    plt.ylabel("Overall Score")
    plt.ylim(0.5, 5.5)
    plt.xlabel("Tool ID", fontsize=8)
    #plt.savefig(r"../results/real/"+date+"/Stage4/MLresults_anon.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/MLresults_anon.png"), bbox_inches='tight')
    plt.figure()
    sns.boxplot(x="tool_name", y = "s_overall_score", data = score, linewidth=0.5, whis=np.inf, color = ".7", meanline=True, showmeans = True, meanprops={"color":"black"})
    sns.swarmplot(x="tool_name", y = "s_overall_score", data = score)
    plt.ylabel("Overall Score")
    plt.ylim(0.5, 5.5)
    plt.xticks(rotation=30) 
    plt.xlabel("Tool ID", fontsize=8)
    #plt.savefig(r"../results/real/"+date+"/Stage4/MLresults.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/MLresults.png"), bbox_inches='tight')
    #score.groupby('tool_name')['s_overall_score'].mean().to_csv(r"../results/real/"+date+"/Stage4/meanML.csv")
    score.groupby('tool_name')['s_overall_score'].mean().to_csv(Path(results_dir, date, "Stage4/meanML.csv"))


    ######################
    ######Final Analysis
    ######################
    rankdfpredicted = score[["user_id", "tool_id", "s_overall_score"]]
    rankdfpredicted = rankdfpredicted.merge(toolnames, on = "tool_id")
    #usernames = pd.read_csv(r"../results/real/"+date+"/Stage1/Users.csv")
    usernames = pd.read_csv(Path(results_dir, date, "Stage1/Users.csv"))
    usernames = usernames[["user_id", "name"]]
    rankdfpredicted = rankdfpredicted.merge(usernames, on = "user_id")
    rankdfpredicted.rename(columns = {"name": "SurveyUser", "tool_name":"Tool"}, inplace = True)
     
    finalpreddf = pd.merge(rankdfpredicted, rankdf, on = ["Tool", "SurveyUser"], how = 'outer')   
    finalpreddf.dropna(subset = ["s_overall_score"], inplace=True)
    
    len(finalpreddf.SurveyUser.unique())
    Users = finalpreddf.SurveyUser.unique()
    
    ###Turns rankings into tuples
    pairs_pred = pd.DataFrame({'from':[0], 'to': [0], 'User': [0]})
    
    ###Loops through user's to create pairs
    for i in range(len(Users)):
        indrank_pred = finalpreddf[finalpreddf['SurveyUser']  == Users[i]]
        indrank_pred = indrank_pred.sort_values(by = ['ToolRank'])
        indrank_pred = indrank_pred.sort_values(by = ['s_overall_score'], ascending=False).reset_index(inplace=False)
        del indrank_pred["index"]
        edges1 = len(indrank_pred)-1
        for x in range(edges1):
            tools = len(indrank_pred)-1-x
            for j in range(tools):
                p = pd.DataFrame({'from': [indrank_pred.Tool[len(indrank_pred)-1-x]], 'to': [indrank_pred.Tool[j]], 'User': Users[i]})
                #print(p)
                pairs_pred = pairs_pred.append([p])
    
    pairs_pred = pairs_pred.iloc[1:]
    pairs_pred['weight'] = 1*0.75
    del pairs_pred['User']
    
    #Weight pairs and sum up negative weights
    weightedpairs_pred = pairs_pred.groupby(['from', 'to'])['weight'].sum().reset_index()
    negweightedpairs_pred = weightedpairs_pred
    negweightedpairs_pred = negweightedpairs_pred.rename(columns = {'to':'from', 'from':'to'}).reset_index()
    negweightedpairs_pred.weight = negweightedpairs_pred.weight*(-1)
    weightedpairs_pred = weightedpairs_pred.append([negweightedpairs_pred])
    df_weight_pred = weightedpairs_pred.groupby(['from', 'to'])['weight'].sum().reset_index()
    
    #Final df for graphing
    df_pred = df_weight_pred[df_weight_pred.weight >0].reset_index()
    
    #Build adjacency matrix
    Gpred=nx.from_pandas_edgelist(df_pred, 'from', 'to', edge_attr='weight', create_using=nx.DiGraph())
    
    #nx.draw_shell(G, with_labels=True, node_size=1500, node_color='skyblue', alpha=0.3, arrows=True,
    #              weight=nx.get_edge_attributes(G,'weight').values(), arrowstyle='fancy')
    nx.draw_shell(Gpred, with_labels=True, node_size=1500, node_color='skyblue', alpha=0.3, arrows=True, arrowstyle='fancy')
    widths = nx.get_edge_attributes(Gpred, 'weight')
    nodelist = Gpred.nodes()
    
    plt.figure(figsize=(11,11))
    pos = nx.circular_layout(Gpred)
    nx.draw_networkx_nodes(Gpred,pos,
                           nodelist=nodelist,
                           node_size=500,
                           node_color='blue',
                           alpha=0.65)
    nx.draw_networkx_edges(Gpred,pos,
                           edgelist = widths.keys(),
                           width=list(widths.values()),
                           edge_color='grey',
                           #arrowstyle='fancy',
                           alpha=0.75)
    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0]+0.01*(1)*np.sign(coords[0]), coords[1]+0.1*(1)*np.sign(coords[1]))
    
    nx.draw_networkx_labels(Gpred, pos=pos_attrs,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='black',
                            font_size = 16)
    #plt.show()
    
    #Calculate page rank
    nx.pagerank(Gpred, alpha = 0.85)

    #Calculate page rank
    print("--Saving network and page rank analysis based on ML Scores--")
    #plt.savefig(r"../results/real/"+date+"/Stage4/mlranking.png", bbox_inches = 'tight')
    plt.savefig(Path(results_dir, date, "Stage4/mlranking.png", bbox_inches='tight'))
    #Calculate and save page rank
    #pd.DataFrame.from_dict(nx.pagerank(Gpred, alpha = 0.85, weight = 'weight'), orient="index").to_csv(r"../results/real/"+date+"/Stage4/predictedrankingPR.csv")
    pd.DataFrame.from_dict(nx.pagerank(Gpred, alpha=0.85, weight='weight'), orient="index").to_csv(Path(results_dir, date, "Stage4/predictedrankingPR.csv"))


#if __name__ == "__main__":
#    sams_stuff('../data/real', date='6_4_21', results_dir='../results/real', best_model_csv=f'../results/real/6_4_21/Stage3/sgd.csv')
