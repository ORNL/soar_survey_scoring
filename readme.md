# Readme for AI ATAC 3 SOAR Challenge scoring repository
using git lfs for any data or large files

## general structure:
- all code is in /code
- raw data is only thing in /data
- outputs of code goes in /results and /figs


## code
list of files/folders with how to run, what/where they output, and description:
- ak_inference.py - runs Adomivicius & Kwon method of creating similarities of users/items from data then predicting unknown ratings. This has a single class for running the base method on our data. See doc block in this file for details.
  - this calls sentiment.py to do the sentiment analysis
  - this was meant to be called by phase1_pipeline.py but can be run on it's own.
  - it reads in the raw user and ratings table csvs
  - upon training it writes results to results folder
  - upon inference (self.fill_r()) it writes du_r_filled.csv to results folder.

- assign-videos.py
  - how to run: Run from command line `python3 code/assign-videos.py`
  - output: This script writes a .json file to data/video-assignments.json, and to data/video-assignments-metadata.json.
  - description: see file header.
- stats-sims.py
  - how to run: run from command line `python3 code/stats-sim.py`
  - output: this script builds folders and puts results in `/results/stats-sim-results/`
  - description:
  This script is designed to examine if we could get statistically significant results based on number of users
  for phase 1.
  It uses two distributions to generate ratings data in {1, 2, .., 5}
  The goal here is to make realistic scores that should clearly allow us to see the second tool is better.
  Using distributions above we generate user ratings for tool 1 and tool 2 for m = 5, 10, .., 45, users.
  NOTE: phase 1 will have each user answer 7 questions per tool video, we're only generating 1 score per user per tool here. Assuming we boil their 7 scores into a single score, that is what we are simulating.

  First we examine the number of users (m) needed to tell with confidence that tool 2 is better than tool 1
      - tool 1 ratings have mean, variance: (2.9285714285714284, 0.9234693877551035)
      - tool 2 ratings have mean, variance: (3.647058823529412, 1.1695501730103768)
      - Results are stored in /results/stats-sim-results/p1-p2-simulations
          - ttest.json: results give the percent of hypothesis tests that confirm tool 1 not equal to tool 1  with a two-sided Welch's t-test for different numbers of users (m = 5, 10, ..40, 100)
          - binary-{z, chi2, binomial}-{threshold}.json: After converting the 1-5 scores into binary using two thresholds (3.5 meaning:1,2,3 --> 0, 4,5 --> 1 and 2.5 meaning:1,2 --> 0, 3,4,5 --> 1 ), using
              - a z-score on the difference of means (observed p1 - observed p2 / shat)
              - a chi-sq. test on difference in proportions p1, p2
              - a custom binomial test (null hypothesis is that tool 2 data is from same binomial as tool 1's--> pvalue 1, null hypoth that tool 1's data is sampled from tool 2's --> pvalu 2, average them. )
      - higher percentages are better for all the results in this folder. e.g., the result 25:.67 implies that with 25 users that test provided at least 95% confidence that tool 2 was not equal to tool 1.

  Second we examine the number of users needed to tell with confidence that tool2 and tool 1 are the same.
      - tool 2 and tool 1 ratings have the same mean, variance (3.647058823529412, 1.1695501730103768)
      - results are stored in /results/stats-sim-results/p2-p2-simulations
      - same tests/thresholds as above used, mapping to same file names
      - lower percentages are better in this folder. e.g. the result 25:.08 means that with 25 users and that that test/scenario, we were 95% confident tool 1 and tool 2 were different (even though they were the same).
- phase1-pipeline.py - glueing code to produce results form phase1.
  - reads in raw csvs.
  - Cleaning code
    - As we collected results, we sifted through the csvs manually to check for any bad data rows. Bad data rows were added to a blacklist csv. The blacklist csvs are then read in during the data cleaning step to programatically filter out the blacklisted rows from the raw csvs.
    - In addition to manual checking, during the data cleaning step, there are a few sanity checks to further ensure any bad rows don't make it through to the final results. Rows which fail the sanity checks are printed out in an error report, and the program halts. For example, we check if any users have the same ranking for more than one tool, among other checks.
    - Finally, the clean tables are massaged into a format that's more convenient for further analysis.
  - runs ak_inference code to fill in missing aspect ratings
  - <SAVANNAH's CODE>
  - Stage4.py - reads in raw scores to generate means of basic scores, uses page rank algorithm based on initial user rankings to rank tools, ingresses predicted overall scores to visualize completed data frame, ranks tools based on completed df from b Stage 2 and Stage 3
  - polite curtsy and ride off into sunset
  - stats_visualization.py - takes in both filled overall ratings and the raw ratings. tests 4 demographic effects for correlation to rating: years of experience, occupation, familiarity, and perceived video quality. tests on both filled and raw ratings to ensure that the way we are filling missing values does not change conclusions.  
- sentiment.py - has functions for converting text strings to floats using 4 different sentiment analyzers. VADER and RoBERTa are the good ones. this is called by ak_inference.py.
- test_multipriocess.py - just a script for testing if multiproccessing code works. never used in pipeline.
- utils.py - serialization functions that are called by other scripts (jsonify, unjsonify, etc.)
  - polite curtsy and ride off into sunset
- sentiment.py - has functions for converting text strings to floats using 4 different sentiment analyzers. VADER and RoBERTa are the good ones. this is called by ak_inference.py.
- test_multipriocess.py - just a script for testing if multiproccessing code works. never used in pipeline.
- utils.py - serialization functions that are called by other scripts (jsonify, unjsonify, etc.)
- where-did-tools-stink.py - prints out the questions for which the tool's raw preference ratings fell in the lowest 25%. see ./phase1/data/Questions.csv for what questions were given. 

## results/real
Results generated
  - Stage 4:
   - Tool{i}results.png: Figures for scores of each tool. Highlights the mean and STD for each score of each tool. Also includes information about which data points are projected from our algorithm and which data points were given by the users. pngs for each tool with named and anonymous versions
   - sentimentresults.png: Figure to visualize sentiment scores
   - RawUserRankging.png: Directed graph of raw ranking
   		- rawrankingPR.csv, associated numerics from page rank
   - mlranking.png: Directed graph of completed rankgings developed from ML predictions of user preferences
   		- predictedrankingPR.csv, associated numerics from page rank
  - Stage 5:
    - exp_regr.png: years of experience regression analysis
    - fam_vq2_barplot.png: barplot with std error bars showing effect of familiarity and video quality on overall tool rating
    - occupation_barplot.png: barplot with std error bars showing effect of occupation on tool rating
