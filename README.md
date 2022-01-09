# Pràctica CasKaggle APC UAB 2021-2022
### Author: Oriol Benítez Bravo
### NIU: 1566931
### Dataset: [csgo_round_snapshots.csv](https://www.kaggle.com/christianlillelund/csgo-round-winner-classification)
## Context of the data
Counter-Strike: Global Offensive (CS:GO) is a multiplayer first-person shooter developed by Valve and Hidden Path Entertainment.The game pits two teams, Terrorists and Counter-Terrorists, against each other in different objective-based game modes. The most common game modes involve the Terrorists planting a bomb while Counter-Terrorists attempt to stop them, or Counter-Terrorists attempting to rescue hostages that the Terrorists have captured.

The dataset consists of round snapshots from about 700 demos from high level tournament play in 2019 and 2020. Warmup rounds and restarts have been filtered, and for the remaining live rounds a round snapshot has been recorded every 20 seconds until the round is decided. Following its initial publication, It has been pre-processed and flattened to improve readability and make it easier for algorithms to process. The total number of snapshots is 122411. Snapshots are i.i.d and should be treated as individual data points, not as part of a match.

The information that has been substracted from this 700 demos every 20 seconds are 97 attributes; 3 are categorical strings, 94 are numerical data.

### Objectives:

The main objective is to train some supervised models in order to predict the team(Terrorist or Counter-Terrorist) who is going to win the round.

### Exploring the Data
