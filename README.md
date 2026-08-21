# Music Streaming Next Best Action — Offline RL

An end-to-end offline reinforcement learning system for user 
re-engagement, inspired by real NBA systems at platforms like 
Spotify and Netflix. Trains a Conservative Q-Learning (CQL) 
agent on historical interaction logs without any live environment 
interaction, and evaluates it using Doubly Robust off-policy estimation.

## The Problem

Subscription platforms need to decide which channel to use to 
re-engage lapsed users - email digest, push notification, in-app 
banner, or personalised mix - without running expensive live 
experiments for every new policy candidate.

Offline RL solves this: train on logged historical decisions, 
evaluate without deploying, only ship policies that clear a 
measurable bar.

## What This Project Builds

- A music streaming simulator with 5 user engagement states 
  and 5 action channels
- Three reward functions demonstrating the reward hacking problem
- CQL offline agent — trains on frozen logs, never touches the environment
- Doubly Robust OPE — used as a deployment gate before any policy ships

## Key Results

- CTR reward causes the agent to spam push notifications to 
  lapsed users (reward hacking in action)
- CQL converges to conservative Q estimates (~7.0) while vanilla 
  DQN inflates to ~9.5, overestimating value of unseen actions
- Both CQL and DQN beat the behavior policy baseline (0.037) 
  under DR evaluation — both clear the deployment gate
- CQL learned to reduce push notifications to lapsed/churning 
  users and increase new_mix recommendations — aligning with 
  the LTV reward structure

## Notebooks

| notebook | what it does |
|----------|-------------|
| 01_music_simulator | simulator, behavior policy, reward design, reward hacking demo |
| 02_cql_training | offline DQN vs CQL, Q value comparison, learned policy |
| 03_ope_evaluation | DR estimator, four-policy comparison, deployment gate |

## Concepts Demonstrated

- Offline RL — training without environment interaction
- Conservative Q-Learning (CQL) — pessimistic Bellman targets 
  for out-of-distribution actions
- Doubly Robust OPE — unbiased policy evaluation from logs
- Reward hacking — why CTR metrics produce misaligned policies
- Heterogeneous reward design — LTV proxy for multi-channel systems
- Propensity scores — correcting for behavior policy bias in OPE

## Running

Open notebooks in Google Colab in order: 01 -> 02 -> 03.
Notebook 01 generates logs saved to Google Drive, loaded by 02 and 03.

## References

- Kumar et al. 2020 : Conservative Q-Learning for Offline RL
- Levine et al. 2020 : Offline RL: Tutorial, Review, and Perspectives
- Fujimoto et al. 2019 : Off-Policy Deep RL without Exploration (BCQ)
