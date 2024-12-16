# EnergyStorageRLBidder
Final Project for Reinforcement Learning (ORCS 4529) at Columbia University

<b>Authors:</b> Davit Shadunts, Eliza Cohn, Neal Ma

<b>Term:</b> Fall 2024

To setup this repository with access to all data, you will need to pull the `git submodules`. To do this, run this command from the root of the directory:

```git submodule update --init --recursive```

## Pre-trained Agents

Pre-trained agents can be accessed from this [Google Drive link](https://drive.google.com/drive/u/0/folders/1C6h_ByAjbMMwieixn4jeFDoWmoWY3vx3). The folder contains the following six `.pkl` files, representing different agent models and strategies:

| **File Name**                     | **Model**       | **Strategy**       |
|-----------------------------------|-----------------|--------------------|
| `meanshift_norm_conservative.pkl` | Mean-Shift      | Conservative       |
| `meanshift_norm_honest.pkl`       | Mean-Shift      | Honest             |
| `meanshift_norm_risky.pkl`        | Mean-Shift      | Risky              |
| `timevarying_norm_conservative.pkl` | Time-Varying  | Conservative       |
| `timevarying_norm_honest.pkl`     | Time-Varying    | Honest             |
| `timevarying_norm_risky.pkl`      | Time-Varying    | Risky              |

Download the `pkl` files into the `/agents` directory to use them in a simulation. 