# GAIKube: Generative AI-based Proactive Kubernetes Container Orchestration Framework for Heterogeneous Edge Computing

Containerized edge computing emerged as a preferred platform for latency-sensitive applications requiring informed and efficient decision-making accounting for the end
user and edge service providers’ interests simultaneously. Edge decision engines exploit pipelined knowledge streams to enhance performance and often fall short by employing inferior resource
predictors subjected to limited available training data. These shortcomings flow through the pipelines and adversely impact other modules, including schedulers leading to such decisions
costing delays, user-experienced accuracy, Service Level Agreements (SLA) violations, and server faults. To address limited data, substandard CPU usage predictions, and container orchestration
considering delay accuracy and SLA violations, we propose a threefold GAIKube framework offering Generative AI (GAI)-enabled proactive container orchestration for a heterogeneous edge computing paradigm.
Addressing data limitation, GAIKube employs DoppelGANger (DGAN) to augment time series CPU usage data for a computationally heterogeneous edge cluster. In the second place, GAIKube leverages Google TimesFM
for its long horizon predictions, 4.84 Root Mean Squared Error (RMSE) and 3.10 Mean Absolute Error (MAE) against veterans Long Short-Term Memory (LSTM) and Bidirectional LSTM (Bi-LSTM) on
concatenated DGAN and original dataset. Considering TimesFM quality predictions utilizing the DGAN extended dataset, GAIKube pipelines CPU usage predictions of edge
servers to a proposed dynamic container orchestrator. GAIKube orchestrator produces container scheduling, migration, dynamic vertical scaling, and hosted application model-switching to balance
contrasting SLA violations, cost, and accuracy objectives avoiding server faults. Google Kubernetes Engine (GKE) based real testbed experiments show that the GAIKube orchestrator
offers 3.43% SLA violations and 3.80% user-experienced accuracy loss with zero server faults at 1.46 CPU cores expense in comparison to industry-standard model-switching, GKE pod scaling, and
GKE optimized scheduler.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0c3b5a9c-cec1-4a91-bb1e-45b93ee3848f" alt="GAIKube">
</p>


# License
BSD-3-Clause. Copyright (c) 2024, Babar Ali. All rights reserved.
See the [License](https://github.com/BabarAli93/GAIKube/blob/main/LICENSE) file for more details.
