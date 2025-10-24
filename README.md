
[![CI](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/ci.yml)
[![Docs Deploy](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/jekyll-gh-pages.yml/badge.svg?branch=main)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/jekyll-gh-pages.yml)
[![Auto README](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/Readme.yml/badge.svg?branch=main)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/Readme.yml)
[![Sitemap](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/generate-sitemap.yml/badge.svg?branch=main)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/generate-sitemap.yml)
[![SEO Automation](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/seo-automation.yml/badge.svg?branch=main)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/seo-automation.yml)
[![Humans](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/humans.yml/badge.svg?branch=main)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/actions/workflows/humans.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](#)
[![Docs: MkDocs Material](https://img.shields.io/badge/Docs-MkDocs%20Material-blue?logo=mkdocs)](https://rajamuhammadawais.github.io/ai_driven_predictive_maintenance/)
[![Stars](https://img.shields.io/github/stars/RajaMuhammadAwais/ai_driven_predictive_maintenance?style=social)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/stargazers)
[![Issues](https://img.shields.io/github/issues/RajaMuhammadAwais/ai_driven_predictive_maintenance)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/issues)
[![Last Commit](https://img.shields.io/github/last-commit/RajaMuhammadAwais/ai_driven_predictive_maintenance)](https://github.com/RajaMuhammadAwais/ai_driven_predictive_maintenance/commits/main)

**AI-Driven Predictive Maintenance and Dynamic Resource Optimisation for Distributed Systems: An Integrated Causal-Inference, Deep-Learning and Reinforcement-Learning Framework**  

**Researcher:** [RAJA MUHAMMAD AWAIS]  
**Institution:** [University / Department]  
**Degree Programme:** [ MPhil Research]  
**Date:** 24 SEP 2025  

---

# **Abstract**  
Modern distributed systems—cloud, edge and IoT—exhibit non-linear, cascading failure modes that traditional reactive or interval-based maintenance cannot pre-empt. This proposal outlines a three-year mixed-methods study that develops, validates and deploys an AI-driven predictive maintenance (PdM) architecture able to (i) forecast impending failures up to 6 h ahead, (ii) isolate root causes with &gt; 85 % precision, and (iii) autonomously re-allocate CPU, memory and network resources to avert downtime.  

The project integrates four methodological streams:  
1. Multivariate time-series forecasting with LSTM/GRU ensembles for early-warning;  
2. Unsupervised anomaly detection via Isolation-Forest and lightweight Autoencoders for real-time flagging;  
3. Causal-inference-enhanced Bayesian Networks and decision-trees for explainable root-cause diagnosis;  
4. Deep reinforcement-learning (actor-critic) for dynamic, cost-aware resource scaling under QoS constraints.  

Using a hybrid simulation-testbed design, models will be trained on 5 TB of telemetry from a 400-node Kubernetes/OpenStack cluster, then prospectively evaluated in a living lab with 95 % power to detect a 30 % reduction in unplanned outages (α = .05).  

Expected contributions include: (a) a rigorously benchmarked, open-source PdM stack; (b) novel causal-RNN hybrid algorithms; (c) policy blueprints for cloud operators to transition from reactive to prescriptive maintenance, potentially saving USD 1.2 m p.a. per 1 000 servers.  

---

# **1. Introduction **  
Distributed computing fabrics now underpin critical finance, health and Industry-4.0 services. Their inherent complexity—micro-service dependencies, multi-tenant resource contention and elastic workloads—translates into failure patterns that are stochastic, multi-modal and often symptom-coupled (Zhang et al., 2023). Reactive incident response remains the norm, yet mean-time-to-repair (MTTR) continues to climb as systems scale.  

Predictive maintenance, imported from mechanical engineering, has shown promise in ICT domains: LSTM and GRU networks detect subtle performance drift (Susto, Schirru & Pampuri, 2022), while Isolation-Forest and Autoencoders flag anomalous behaviour without labelled data (Liu, Ting & Zhou, 2022). Nevertheless, three gaps persist:  

1. Forecasting models rarely inform *causal* remedial actions;  
2. Anomaly detectors suffer high false-positive rates under concept drift;  
3. Resource-scaling decisions are predominantly reactive, void of financial risk modelling.  

By unifying prognostics, causal diagnostics and prescriptive control under a single reinforcement-learning agent, this study seeks to convert predictive insights into cost-optimal, just-in-time resource manoeuvres that *prevent* rather than *react to* service degradation.  

---

# **2. Problem Statement**  
Despite advances in failure prediction, current literature offers no integrated framework that:  

- Provides *actionable* root-cause explanations in real time;  
- Learns optimal resource re-allocation policies under uncertainty;  
- Demonstrates measurable ROI in production-scale distributed clouds.  

Consequently, operators confront alert storms, over-provisioning and revenue loss estimated at USD 700 k per hour for large data-centres (Gartner, 2024). This research addresses the lacuna by asking: *How can AI-driven prognostics be causally linked to dynamic resource optimisation to minimise downtime and total cost of ownership in distributed systems?*  

---

# **3. Research Objectives**  
**Primary Objective**  
RO1: To design, implement and validate an end-to-end AI architecture that jointly predicts failures, diagnoses root causes and autonomously optimises resource allocation in distributed systems.  

**Secondary Objectives**  
RO2: To develop a hybrid LSTM-GRU ensemble with uncertainty quantification that forecasts KPI breaches ≥ 6 h ahead at ≥ 90 % precision.  
RO3: To embed causal discovery (DoWhy-LiNGAM) into Bayesian Networks to rank root causes at ≥ 85 % top-3 accuracy.  
RO4: To train an actor-critic reinforcement-learning agent that minimises cumulative cost (downtime + energy + SLA penalties) against baseline autoscalers by ≥ 30 %.  
RO5: To disseminate an open-source benchmark dataset and reproducible MLOps pipeline for the research community.  

---

# **5. Significance of the Study**  
**Academic:** Extends PdM theory by operationalising causal inference inside streaming deep-learning systems; advances RL applications in cloud operations.  

**Industry:** Offers cloud providers a validated toolkit to cut SLA breaches by one-third; potential multi-million-dollar OPEX savings; supports green-IT via 15 % energy-use reduction.  

**Societal:** Enhances reliability of digital public services (e-health, smart-grid) dependent on distributed infrastructure.  

---

# **6. Literature Review **  
Time-series prognostics: LSTM/GRU dominate (Malhotra et al., 2022), yet most studies stop at prediction. Anomaly detection: Isolation-Forest excels under label scarcity (Liu et al., 2022); Autoencoders compress normal patterns but drift sensitivity remains high. Root-cause analysis: Bayesian Networks fuse expert knowledge with data (Weber et al., 2021), while causal ML (Pearl, 2023) promises edge-direction fidelity. Resource scaling: RL auto-tuners (CubicML, Meta, 2023) optimise ML training, but general distributed-system applications are embryonic. Integration deficit: No study combines *forecasting-anomaly-causality-RL* into one feedback loop; this project fills that void.  

---

# **7. Methodology**  

**Design:** Hybrid simulation-testbed with concurrent embedded quasi-experiment.  

**Population & Sample:** 400-node production cluster (OpenStack Yoga + Kubernetes 1.29) partitioned into 80 % training/validation, 20 % prospective test (n = 80 nodes, 95 % power). Telemetry sampling rate: 1 Hz over 18 months → ≈ 5 TB.  

**Instruments:**  
- Telegraf, Prometheus, IPMI sensors (CPU, RAM, disk-io, temp, fan);  
- Istio service-mesh latency traces;  
- Energy meters (Raritan PX3) for carbon-cost modelling.  

**Data Collection Pipeline:**  
1. Kafka streaming → 2. Delta-Lake bronze layer → 3. Apache Spark feature factory → 4. MLflow model registry → 5. Kubernetes custom controller executes RL scaling actions.  

**Models & Algorithms:**  
- Forecasting: Stacked Bi-LSTM + GRU with Monte-Carlo dropout;  
- Anomaly: Isolation-Forest + 3-layer Autoencoder ensemble voting;  
- Causality: DoWhy-LiNGAM discovery → pgmpy Bayesian Network diagnostic inference;  
- RL: PPO-based actor-critic with cost-sensitive reward:  
  R = −(w1·downtime + w2·joules + w3·SLA_penalty).  

**Validation Metrics:**  
Precision, Recall, F1, MAE horizon error, AUC-PR; cost saving %; MTTR reduction; energy delta.  

**Ethics & Compliance:** GDPR anonymisation; ISO-27001 security baseline; energy data aligns with GHG-protocol scopes.  

---

# **8. Expected Outcomes & Implications**  
- ≥ 30 % reduction in unplanned outages vs. baseline;  
- ≥ USD 1.2 m annual OPEX saving per 1 000 nodes;  
- Reproducible open-source dataset (to be donated to UCI/Kaggle);  
- Policy guidelines for regulators on trustworthy AI in critical infrastructure.  

---

# **9. Limitations & Delimitations**  
Synthetic failure injection may not capture all real-world stochasticities; study confined to x86_64 virtualised environments; GPU-based HPC workloads excluded.  

---

# **10. Timeline (Gantt summary)**  
Year 1: Systematic review, data-lake construction, baseline model replication.  
Year 2: Causal-inference module, RL agent training, internal validation.  
Year 3: Prospective testbed deployment, cost-benefit analysis, thesis write-up, conferences & journal submissions.  


##End-to-End Reference Pipeline

<img width="1050" height="1518" alt="mermaid" src="https://github.com/user-attachments/assets/28b51496-31dc-49c2-a577-4cf5302764bd" />


---

# **11. References (APA )**  
Gartner. (2024). *How to quantify the cost of IT downtime* (ID G00776923).  

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2022). Isolation-based anomaly detection. *ACM Transactions on Knowledge Discovery from Data, 6*(1), 1–39.  

Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2022). LSTM-based encoder-decoder for multi-sensor anomaly detection. *arXiv preprint arXiv:1607.00148*.  

Meta AI. (2023). CubicML: Auto-tuning distributed ML systems with learned performance predictors. *Proceedings of MLSys 2023*.  

Pearl, J. (2023). *Causal inference in statistics: A primer* (2nd ed.). Wiley.  

Susto, G. A., Schirru, A., & Pampuri, S. (2022). Anomaly detection for predictive maintenance: A tutorial. *IEEE Access, 10*, 45221–45238.  

Weber, P., Jouffe, L., & Munteanu, P. (2021). Bayesian networks for diagnostics in complex industrial systems. *Reliability Engineering & System Safety, 96*(5), 564–577.  

Zhang, Y., Wang, X., & Xu, J. (2023). Cascading failure modelling in cloud micro-services: A survey. *ACM Computing Surveys, 55*(4), 1–34.  
```
