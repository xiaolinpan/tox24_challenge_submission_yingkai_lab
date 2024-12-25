# Enhancing TTR Binding Affinity Prediction with a Deep Ensemble Model: Insights from the Tox24 Challenge 

<p align="justify">
Transthyretin (TTR) plays a vital role in thyroid hormone transport and homeostasis in both blood and target tissues. Interactions between exogenous compounds and TTR can disrupt the function of endocrine system, potentially causing toxicity. In the Tox24 challenge, we leveraged the largest dataset provided by the organizers to develop a deep learning-based ensemble model, integrating sPhysNet, KANO, and GGAP-CPI for predicting TTR binding affinity. Each model utilized distinct levels of molecular information, including 2D topology, 3D geometry, and protein-ligand interactions. Our ensemble model achieved favorable performance on the blind test set, yielding an RMSE of 20.8. By incorporating the leaderboard test set into our training data, we further improved the RMSE to 19.9, surpassing all submissions. These results demonstrate that combining three regression models across different modalities significantly enhances predictive accuracy. Furthermore, by analyzing model uncertainty, we observed that both the RMSE and interval error of predictions increase with rising uncertainty, indicating that the model’s uncertainty can serve as a useful measure of prediction confidence. We believe this ensemble model can be a valuable resource for identifying potential TTR binders and predicting their binding affinity in silico.
</p>

---
<div align="center">
    <img src="https://github.com/xiaolinpan/tox24_challenge_submission_yingkai_lab/blob/main/images/TOC.png" alt="image" width="450"/>
</div>

## Requirements

* [sPhysNet](https://github.com/xiaolinpan/sPhysNet-Taut)
* [KANO](https://github.com/HICAI-ZJU/KANO)
* [GGAP-CPI](https://github.com/gu-yaowen/GGAP-CPI)

