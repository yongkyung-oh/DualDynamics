# Stable Neural Stochastic Differential Equations (Neural SDEs)
This repository contains the PyTorch implementation for the paper [DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis](https://arxiv.org/abs/2401.04979).

> Oh, Y., Lim, D., & Kim, S. (2025, Feb). DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis. In The 39th Annual AAAI Conference on Artificial Intelligence.

> Oh, Y., Lim, D., & Kim, S. (2025). DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis. arXiv preprint arXiv:2401.04979.

---

# **Code architecture**
The code for each experiment is meticulously organized into separate folders, aligned with the original references used for implementation. 

- `exp_classification`: PhysioNet Sepsis, implemented from Kidger, P. et al. (2020) [1] (https://github.com/patrick-kidger/NeuralCDE)
- `exp_interpolation`: PhysionNet Mortality, implemented from Shukla, S. et al. (2020) [2] (https://github.com/reml-lab/mTAN)
- `exp_MuJoCo`: MuJoCo Foresting task, implemented from Jhin, S. et al. (2021) [3] (https://github.com/sheoyon-jhin/ANCDE)
- `exp_Google_2021`: Google Foresting task, implemented from Jhin, S. et al. (2022) [4] (https://github.com/sheoyon-jhin/EXIT)
- `torch_ists`: motivated from Kidger et al. (2020) [1] (https://github.com/patrick-kidger/NeuralCDE) and Oh et al. (2024) [5] (https://github.com/yongkyung-oh/Stable-Neural-SDEs), we develop new python/pytorch wrapper for extensive experiments on robustness to missing data

[1] Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural controlled differential equations for irregular time series. Advances in Neural Information Processing Systems, 33, 6696-6707.

[2] Shukla, S. N., & Marlin, B. (2020, October). Multi-Time Attention Networks for Irregularly Sampled Time Series. In International Conference on Learning Representations.

[3] Jhin, S. Y., Shin, H., Hong, S., Jo, M., Park, S., Park, N., ... & Jeon, S. (2021, December). Attentive Neural Controlled Differential Equations for Time-series Classification and Forecasting. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 250-259). IEEE Computer Society.

[4] Jhin, S. Y., Lee, J., Jo, M., Kook, S., Jeon, J., Hyeong, J., ... & Park, N. (2022, April). Exit: Extrapolation and interpolation-based neural controlled differential equations for time-series classification and forecasting. In Proceedings of the ACM Web Conference 2022 (pp. 3102-3112).
 
[5] Oh, Y., Lim, D., & Kim, S. (2024), Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data, The Twelfth International Conference on Learning Representations (ICLR) 2024, May 2024. Spotlight presentation (Notable Top 5%). 

---

**Current State of the Code and Future Plans**:
- It is acknowledged that the current version of the code is somewhat messy. This candid admission suggests ongoing development and refinement of the codebase.
- Despite its current state, the code provides valuable insights into the code-level details of the implementation, which can be beneficial for researchers and practitioners interested in understanding or replicating the study.
- Future efforts may focus on cleaning and documenting the code further to enhance its accessibility and usability for the wider research community.

---

**Robustness to Missingness Experiment**:

We are refactoring our experimental pipeline to use the independent library `torch-ists`. For reproducibility, we recommend using this updated version. You can find the library at the [torch-ists repository](https://github.com/yongkyung-oh/torch-ists).

---

## Reference
```
@inproceedings{
    oh2025dualdynamics,
    title={DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis},
    author={YongKyung Oh and Dongyoung Lim and Sungil Kim},
    booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
    year={2025},
}
```
