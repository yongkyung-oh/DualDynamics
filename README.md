# DualDynamics

**DualDynamics: Synergizing Implicit and Explicit Methods for Robust Irregular
Time Series Analysis**

DualDynamics is a deep learning framework for **irregular time series analysis** that
combines **implicit** (Neural Differential Equation–based) and **explicit**
(Neural Flow–based) methods to stay accurate and robust under missing data and irregular
sampling. This repository provides the official PyTorch implementation, with experiments
covering classification, interpolation, and forecasting on PhysioNet, MuJoCo, and Google.

**Authors:** YongKyung Oh, Dong-Young Lim, Sungil Kim
**Venue:** Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-25),
vol. 39, no. 18, pp. 19730–19739, 2025.
**Links:** [Paper (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/34173) ·
[DOI: 10.1609/aaai.v39i18.34173](https://doi.org/10.1609/aaai.v39i18.34173) ·
[arXiv:2401.04979](https://arxiv.org/abs/2401.04979)

**Keywords:** Machine Learning (ML): ML: General, Machine Learning (ML): ML:
Classification and Regression, Machine Learning (ML): ML: Time-Series/Data
Streams, Data Mining & Knowledge Management (DMKM): DMKM: Mining of Spatial &
Temporal or Spatio-Temporal Data

## Overview

Irregularly sampled and partially observed time series are common in healthcare, sensing,
and scientific data. DualDynamics synergizes implicit continuous-time dynamics (NDEs)
with explicit Neural Flow–based methods to deliver robust performance across classification,
interpolation, and forecasting tasks, including under varying levels of missingness.

> Reproducibility: robustness-to-missingness experiments use the standalone
> [`torch-ists`](https://github.com/yongkyung-oh/torch-ists) library; the bundled
> `torch-ists/` folder mirrors that wrapper.

## Code architecture

The code for each experiment is organized into separate folders, aligned with the
original references used for implementation.

- `exp_classification`: PhysioNet Sepsis, from Kidger, P. et al. (2020) [1] ([NeuralCDE](https://github.com/patrick-kidger/NeuralCDE))
- `exp_interpolation`: PhysioNet Mortality, from Shukla, S. et al. (2021) [2] ([mTAN](https://github.com/reml-lab/mTAN))
- `exp_MuJoCo`: MuJoCo forecasting, from Jhin, S. et al. (2021) [3] ([ANCDE](https://github.com/sheoyon-jhin/ANCDE))
- `exp_Google_2021`: Google forecasting, from Jhin, S. et al. (2022) [4] ([EXIT](https://github.com/sheoyon-jhin/EXIT))
- `torch-ists`: a Python/PyTorch wrapper for robustness-to-missing-data experiments,
  motivated by Kidger et al. (2020) [1] and Oh et al. (2024) [5] ([Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs))

## References

[1] Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). Neural controlled
differential equations for irregular time series. Advances in Neural Information
Processing Systems, 33, 6696-6707.
[2] Shukla, S. N., & Marlin, B. M. (2021). Multi-Time Attention Networks for
Irregularly Sampled Time Series. In 9th International Conference on Learning Representations (ICLR 2021). https://openreview.net/forum?id=4c0J6lwQ4_
[3] Jhin, S. Y., Shin, H., Hong, S., Jo, M., Park, S., Park, N., ... & Jeon, S. (2021,
December). Attentive Neural Controlled Differential Equations for Time-series
Classification and Forecasting. In 2021 IEEE International Conference on Data Mining
(ICDM) (pp. 250-259). IEEE Computer Society.
[4] Jhin, S. Y., Lee, J., Jo, M., Kook, S., Jeon, J., Hyeong, J., ... & Park, N. (2022,
April). Exit: Extrapolation and interpolation-based neural controlled differential
equations for time-series classification and forecasting. In Proceedings of the ACM Web
Conference 2022 (pp. 3102-3112).
[5] Oh, Y., Lim, D., & Kim, S. (2024). Stable Neural Stochastic Differential Equations in
Analyzing Irregular Time Series Data. The Twelfth International Conference on Learning
Representations (ICLR) 2024, May 2024. Spotlight presentation (Notable Top 5%).

## Citation

If you use this work, please cite the AAAI-25 paper. Machine-readable metadata is also
available in [`CITATION.cff`](CITATION.cff).

```bibtex
@inproceedings{oh_dualdynamics_2025,
  title     = {{DualDynamics}: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis},
  author    = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
  year      = 2025,
  booktitle = {Proceedings of the {AAAI} Conference on Artificial Intelligence},
  publisher = {AAAI Press},
  volume    = 39,
  number    = 18,
  pages     = {19730--19739},
  doi       = {10.1609/aaai.v39i18.34173}
}
```

```bibtex
@misc{oh_dualdynamics_2024_arxiv,
  title     = {{DualDynamics}: Synergizing Implicit and Explicit Methods for Robust Irregular Time Series Analysis},
  author    = {Oh, YongKyung and Lim, Dong-Young and Kim, Sungil},
  year      = {2024},
  publisher = {arXiv},
  doi       = {10.48550/arXiv.2401.04979},
  url       = {https://arxiv.org/abs/2401.04979}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
