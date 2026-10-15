# Third-party notices

## EyeNavGS datasets and derived Normal HMD runtime material

This repository includes processed runtime exports at:

```text
demo/UnityNormalHmdAgentDemo/Assets/Resources/NormalHmdAgentModel.json
demo/UnityRandomAgentDemo/Assets/Resources/EyeNavAgentModel.json
```

The current Normal HMD export contains learned parameters, normalization/calibration metadata, golden vectors,
and 414 short processed validation seed windows from 104 trajectories and 7 pseudonymous participants, derived
from the EyeNavGS Rutgers and NTHU datasets. The older Random Agent export contains 504 processed initialization
windows from 84 trajectories and 7 pseudonymous participants. The windows have been converted into
project-specific motion representations and are used to initialize seeded Unity rollouts. They are not the
complete original trajectory CSV files.

Upstream works:

- EyeNavGS Rutgers Dataset: <https://github.com/symmru/EyeNavGS_Rutgers_Dataset>, audited commit
  `7f264fda0223cbc74e191f115ef9fd3b580d8cf4`.
- EyeNavGS NTHU Dataset: <https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset>, audited commit
  `608790e7d932534f18a2a124887605d1f6c85384`.
- Project page: <https://symmru.github.io/EyeNavGS/>.

Authors named by the upstream dataset:

Zihao Ding, Cheng-Tse Lee, Mufeng Zhu, Tao Guan, Yuan-Chun Sun, Cheng-Hsin Hsu, and Yao Liu;
Rutgers University and National Tsing Hua University.

Upstream copyright notices include:

- Copyright 2025 Zihao Ding, Cheng-Tse Lee, Mufeng Zhu, Tao Guan, Yuan-Chun Sun, Cheng-Hsin Hsu, Yao Liu.
- Copyright 2025 Systems and Multimedia Lab @ Rutgers University.

License: Apache License, Version 2.0. A copy is provided at `LICENSES/Apache-2.0.txt`.

Suggested citation:

```bibtex
@article{ding2025eyenavgs,
  title={EyeNavGS: A 6-DoF Navigation Dataset and Record-n-Replay Software for Real-World 3DGS Scenes in VR},
  author={Ding, Zihao and Lee, Cheng-Tse and Zhu, Mufeng and Guan, Tao and Sun, Yuan-Chun and Hsu, Cheng-Hsin and Liu, Yao},
  journal={arXiv preprint arXiv:2506.02380},
  year={2025}
}
```

The original CSV datasets are not redistributed in this repository.

## Habitat-Web

No Habitat-Web dataset or trained weight is redistributed in this snapshot. The repository contains independent
preparation/training source code and documentation for an experiment that used Habitat-Web ObjectNav THDA 70k.
The upstream demonstration datasets are licensed under Creative Commons Attribution-NonCommercial 4.0:

- <https://github.com/Ram81/habitat-imitation-baselines>
- <https://huggingface.co/datasets/axel81/habitat-web>
- <https://creativecommons.org/licenses/by-nc/4.0/>

Do not add the Habitat-Web data or derived prior weights to a release without preserving attribution and checking
the non-commercial restriction.

## VR.net

No VR.net raw data, event-level features, predictions, or model pickle is redistributed. The repository contains
independent processing/training source and aggregate metrics only. The current public package's redistribution
license was not sufficiently established during this snapshot, so conservative exclusion is intentional.
