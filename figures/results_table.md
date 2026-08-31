| Arm | Divergence | Advantage | avg@4 (best) | greedy | A100 min | rollouts |
| --- | --- | --- | --- | --- | --- | --- |
| SFT init (student) | - | - | 0.331 | 0.490 | 3.4 | 0 |
| +2x off-policy SFT data | - | - | 0.366 | 0.550 | 4.5 | 0 |
| GRPO (outcome reward RL) | - | - | 0.404 | 0.540 | 17.5 | 2560 |
| Reverse KL / OPD | reverse_kl | opd | 0.494 | 0.585 | 18.6 | 2560 |
| Reverse KL / OPD+ | reverse_kl | opd_plus | 0.472 | 0.555 | 18.6 | 2560 |
| Forward KL / OPD | forward_kl | opd | 0.331 | 0.490 | 18.6 | 2560 |
| Forward KL / OPD+ | forward_kl | opd_plus | 0.450 | 0.470 | 19.2 | 2560 |
| JSD / OPD | jsd | opd | 0.331 | 0.490 | 18.4 | 2560 |
| JSD / OPD+ | jsd | opd_plus | 0.460 | 0.405 | 18.6 | 2560 |
