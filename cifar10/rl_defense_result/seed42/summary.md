| Attack | Defence | Final recon MSE | Final PSNR | Final SSIM | Lowest MSE reached | Accept / downweight / clip / reject | ms per round |
|---|---|---|---|---|---|---|---|
| URVFL, 25 ep | No defence (original URVFL) | 0.0310 | 15.10 | 0.590 | 0.0310 | n/a | n/a |
| URVFL, 25 ep | RL defence (ours) | 0.3902 | 4.11 | -0.019 | 0.3902 | 0.00 / 0.00 / 0.00 / 1.00 | 9.47 |
| URVFL, 25 ep | Simple clipping | 0.0240 | 16.21 | 0.658 | 0.0240 | 0.00 / 0.00 / 1.00 / 0.00 | 9.07 |
| URVFL, 25 ep | SplitGuard (detected at round 140, training stopped) | 0.1843 | 7.36 | 0.205 | 0.1843 | n/a | n/a |
| URVFL, small aux (held-out), 10 ep | No defence (original URVFL) | 0.0727 | 11.39 | 0.363 | 0.0727 | n/a | n/a |
| URVFL, small aux (held-out), 10 ep | RL defence (ours) | 0.3218 | 4.94 | -0.039 | 0.3218 | 0.00 / 0.00 / 0.00 / 1.00 | 19.72 |
| URVFL, small aux (held-out), 10 ep | Simple clipping | 0.0569 | 12.46 | 0.464 | 0.0569 | 0.00 / 0.00 / 1.00 / 0.00 | 14.97 |
| URVFL, 5-class aux (held-out), 10 ep | No defence (original URVFL) | 0.0551 | 12.60 | 0.453 | 0.0551 | n/a | n/a |
| URVFL, 5-class aux (held-out), 10 ep | RL defence (ours) | 0.3700 | 4.34 | -0.021 | 0.3700 | 0.00 / 0.00 / 0.00 / 1.00 | 13.49 |
| URVFL, 5-class aux (held-out), 10 ep | Simple clipping | 0.0795 | 11.01 | 0.345 | 0.0763 | 0.00 / 0.00 / 1.00 / 0.00 | 19.15 |
| URVFL-sync (held-out), 10 ep | No defence (original URVFL) | 0.0125 | 19.04 | 0.782 | 0.0125 | n/a | n/a |
| URVFL-sync (held-out), 10 ep | RL defence (ours) | 0.0173 | 17.62 | 0.719 | 0.0173 | 0.59 / 0.00 / 0.00 / 0.41 | 16.76 |
| URVFL-sync (held-out), 10 ep | Simple clipping | 0.0174 | 17.61 | 0.710 | 0.0174 | 0.00 / 0.00 / 1.00 / 0.00 | 9.36 |
| FSHA, 10 ep | No defence (original URVFL) | 0.0662 | n/a | n/a | 0.0472 | n/a | n/a |
| FSHA, 10 ep | RL defence (ours) | 0.2581 | n/a | n/a | 0.2378 | 0.00 / 0.00 / 0.00 / 1.00 | 9.51 |
| FSHA, 10 ep | Simple clipping | 0.1061 | n/a | n/a | 0.0564 | 0.00 / 0.00 / 1.00 / 0.00 | 9.29 |
| AGN, 10 ep | No defence (original URVFL) | 0.6615 | n/a | n/a | 0.4441 | n/a | n/a |
| AGN, 10 ep | RL defence (ours) | 0.5558 | n/a | n/a | 0.5011 | 0.00 / 0.00 / 0.00 / 1.00 | 43.87 |
| AGN, 10 ep | Simple clipping | 0.6032 | n/a | n/a | 0.4254 | 0.00 / 0.00 / 1.00 / 0.00 | 40.38 |

| Honest VFL | Final test accuracy (%) | Mean accuracy, last 3 epochs (%) | False-positive rate (non-accept share) |
|---|---|---|---|
| No defence | 52.10 | 51.11 | n/a |
| RL defence (ours) | 48.70 | 50.15 | 0.000 |
| Simple clipping | 50.10 | 50.99 | 1.000 |
| SplitGuard (no false alarm) | 45.57 | n/a | 0.000 |
