# Evaluation Summary

## Binary SFT, lr 8e-6, 330 steps

Summer-camp testset:

- Accuracy: 0.7303240740740741
- F1 unsafe: 0.7791469194312797
- Precision unsafe: 0.6771004942339374
- Recall unsafe: 0.9174107142857143
- Invalid rate: 0.0

R-Judge, binary safe/unsafe:

- Accuracy: 0.8102836879432624
- F1 unsafe: 0.831496062992126
- Precision unsafe: 0.7833827893175074
- Recall unsafe: 0.8859060402684564
- Precision safe: 0.8502202643171806
- Recall safe: 0.7255639097744361
- Invalid rate: 0.0

## Taxonomy LoRA, lr 8e-6, 200 steps

ATBench 300:

- Accuracy: 0.52
- Unsafe precision / recall / F1: 0.5394736842105263 / 0.2733333333333333 / 0.3628318584070796
- Safe precision / recall / F1: 0.5133928571428571 / 0.7666666666666667 / 0.6149732620320856
- Risk source accuracy: 0.10666666666666667
- Failure mode accuracy: 0.14333333333333334
- Harm type accuracy: 0.17
- Exact 4D accuracy: 0.0
- Invalid rate: 0.0

R-Judge, binary safe/unsafe only:

- Accuracy: 0.5230496453900709
- Unsafe precision / recall / F1: 0.8717948717948718 / 0.11409395973154363 / 0.20178041543026706
- Safe precision / recall / F1: 0.49714285714285716 / 0.981203007518797 / 0.6599241466498104
- Invalid rate: 0.0

## Application, binary SFT lr 8e-6 / 330 steps, 50 cases each

Email:

- Decision accuracy: 0.76
- Unsafe recall: 1.0
- Safe pass rate: 1.0
- Block precision: 0.6129032258064516
- Ask-confirm accuracy: 0.0

Database:

- Decision accuracy: 0.96
- Unsafe recall: 1.0
- Safe pass rate: 1.0
- Block precision: 0.9166666666666666
- Ask-confirm accuracy: 0.8333333333333334
