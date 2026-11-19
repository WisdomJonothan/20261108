# Replacement case studies (dev, Direct policy)

Chosen swaps only; delta is the per-query nDCG@10 change from the single swap.

## Largest gains

| domain | qid | slot | cand rank | source | judged P | dense gap | delta |
|---|---|---:|---:|---|---:|---:|---:|
| monero | 4053_899 | 8 | 4 | dense | 0.985 | 0.020 | +0.3155 |
| history | 58699_662 | 9 | 6 | ce | 0.998 | -0.024 | +0.3010 |
| law | 106440_824 | 8 | 5 | dense | 1.000 | 0.060 | +0.1934 |
| history | 57118_700 | 9 | 1 | ce | 0.992 | 0.013 | +0.1846 |
| history | 58540_674 | 10 | 4 | dense | 1.000 | 0.016 | +0.1772 |
| travel | 188098_1152 | 10 | 2 | ce | 0.999 | -0.016 | +0.1772 |

## Largest losses

| domain | qid | slot | cand rank | source | judged P | dense gap | delta |
|---|---|---:|---:|---|---:|---:|---:|
| economics | 47636_213 | 8 | 6 | ce | 1.000 | -0.015 | -0.1934 |
| history | 70693_263 | 10 | 5 | dense | 0.828 | 0.017 | -0.1772 |
| bitcoin | 124603_8 | 8 | 3 | dense | 0.920 | 0.021 | -0.1480 |
| history | 77003_69 | 8 | 9 | dense | 1.000 | -0.006 | -0.1480 |
| history | 77154_56 | 8 | 6 | dense | 0.988 | 0.043 | -0.1480 |
| history | 59294_622 | 9 | 3 | ce | 1.000 | 0.002 | -0.1413 |

## Swaps by domain

| domain | swaps | helpful | harmful | net delta |
|---|---:|---:|---:|---:|
| bitcoin | 22 | 0 | 2 | -0.2550 |
| cardano | 9 | 1 | 0 | +0.1413 |
| economics | 19 | 2 | 1 | +0.0252 |
| genealogy | 22 | 0 | 0 | +0.0000 |
| history | 135 | 17 | 16 | +0.3632 |
| hsm | 32 | 2 | 0 | +0.2656 |
| iota | 2 | 0 | 0 | +0.0000 |
| law | 5 | 1 | 0 | +0.1934 |
| monero | 14 | 1 | 0 | +0.3155 |
| politics | 27 | 3 | 0 | +0.3839 |
| quant | 7 | 1 | 0 | +0.1232 |
| travel | 23 | 4 | 0 | +0.6078 |
| workplace | 7 | 0 | 0 | +0.0000 |

## Query text for the cases above


Queries: 
- `monero/4053_899`: Are [solo] mining yield calculations like dice or playing card calculations?
 If I want to know how many times I need to roll a die in order to have a given chance of seeing at least one  six , I can use the formula: 

 
- `history/58699_662`: Where was Mao Zedong during the Japanese surrender in Zhijiang (芷江) at the end of WWII in 1945?
 Timeline for context: 

 
 9 August, 1945:  atomic bomb on Nagasaki . 
 15 August: the Japanese announced their acceptance 
- `economics/47636_213`: How to explain the meaning of the coefficient of interest of Difference-in-Differences?
 Considering a Difference-in-Differences (DiD) equation with a dummy variable: 
  $$
y_i = \alpha + \beta D_i + Xit + \varepsilon_i.
- `history/70693_263`: How did Raphael paint halos?
 Yesterday, I had the pleasure of seeing  Raphael's cartoon  "Christ's Charge to Peter": 
     
 In it, Christ and the apostles are depicted with halos that appear to be perfect  ellipses . 

