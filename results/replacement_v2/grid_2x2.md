| judge | passage window | nDCG@10 | delta | swaps | helpful | harmful | beneficial-action recall | harmful loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct | 112 words | 0.364609 | +0.006510 | 324 | 32 | 19 | 0.153 | -2.316 |
| Qwen3-4B-Instruct | 400 words | 0.365998 | +0.007900 | 331 | 36 | 10 | 0.172 | -1.283 |
| Llama-3.1-8B-Instruct | 112 words | 0.362195 | +0.004096 | 330 | 24 | 12 | 0.115 | -1.532 |

Qwen3-4B-Instruct / 400 words minus Direct-112 words: +0.001390, CI [-0.00082,+0.00438], 21 better / 11 worse
Llama-3.1-8B-Instruct / 112 words minus Direct-112 words: -0.002414, CI [-0.00608,+0.00086], 21 better / 22 worse
