# Qwen3-4B Stage2-Full Parserfix Official Tau2 Eval

- checkpoint: `qwen3_4b_stage2_full_1epoch/checkpoint-final`
- user/judge: `openai/openai/gpt-5.2`
- split: `test`, seed: `300`

| save | pass | n | pass_rate | tool_calls | user_tags | unexpected_kwarg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage2_full_parserfix_20260617_005840_test_airline20 | 6 | 20 | 0.300 | 407 | 0 | 0 |
| stage2_full_parserfix_20260617_005840_test_retail40 | 14 | 40 | 0.350 | 317 | 0 | 0 |
| stage2_full_parserfix_20260617_005840_test_telecom40 | 15 | 40 | 0.375 | 605 | 0 | 0 |
| total | 35 | 100 | 0.350 | 1329 | 0 | 0 |
