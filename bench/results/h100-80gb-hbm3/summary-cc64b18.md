# Final results, commit cc64b18

Medians of per-run medians across the runs listed; ranges in parentheses are the smallest and largest run. Ratios to flash-attn are taken within each run.

## decode_batch (3 runs)

| config | impl | ms | GB/s | % of peak | speed vs flash-attn |
|---|---|---|---|---|---|
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_copy | 0.298 (0.298-0.298) | 28.2 | 0.8 | 0.07 (0.06-0.07) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_inplace | 0.312 (0.312-0.312) | 27.0 | 0.8 | 0.06 (0.06-0.06) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | splitkv | 0.052 (0.052-0.052) | 160.8 | 4.8 | 0.37 (0.37-0.37) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | flash_attn | 0.019 (0.019-0.020) | 434.1 | 13.0 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_flash | 0.019 (0.019-0.020) | 433.4 | 12.9 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=4, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_copy | 0.453 (0.453-0.454) | 74.2 | 2.2 | 0.07 (0.06-0.07) |
| batch=4, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_inplace | 0.321 (0.321-0.321) | 104.8 | 3.1 | 0.09 (0.09-0.09) |
| batch=4, heads=32, heads_kv=8, context=2048, head_dim=128 | splitkv | 0.059 (0.059-0.059) | 566.2 | 16.9 | 0.50 (0.49-0.50) |
| batch=4, heads=32, heads_kv=8, context=2048, head_dim=128 | flash_attn | 0.029 (0.029-0.029) | 1142.0 | 34.1 | 1.00 (1.00-1.00) |
| batch=4, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_flash | 0.030 (0.030-0.030) | 1132.1 | 33.8 | 0.99 (0.98-0.99) |
| batch=4, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=16, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_copy | 0.648 (0.647-0.650) | 207.4 | 6.2 | 0.10 (0.10-0.10) |
| batch=16, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_inplace | 0.325 (0.324-0.325) | 414.4 | 12.4 | 0.20 (0.20-0.20) |
| batch=16, heads=32, heads_kv=8, context=2048, head_dim=128 | splitkv | 0.149 (0.149-0.149) | 904.5 | 27.0 | 0.45 (0.44-0.45) |
| batch=16, heads=32, heads_kv=8, context=2048, head_dim=128 | flash_attn | 0.066 (0.066-0.066) | 2030.7 | 60.6 | 1.00 (1.00-1.00) |
| batch=16, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_flash | 0.066 (0.066-0.067) | 2024.3 | 60.4 | 1.00 (1.00-1.00) |
| batch=16, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=64, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_copy | 2.320 (2.319-2.321) | 231.9 | 6.9 | 0.09 (0.09-0.09) |
| batch=64, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_inplace | 0.643 (0.643-0.644) | 836.2 | 25.0 | 0.32 (0.32-0.32) |
| batch=64, heads=32, heads_kv=8, context=2048, head_dim=128 | splitkv | 0.491 (0.490-0.491) | 1096.4 | 32.7 | 0.42 (0.42-0.42) |
| batch=64, heads=32, heads_kv=8, context=2048, head_dim=128 | flash_attn | 0.206 (0.205-0.206) | 2615.9 | 78.1 | 1.00 (1.00-1.00) |
| batch=64, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_flash | 0.218 (0.214-0.219) | 2472.8 | 73.8 | 0.95 (0.94-0.96) |
| batch=64, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_efficient | unsupported | | | |

## decode_ctx (3 runs)

| config | impl | ms | GB/s | % of peak | speed vs flash-attn |
|---|---|---|---|---|---|
| batch=1, heads=32, heads_kv=32, context=512, head_dim=128 | decode_copy | 0.111 (0.110-0.111) | 75.7 | 2.3 | 0.15 (0.15-0.15) |
| batch=1, heads=32, heads_kv=32, context=512, head_dim=128 | decode_inplace | 0.082 (0.081-0.082) | 103.0 | 3.1 | 0.21 (0.20-0.21) |
| batch=1, heads=32, heads_kv=32, context=512, head_dim=128 | splitkv | 0.048 (0.048-0.048) | 174.4 | 5.2 | 0.35 (0.34-0.35) |
| batch=1, heads=32, heads_kv=32, context=512, head_dim=128 | flash_attn | 0.017 (0.017-0.017) | 501.3 | 15.0 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=32, context=512, head_dim=128 | sdpa_flash | 0.017 (0.017-0.017) | 492.8 | 14.7 | 0.98 (0.98-0.99) |
| batch=1, heads=32, heads_kv=32, context=512, head_dim=128 | sdpa_efficient | 0.025 (0.025-0.025) | 340.2 | 10.2 | 0.68 (0.67-0.68) |
| batch=1, heads=32, heads_kv=32, context=2048, head_dim=128 | decode_copy | 0.449 (0.448-0.450) | 74.7 | 2.2 | 0.07 (0.06-0.07) |
| batch=1, heads=32, heads_kv=32, context=2048, head_dim=128 | decode_inplace | 0.306 (0.305-0.306) | 109.8 | 3.3 | 0.10 (0.10-0.10) |
| batch=1, heads=32, heads_kv=32, context=2048, head_dim=128 | splitkv | 0.056 (0.056-0.056) | 597.1 | 17.8 | 0.52 (0.52-0.53) |
| batch=1, heads=32, heads_kv=32, context=2048, head_dim=128 | flash_attn | 0.029 (0.029-0.030) | 1140.3 | 34.0 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=32, context=2048, head_dim=128 | sdpa_flash | 0.030 (0.030-0.030) | 1128.1 | 33.7 | 0.99 (0.99-0.99) |
| batch=1, heads=32, heads_kv=32, context=2048, head_dim=128 | sdpa_efficient | 0.079 (0.079-0.079) | 424.4 | 12.7 | 0.37 (0.37-0.37) |
| batch=1, heads=32, heads_kv=32, context=8192, head_dim=128 | decode_copy | 1.693 (1.692-1.693) | 79.3 | 2.4 | 0.04 (0.04-0.04) |
| batch=1, heads=32, heads_kv=32, context=8192, head_dim=128 | decode_inplace | 1.184 (1.184-1.186) | 113.4 | 3.4 | 0.06 (0.06-0.06) |
| batch=1, heads=32, heads_kv=32, context=8192, head_dim=128 | splitkv | 0.080 (0.080-0.080) | 1679.9 | 50.1 | 0.85 (0.85-0.85) |
| batch=1, heads=32, heads_kv=32, context=8192, head_dim=128 | flash_attn | 0.068 (0.067-0.068) | 1986.2 | 59.3 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=32, context=8192, head_dim=128 | sdpa_flash | 0.068 (0.068-0.068) | 1975.9 | 59.0 | 0.99 (0.99-1.00) |
| batch=1, heads=32, heads_kv=32, context=8192, head_dim=128 | sdpa_efficient | 0.283 (0.281-0.284) | 474.6 | 14.2 | 0.24 (0.24-0.24) |
| batch=1, heads=32, heads_kv=32, context=32768, head_dim=128 | decode_copy | 6.531 (6.529-6.558) | 82.2 | 2.5 | 0.03 (0.03-0.03) |
| batch=1, heads=32, heads_kv=32, context=32768, head_dim=128 | decode_inplace | 4.689 (4.687-4.699) | 114.5 | 3.4 | 0.04 (0.04-0.04) |
| batch=1, heads=32, heads_kv=32, context=32768, head_dim=128 | splitkv | 0.247 (0.247-0.247) | 2171.0 | 64.8 | 0.78 (0.78-0.78) |
| batch=1, heads=32, heads_kv=32, context=32768, head_dim=128 | flash_attn | 0.193 (0.193-0.193) | 2779.6 | 83.0 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=32, context=32768, head_dim=128 | sdpa_flash | 0.193 (0.193-0.194) | 2776.2 | 82.9 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=32, context=32768, head_dim=128 | sdpa_efficient | 1.060 (1.059-1.071) | 506.4 | 15.1 | 0.18 (0.18-0.18) |
| batch=1, heads=32, heads_kv=8, context=512, head_dim=128 | decode_copy | 0.102 (0.102-0.102) | 20.7 | 0.6 | 0.14 (0.14-0.14) |
| batch=1, heads=32, heads_kv=8, context=512, head_dim=128 | decode_inplace | 0.084 (0.083-0.084) | 25.3 | 0.8 | 0.17 (0.17-0.17) |
| batch=1, heads=32, heads_kv=8, context=512, head_dim=128 | splitkv | 0.049 (0.049-0.049) | 43.0 | 1.3 | 0.29 (0.28-0.29) |
| batch=1, heads=32, heads_kv=8, context=512, head_dim=128 | flash_attn | 0.014 (0.014-0.014) | 149.3 | 4.5 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=8, context=512, head_dim=128 | sdpa_flash | 0.015 (0.014-0.015) | 145.5 | 4.3 | 0.97 (0.97-0.98) |
| batch=1, heads=32, heads_kv=8, context=512, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_copy | 0.288 (0.286-0.288) | 29.2 | 0.9 | 0.07 (0.07-0.07) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_inplace | 0.311 (0.311-0.312) | 27.0 | 0.8 | 0.06 (0.06-0.06) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | splitkv | 0.052 (0.052-0.052) | 161.1 | 4.8 | 0.37 (0.37-0.37) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | flash_attn | 0.019 (0.019-0.019) | 434.1 | 13.0 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_flash | 0.019 (0.019-0.019) | 432.7 | 12.9 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=1, heads=32, heads_kv=8, context=8192, head_dim=128 | decode_copy | 1.522 (1.521-1.528) | 22.1 | 0.7 | 0.02 (0.02-0.02) |
| batch=1, heads=32, heads_kv=8, context=8192, head_dim=128 | decode_inplace | 1.222 (1.222-1.224) | 27.5 | 0.8 | 0.03 (0.03-0.03) |
| batch=1, heads=32, heads_kv=8, context=8192, head_dim=128 | splitkv | 0.064 (0.064-0.064) | 521.9 | 15.6 | 0.53 (0.53-0.53) |
| batch=1, heads=32, heads_kv=8, context=8192, head_dim=128 | flash_attn | 0.034 (0.034-0.034) | 981.4 | 29.3 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=8, context=8192, head_dim=128 | sdpa_flash | 0.034 (0.034-0.035) | 975.0 | 29.1 | 0.99 (0.99-0.99) |
| batch=1, heads=32, heads_kv=8, context=8192, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=1, heads=32, heads_kv=8, context=32768, head_dim=128 | decode_copy | 6.023 (6.019-6.043) | 22.3 | 0.7 | 0.01 (0.01-0.01) |
| batch=1, heads=32, heads_kv=8, context=32768, head_dim=128 | decode_inplace | 4.863 (4.861-4.871) | 27.6 | 0.8 | 0.01 (0.01-0.01) |
| batch=1, heads=32, heads_kv=8, context=32768, head_dim=128 | splitkv | 0.199 (0.199-0.200) | 674.2 | 20.1 | 0.36 (0.36-0.36) |
| batch=1, heads=32, heads_kv=8, context=32768, head_dim=128 | flash_attn | 0.072 (0.071-0.072) | 1874.4 | 56.0 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=8, context=32768, head_dim=128 | sdpa_flash | 0.072 (0.072-0.072) | 1867.7 | 55.8 | 1.00 (0.99-1.00) |
| batch=1, heads=32, heads_kv=8, context=32768, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=8, heads=32, heads_kv=32, context=512, head_dim=128 | decode_copy | 0.233 (0.233-0.234) | 288.2 | 8.6 | 0.17 (0.17-0.17) |
| batch=8, heads=32, heads_kv=32, context=512, head_dim=128 | decode_inplace | 0.092 (0.091-0.092) | 734.7 | 21.9 | 0.44 (0.44-0.44) |
| batch=8, heads=32, heads_kv=32, context=512, head_dim=128 | splitkv | 0.061 (0.061-0.061) | 1102.4 | 32.9 | 0.66 (0.66-0.66) |
| batch=8, heads=32, heads_kv=32, context=512, head_dim=128 | flash_attn | 0.040 (0.040-0.041) | 1663.7 | 49.7 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=32, context=512, head_dim=128 | sdpa_flash | 0.041 (0.040-0.041) | 1655.8 | 49.4 | 1.00 (0.99-1.00) |
| batch=8, heads=32, heads_kv=32, context=512, head_dim=128 | sdpa_efficient | 0.039 (0.038-0.039) | 1745.2 | 52.1 | 1.05 (1.05-1.05) |
| batch=8, heads=32, heads_kv=32, context=2048, head_dim=128 | decode_copy | 0.788 (0.787-0.788) | 341.0 | 10.2 | 0.14 (0.14-0.14) |
| batch=8, heads=32, heads_kv=32, context=2048, head_dim=128 | decode_inplace | 0.314 (0.314-0.315) | 854.5 | 25.5 | 0.35 (0.35-0.35) |
| batch=8, heads=32, heads_kv=32, context=2048, head_dim=128 | splitkv | 0.128 (0.128-0.128) | 2100.0 | 62.7 | 0.86 (0.86-0.86) |
| batch=8, heads=32, heads_kv=32, context=2048, head_dim=128 | flash_attn | 0.110 (0.110-0.111) | 2448.3 | 73.1 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=32, context=2048, head_dim=128 | sdpa_flash | 0.115 (0.115-0.116) | 2333.3 | 69.6 | 0.95 (0.95-0.96) |
| batch=8, heads=32, heads_kv=32, context=2048, head_dim=128 | sdpa_efficient | 0.116 (0.116-0.116) | 2319.7 | 69.2 | 0.95 (0.95-0.96) |
| batch=8, heads=32, heads_kv=32, context=8192, head_dim=128 | decode_copy | 3.067 (3.066-3.067) | 350.2 | 10.5 | 0.13 (0.13-0.13) |
| batch=8, heads=32, heads_kv=32, context=8192, head_dim=128 | decode_inplace | 1.205 (1.205-1.205) | 891.1 | 26.6 | 0.33 (0.32-0.33) |
| batch=8, heads=32, heads_kv=32, context=8192, head_dim=128 | splitkv | 0.432 (0.432-0.432) | 2483.4 | 74.1 | 0.91 (0.90-0.92) |
| batch=8, heads=32, heads_kv=32, context=8192, head_dim=128 | flash_attn | 0.395 (0.391-0.398) | 2718.6 | 81.2 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=32, context=8192, head_dim=128 | sdpa_flash | 0.423 (0.421-0.424) | 2536.0 | 75.7 | 0.94 (0.92-0.94) |
| batch=8, heads=32, heads_kv=32, context=8192, head_dim=128 | sdpa_efficient | 0.425 (0.423-0.427) | 2524.5 | 75.4 | 0.93 (0.92-0.93) |
| batch=8, heads=32, heads_kv=32, context=32768, head_dim=128 | decode_copy | 12.114 (12.066-12.115) | 354.5 | 10.6 | 0.13 (0.13-0.13) |
| batch=8, heads=32, heads_kv=32, context=32768, head_dim=128 | decode_inplace | 4.762 (4.753-4.763) | 901.9 | 26.9 | 0.34 (0.34-0.34) |
| batch=8, heads=32, heads_kv=32, context=32768, head_dim=128 | splitkv | 1.596 (1.594-1.596) | 2691.7 | 80.3 | 1.02 (1.02-1.02) |
| batch=8, heads=32, heads_kv=32, context=32768, head_dim=128 | flash_attn | 1.632 (1.623-1.632) | 2632.1 | 78.6 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=32, context=32768, head_dim=128 | sdpa_flash | 1.821 (1.804-1.856) | 2358.1 | 70.4 | 0.90 (0.87-0.90) |
| batch=8, heads=32, heads_kv=32, context=32768, head_dim=128 | sdpa_efficient | 1.719 (1.715-1.722) | 2498.0 | 74.6 | 0.95 (0.95-0.95) |
| batch=8, heads=32, heads_kv=8, context=512, head_dim=128 | decode_copy | 0.142 (0.142-0.143) | 118.8 | 3.5 | 0.15 (0.15-0.15) |
| batch=8, heads=32, heads_kv=8, context=512, head_dim=128 | decode_inplace | 0.087 (0.087-0.087) | 194.1 | 5.8 | 0.24 (0.24-0.24) |
| batch=8, heads=32, heads_kv=8, context=512, head_dim=128 | splitkv | 0.052 (0.052-0.053) | 322.2 | 9.6 | 0.40 (0.40-0.41) |
| batch=8, heads=32, heads_kv=8, context=512, head_dim=128 | flash_attn | 0.021 (0.021-0.021) | 795.8 | 23.8 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=8, context=512, head_dim=128 | sdpa_flash | 0.022 (0.021-0.022) | 786.3 | 23.5 | 0.99 (0.98-0.99) |
| batch=8, heads=32, heads_kv=8, context=512, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=8, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_copy | 0.507 (0.507-0.508) | 132.5 | 4.0 | 0.08 (0.08-0.08) |
| batch=8, heads=32, heads_kv=8, context=2048, head_dim=128 | decode_inplace | 0.324 (0.324-0.324) | 207.7 | 6.2 | 0.13 (0.13-0.13) |
| batch=8, heads=32, heads_kv=8, context=2048, head_dim=128 | splitkv | 0.103 (0.103-0.103) | 652.5 | 19.5 | 0.42 (0.42-0.42) |
| batch=8, heads=32, heads_kv=8, context=2048, head_dim=128 | flash_attn | 0.043 (0.043-0.043) | 1568.1 | 46.8 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_flash | 0.043 (0.043-0.043) | 1556.5 | 46.5 | 0.99 (0.99-0.99) |
| batch=8, heads=32, heads_kv=8, context=2048, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=8, heads=32, heads_kv=8, context=8192, head_dim=128 | decode_copy | 1.869 (1.869-1.875) | 143.7 | 4.3 | 0.06 (0.06-0.06) |
| batch=8, heads=32, heads_kv=8, context=8192, head_dim=128 | decode_inplace | 1.242 (1.242-1.244) | 216.2 | 6.5 | 0.09 (0.09-0.09) |
| batch=8, heads=32, heads_kv=8, context=8192, head_dim=128 | splitkv | 0.283 (0.283-0.283) | 948.8 | 28.3 | 0.38 (0.38-0.38) |
| batch=8, heads=32, heads_kv=8, context=8192, head_dim=128 | flash_attn | 0.108 (0.108-0.108) | 2486.4 | 74.2 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=8, context=8192, head_dim=128 | sdpa_flash | 0.108 (0.108-0.109) | 2478.6 | 74.0 | 1.00 (0.99-1.00) |
| batch=8, heads=32, heads_kv=8, context=8192, head_dim=128 | sdpa_efficient | unsupported | | | |
| batch=8, heads=32, heads_kv=8, context=32768, head_dim=128 | decode_copy | 7.567 (7.530-7.569) | 141.9 | 4.2 | 0.05 (0.05-0.05) |
| batch=8, heads=32, heads_kv=8, context=32768, head_dim=128 | decode_inplace | 4.926 (4.916-4.926) | 218.0 | 6.5 | 0.07 (0.07-0.07) |
| batch=8, heads=32, heads_kv=8, context=32768, head_dim=128 | splitkv | 0.946 (0.944-0.946) | 1135.0 | 33.9 | 0.38 (0.38-0.38) |
| batch=8, heads=32, heads_kv=8, context=32768, head_dim=128 | flash_attn | 0.357 (0.357-0.358) | 3004.3 | 89.7 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=8, context=32768, head_dim=128 | sdpa_flash | 0.358 (0.358-0.358) | 2998.2 | 89.5 | 1.00 (1.00-1.00) |
| batch=8, heads=32, heads_kv=8, context=32768, head_dim=128 | sdpa_efficient | unsupported | | | |

## prefill_bf16 (3 runs)

| config | impl | ms | TFLOP/s | % of peak | speed vs flash-attn |
|---|---|---|---|---|---|
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | mma | 0.862 (0.859-0.862) | 79.7 | 8.1 | 0.23 (0.23-0.23) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | mma_pipelined | 0.313 (0.313-0.316) | 219.3 | 22.2 | 0.63 (0.62-0.63) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | triton | 0.188 (0.188-0.189) | 365.6 | 36.9 | 1.05 (1.05-1.05) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | sdpa_flash | 0.210 (0.209-0.211) | 326.9 | 33.0 | 0.94 (0.94-0.94) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | sdpa_cudnn | 0.120 (0.119-0.121) | 575.0 | 58.1 | 1.65 (1.63-1.66) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | sdpa_efficient | 0.412 (0.409-0.417) | 166.6 | 16.8 | 0.48 (0.47-0.48) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | flash_attn | 0.197 (0.197-0.198) | 349.1 | 35.3 | 1.00 (1.00-1.00) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | mma | 0.497 (0.497-0.497) | 69.1 | 7.0 | 0.29 (0.29-0.30) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | mma_pipelined | 0.194 (0.193-0.195) | 177.4 | 17.9 | 0.75 (0.75-0.76) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | triton | 0.130 (0.128-0.130) | 265.1 | 26.8 | 1.13 (1.13-1.13) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | sdpa_flash | 0.159 (0.159-0.159) | 216.0 | 21.8 | 0.92 (0.92-0.93) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | sdpa_cudnn | 0.085 (0.085-0.086) | 402.8 | 40.7 | 1.71 (1.71-1.72) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | sdpa_efficient | 0.259 (0.258-0.261) | 132.6 | 13.4 | 0.56 (0.56-0.56) |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | flash_attn | 0.146 (0.146-0.147) | 235.0 | 23.7 | 1.00 (1.00-1.00) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | mma | 10.517 (10.514-10.518) | 104.5 | 10.6 | 0.28 (0.28-0.29) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | mma_pipelined | 4.798 (4.797-4.841) | 229.2 | 23.2 | 0.62 (0.62-0.63) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | triton | 2.486 (2.482-2.500) | 442.3 | 44.7 | 1.20 (1.20-1.20) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | sdpa_flash | 3.346 (3.315-3.408) | 328.7 | 33.2 | 0.89 (0.87-0.91) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | sdpa_cudnn | 1.801 (1.770-1.836) | 610.5 | 61.7 | 1.65 (1.64-1.68) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | sdpa_efficient | 6.061 (6.061-6.062) | 181.4 | 18.3 | 0.49 (0.49-0.50) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | flash_attn | 2.977 (2.977-3.005) | 369.3 | 37.3 | 1.00 (1.00-1.00) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | mma | 5.742 (5.735-5.758) | 95.7 | 9.7 | 0.29 (0.28-0.30) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | mma_pipelined | 2.537 (2.496-2.546) | 216.7 | 21.9 | 0.67 (0.62-0.68) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | triton | 1.405 (1.390-1.412) | 391.4 | 39.6 | 1.21 (1.13-1.22) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | sdpa_flash | 1.831 (1.826-1.856) | 300.3 | 30.4 | 0.92 (0.86-0.93) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | sdpa_cudnn | 1.011 (0.959-1.011) | 543.8 | 55.0 | 1.67 (1.56-1.78) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | sdpa_efficient | 3.288 (3.279-3.293) | 167.2 | 16.9 | 0.51 (0.48-0.52) |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | flash_attn | 1.690 (1.581-1.707) | 325.4 | 32.9 | 1.00 (1.00-1.00) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=False | mma | 18.939 (18.927-18.958) | 116.1 | 11.7 | 0.33 (0.32-0.33) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=False | mma_pipelined | 9.526 (9.525-9.593) | 230.8 | 23.3 | 0.64 (0.64-0.65) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=False | triton | 5.013 (4.949-5.067) | 438.7 | 44.3 | 1.22 (1.22-1.25) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=False | sdpa_flash | 6.404 (6.348-6.405) | 343.4 | 34.7 | 0.96 (0.96-0.96) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=False | sdpa_cudnn | 3.723 (3.682-3.745) | 590.6 | 59.7 | 1.66 (1.65-1.66) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=False | sdpa_efficient | 12.129 (12.123-12.130) | 181.3 | 18.3 | 0.51 (0.50-0.51) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=False | flash_attn | 6.168 (6.112-6.168) | 356.5 | 36.0 | 1.00 (1.00-1.00) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=True | mma | 10.546 (10.544-10.550) | 104.3 | 10.5 | 0.29 (0.29-0.30) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=True | mma_pipelined | 4.904 (4.878-4.911) | 224.2 | 22.7 | 0.64 (0.63-0.64) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=True | triton | 2.596 (2.583-2.617) | 423.5 | 42.8 | 1.20 (1.19-1.20) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=True | sdpa_flash | 3.704 (3.634-3.731) | 296.9 | 30.0 | 0.84 (0.84-0.85) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=True | sdpa_cudnn | 1.954 (1.855-1.976) | 562.6 | 56.9 | 1.58 (1.57-1.68) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=True | sdpa_efficient | 6.254 (6.245-6.276) | 175.8 | 17.8 | 0.50 (0.49-0.50) |
| batch=2, heads=32, heads_kv=32, seqlen=8192, head_dim=128, causal=True | flash_attn | 3.105 (3.096-3.122) | 354.2 | 35.8 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=False | mma | 37.742 (37.706-37.794) | 116.5 | 11.8 | 0.32 (0.32-0.32) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=False | mma_pipelined | 19.285 (19.142-19.311) | 228.1 | 23.0 | 0.62 (0.62-0.62) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=False | triton | 9.925 (9.924-10.190) | 443.1 | 44.8 | 1.20 (1.17-1.21) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=False | sdpa_flash | 13.215 (13.212-13.275) | 332.8 | 33.6 | 0.90 (0.90-0.91) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=False | sdpa_cudnn | 7.153 (7.152-7.154) | 614.9 | 62.1 | 1.67 (1.67-1.68) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=False | sdpa_efficient | 24.935 (24.822-24.951) | 176.4 | 17.8 | 0.48 (0.48-0.48) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=False | flash_attn | 11.947 (11.946-12.020) | 368.1 | 37.2 | 1.00 (1.00-1.00) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=True | mma | 18.613 (18.612-18.614) | 118.1 | 11.9 | 0.34 (0.34-0.35) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=True | mma_pipelined | 9.495 (9.494-9.496) | 231.6 | 23.4 | 0.67 (0.67-0.68) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=True | triton | 4.995 (4.944-5.049) | 440.2 | 44.5 | 1.28 (1.26-1.29) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=True | sdpa_flash | 6.902 (6.886-6.915) | 318.6 | 32.2 | 0.92 (0.92-0.93) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=True | sdpa_cudnn | 3.725 (3.705-3.835) | 590.3 | 59.7 | 1.70 (1.66-1.74) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=True | sdpa_efficient | 12.409 (12.381-12.447) | 177.2 | 17.9 | 0.51 (0.51-0.52) |
| batch=1, heads=32, heads_kv=32, seqlen=16384, head_dim=128, causal=True | flash_attn | 6.355 (6.347-6.436) | 346.0 | 35.0 | 1.00 (1.00-1.00) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | mma | 1.129 (1.127-1.129) | 121.8 | 12.3 | 0.39 (0.39-0.39) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | mma_pipelined | 0.687 (0.687-0.689) | 200.2 | 20.2 | 0.64 (0.64-0.64) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | triton | 0.360 (0.354-0.363) | 382.2 | 38.6 | 1.23 (1.20-1.24) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | sdpa_flash | 0.463 (0.458-0.475) | 296.7 | 30.0 | 0.95 (0.92-0.96) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | sdpa_cudnn | 0.276 (0.275-0.276) | 498.8 | 50.4 | 1.59 (1.58-1.60) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | sdpa_efficient | 1.044 (1.044-1.051) | 131.6 | 13.3 | 0.42 (0.42-0.42) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | flash_attn | 0.438 (0.437-0.441) | 313.9 | 31.7 | 1.00 (1.00-1.00) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | mma | 0.640 (0.639-0.644) | 107.4 | 10.9 | 0.43 (0.43-0.44) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | mma_pipelined | 0.388 (0.386-0.393) | 177.2 | 17.9 | 0.72 (0.70-0.72) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | triton | 0.226 (0.222-0.226) | 303.9 | 30.7 | 1.23 (1.22-1.25) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | sdpa_flash | 0.281 (0.280-0.283) | 244.9 | 24.8 | 0.99 (0.98-0.99) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | sdpa_cudnn | 0.181 (0.181-0.182) | 379.5 | 38.4 | 1.53 (1.53-1.54) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | sdpa_efficient | 0.591 (0.588-0.598) | 116.3 | 11.8 | 0.47 (0.47-0.47) |
| batch=8, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | flash_attn | 0.278 (0.277-0.278) | 247.1 | 25.0 | 1.00 (1.00-1.00) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=False | mma | 8.838 (8.832-8.839) | 124.4 | 12.6 | 0.34 (0.33-0.34) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=False | mma_pipelined | 4.835 (4.793-4.836) | 227.4 | 23.0 | 0.62 (0.61-0.63) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=False | triton | 2.475 (2.445-2.486) | 444.3 | 44.9 | 1.20 (1.19-1.24) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=False | sdpa_flash | 3.227 (3.227-3.256) | 340.7 | 34.4 | 0.92 (0.91-0.94) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=False | sdpa_cudnn | 1.892 (1.873-1.909) | 581.1 | 58.7 | 1.59 (1.56-1.59) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=False | sdpa_efficient | unsupported | | | |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=False | flash_attn | 2.978 (2.952-3.030) | 369.2 | 37.3 | 1.00 (1.00-1.00) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=True | mma | 4.679 (4.666-4.689) | 117.5 | 11.9 | 0.35 (0.35-0.35) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=True | mma_pipelined | 2.587 (2.489-2.608) | 212.5 | 21.5 | 0.63 (0.63-0.66) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=True | triton | 1.371 (1.328-1.372) | 400.9 | 40.5 | 1.20 (1.20-1.23) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=True | sdpa_flash | 1.793 (1.776-1.797) | 306.6 | 31.0 | 0.91 (0.91-0.93) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=True | sdpa_cudnn | 0.906 (0.904-0.990) | 606.7 | 61.3 | 1.81 (1.65-1.82) |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=True | sdpa_efficient | unsupported | | | |
| batch=4, heads=32, heads_kv=8, seqlen=4096, head_dim=128, causal=True | flash_attn | 1.640 (1.632-1.650) | 335.1 | 33.9 | 1.00 (1.00-1.00) |

## prefill_fp32 (3 runs)

| config | impl | ms | TFLOP/s | % of peak | speed vs flash-attn |
|---|---|---|---|---|---|
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | naive | 18.022 (17.890-18.022) | 3.8 | 5.7 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | fp32_fused | 9.234 (9.163-9.234) | 7.4 | 11.1 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | fp32_regtile | 4.988 (4.963-4.990) | 13.8 | 20.6 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | sdpa_math | 3.022 (3.010-3.024) | 22.7 | 33.9 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=False | sdpa_efficient | 1.570 (1.556-1.582) | 43.8 | 65.3 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | naive | 17.927 (17.927-17.927) | 1.9 | 2.9 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | fp32_fused | 4.998 (4.993-5.003) | 6.9 | 10.3 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | fp32_regtile | 2.890 (2.888-2.891) | 11.9 | 17.7 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | sdpa_math | 3.480 (3.478-3.481) | 9.9 | 14.7 |  |
| batch=4, heads=32, heads_kv=32, seqlen=1024, head_dim=128, causal=True | sdpa_efficient | 0.911 (0.910-0.912) | 37.7 | 56.3 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | naive | 286.846 (284.888-286.846) | 3.8 | 5.7 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | fp32_fused | 142.904 (141.837-142.908) | 7.7 | 11.5 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | fp32_regtile | 62.356 (62.093-62.521) | 17.6 | 26.3 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | sdpa_math | 42.976 (42.960-43.044) | 25.6 | 38.2 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=False | sdpa_efficient | 23.842 (23.840-23.844) | 46.1 | 68.8 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | naive | 283.286 (283.274-285.262) | 1.9 | 2.9 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | fp32_fused | 72.543 (72.515-73.029) | 7.6 | 11.3 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | fp32_regtile | 34.224 (34.215-34.357) | 16.1 | 24.0 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | sdpa_math | 53.058 (52.824-53.059) | 10.4 | 15.5 |  |
| batch=4, heads=32, heads_kv=32, seqlen=4096, head_dim=128, causal=True | sdpa_efficient | 12.113 (12.112-12.114) | 45.4 | 67.7 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | naive | 9.774 (9.767-9.833) | 3.5 | 5.2 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | fp32_fused | 4.693 (4.693-4.725) | 7.3 | 10.9 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | fp32_regtile | 1.055 (1.054-1.062) | 32.6 | 48.6 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | sdpa_math | 2.126 (2.125-2.132) | 16.2 | 24.1 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=False | sdpa_efficient | 1.057 (1.056-1.062) | 32.5 | 48.5 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | naive | 9.741 (9.740-9.782) | 1.8 | 2.6 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | fp32_fused | 2.845 (2.844-2.863) | 6.0 | 9.0 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | fp32_regtile | 0.691 (0.690-0.693) | 24.9 | 37.1 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | sdpa_math | 2.708 (2.707-2.709) | 6.3 | 9.5 |  |
| batch=2, heads=16, heads_kv=16, seqlen=2048, head_dim=64, causal=True | sdpa_efficient | 0.682 (0.674-0.684) | 25.2 | 37.6 |  |

## e2e_llama (3 runs)

Model meta-llama/Llama-3.1-8B-Instruct.

| prompt | impl | time to first token (ms) | decode step (ms) | tokens/s | max logit error vs fp32 |
|---|---|---|---|---|---|
| 512 | flash_attention_2 | incorrect | | | 13.242 |
| 512 | flash_lab | 28.7 (28.7-28.9) | 8.21 (8.21-8.25) | 121.8 | 0.064 |
| 512 | flash_lab_cuda | 23.1 (23.1-23.3) | 8.22 (8.20-8.22) | 121.7 | 0.067 |
| 512 | sdpa | 24.8 (24.6-25.0) | 10.40 (10.36-10.44) | 96.2 | 0.064 |
| 8192 | flash_attention_2 | incorrect | | | 2.616 |
| 8192 | flash_lab | 274.8 (273.9-275.5) | 9.58 (9.41-9.59) | 104.3 | 0.074 |
| 8192 | flash_lab_cuda | 311.7 (311.2-312.4) | 9.58 (9.35-9.59) | 104.4 | 0.099 |
| 8192 | sdpa | 523.3 (520.5-523.4) | 22.68 (22.56-22.85) | 44.1 | 0.094 |
