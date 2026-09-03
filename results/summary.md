# Results summary (auto-generated from results/*.json — do not edit by hand)

## Latency & memory (H100, GPU 1, batch=1)

| model | precision | chunk | e2e ms (mean±std) | vision | prefill | decode | preprocess | weights GB | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| openvla-7b | bf16 | - | 195.0±2.0 | 9.2 | 30.7 | 126.6 | 13.3 | 14.09 | 14.38 |
| openvla-7b | int4 | - | 201.3±2.0 | 18.1 | 52.8 | 101.3 | 13.3 | 4.08 | 4.37 |
| openvla-7b | int8 | - | 308.2±4.8 | 30.5 | 47.5 | 200.7 | 13.3 | 7.43 | 7.72 |
| smolvla-450m | bf16 | 1 | 172.7±2.2 | 14.2 | 9.1 | 81.4 | 7.1 | 0.89 | 0.91 |
| smolvla-450m | bf16 | 5 | 177.1±2.8 | 14.0 | 9.2 | 83.8 | 6.8 | 0.89 | 0.91 |
| smolvla-450m | bf16 | 10 | 178.2±3.2 | 14.1 | 9.1 | 84.6 | 7.3 | 0.89 | 0.91 |
| smolvla-450m | bf16 | 25 | 178.7±2.6 | 14.3 | 9.1 | 85.5 | 6.9 | 0.89 | 0.91 |
| smolvla-450m | bf16 | 50 | 177.8±2.7 | 14.0 | 9.0 | 85.0 | 7.0 | 0.89 | 0.91 |
| smolvla-450m | bf16 (graph) | 50 | 93.0±2.3 | 14.2 | 12.2 | 0.0 | 7.0 | 0.96 | 0.98 |
| smolvla-450m | bf16 (hoist) | 50 | 162.1±19.6 | 14.4 | 12.3 | 116.3 | 7.2 | 0.89 | 0.91 |
| smolvla-450m | fp32 | 1 | 181.7±18.9 | 14.5 | 10.4 | 92.2 | 6.9 | 0.90 | 0.92 |
| smolvla-450m | fp32 (graph) | 1 | 82.5±1.5 | 14.4 | 9.1 | 0.0 | 7.1 | 0.96 | 0.98 |
| smolvla-450m | fp32 (hoist) | 1 | 129.7±1.9 | 14.4 | 9.0 | 90.9 | 6.9 | 0.90 | 0.92 |
| smolvla-450m | fp32 | 5 | 171.2±1.6 | 14.2 | 8.9 | 84.2 | 7.0 | 0.90 | 0.92 |
| smolvla-450m | fp32 | 10 | 172.5±2.1 | 14.0 | 8.9 | 84.6 | 7.0 | 0.90 | 0.92 |
| smolvla-450m | fp32 (graph) | 10 | 92.2±1.4 | 14.3 | 11.5 | 0.0 | 6.9 | 0.96 | 0.99 |
| smolvla-450m | fp32 (hoist) | 10 | 162.9±14.3 | 14.2 | 11.0 | 118.8 | 6.9 | 0.90 | 0.92 |
| smolvla-450m | fp32 | 25 | 173.9±1.4 | 14.2 | 8.9 | 85.0 | 7.1 | 0.90 | 0.92 |
| smolvla-450m | fp32 | 50 | 175.7±1.4 | 14.4 | 8.9 | 85.9 | 7.1 | 0.90 | 0.92 |
| smolvla-450m | fp32 (graph) | 50 | 96.1±2.3 | 14.4 | 11.2 | 0.0 | 6.6 | 0.96 | 0.99 |
| smolvla-450m | fp32 (hoist) | 50 | 132.9±2.7 | 14.4 | 9.0 | 91.3 | 6.8 | 0.90 | 0.92 |
| smolvla-450m | int4 | 1 | 204.1±1.7 | 14.8 | 12.0 | 108.5 | 6.9 | 0.46 | 0.48 |
| smolvla-450m | int4 | 5 | 208.1±2.6 | 14.9 | 12.0 | 110.8 | 6.8 | 0.46 | 0.48 |
| smolvla-450m | int4 | 10 | 211.4±2.1 | 14.8 | 12.2 | 113.5 | 7.1 | 0.46 | 0.48 |
| smolvla-450m | int4 | 25 | 209.8±2.8 | 14.6 | 12.0 | 112.8 | 6.9 | 0.46 | 0.48 |
| smolvla-450m | int4 | 50 | 213.9±5.7 | 14.7 | 12.0 | 115.8 | 7.0 | 0.46 | 0.48 |
| smolvla-450m | int8 | 1 | 263.2±4.5 | 15.9 | 17.3 | 160.6 | 6.9 | 0.65 | 0.68 |
| smolvla-450m | int8 | 5 | 266.4±1.7 | 15.9 | 16.9 | 161.8 | 7.2 | 0.65 | 0.68 |
| smolvla-450m | int8 | 10 | 266.2±2.1 | 15.6 | 17.0 | 162.2 | 7.1 | 0.65 | 0.68 |
| smolvla-450m | int8 | 25 | 269.5±4.0 | 15.7 | 17.1 | 165.4 | 6.8 | 0.65 | 0.68 |
| smolvla-450m | int8 | 50 | 267.1±2.1 | 15.7 | 16.8 | 163.1 | 7.2 | 0.65 | 0.68 |

## Deviation vs reference (OpenVLA: BF16 ref; SmolVLA: fp32 as-shipped ref)

| model | precision | chunk | metric(s) |
|---|---|---|---|
| openvla-7b | int4 | - | token_mismatch_rate=0.6381, action_l2_mean=0.5229, action_l2_std=0.4747, action_l2_relative=11.71, steps_fully_matching=0 |
| openvla-7b | int8 | - | token_mismatch_rate=0.6143, action_l2_mean=0.1467, action_l2_std=0.2845, action_l2_relative=0.5529, steps_fully_matching=0 |
| smolvla-450m | bf16 | 1 | chunk_mse=0.0002582, action_l2_mean=0.03696, action_l2_std=0.01353, action_l2_relative=0.01869, max_abs_diff=0.05529 |
| smolvla-450m | bf16 | 10 | chunk_mse=0.0007516, action_l2_mean=0.05296, action_l2_std=0.04129, action_l2_relative=0.02158, max_abs_diff=0.3072 |
| smolvla-450m | bf16 | 25 | chunk_mse=0.001935, action_l2_mean=0.08404, action_l2_std=0.06741, action_l2_relative=0.02335, max_abs_diff=0.4296 |
| smolvla-450m | bf16 | 5 | chunk_mse=0.0003847, action_l2_mean=0.04308, action_l2_std=0.02127, action_l2_relative=0.01929, max_abs_diff=0.1293 |
| smolvla-450m | bf16 | 50 | chunk_mse=0.002418, action_l2_mean=0.1008, action_l2_std=0.06596, action_l2_relative=0.02139, max_abs_diff=0.5443 |
| smolvla-450m | bf16 | 50 | chunk_mse=0.00281, action_l2_mean=0.09639, action_l2_std=0.08698, action_l2_relative=0.02042, max_abs_diff=0.605 |
| smolvla-450m | bf16 | 50 | chunk_mse=0.00281, action_l2_mean=0.09639, action_l2_std=0.08698, action_l2_relative=0.02042, max_abs_diff=0.605 |
| smolvla-450m | fp32 | 10 | chunk_mse=0, action_l2_mean=0, action_l2_std=0, action_l2_relative=0, max_abs_diff=0 |
| smolvla-450m | fp32 | 10 | chunk_mse=0, action_l2_mean=0, action_l2_std=0, action_l2_relative=0, max_abs_diff=0 |
| smolvla-450m | fp32 | 1 | chunk_mse=0, action_l2_mean=0, action_l2_std=0, action_l2_relative=0, max_abs_diff=0 |
| smolvla-450m | fp32 | 1 | chunk_mse=0, action_l2_mean=0, action_l2_std=0, action_l2_relative=0, max_abs_diff=0 |
| smolvla-450m | fp32 | 50 | chunk_mse=0, action_l2_mean=0, action_l2_std=0, action_l2_relative=0, max_abs_diff=0 |
| smolvla-450m | fp32 | 50 | chunk_mse=0, action_l2_mean=0, action_l2_std=0, action_l2_relative=0, max_abs_diff=0 |
| smolvla-450m | int4 | 1 | chunk_mse=0.03495, action_l2_mean=0.4258, action_l2_std=0.1685, action_l2_relative=0.2225, max_abs_diff=0.7349 |
| smolvla-450m | int4 | 10 | chunk_mse=0.07759, action_l2_mean=0.572, action_l2_std=0.3719, action_l2_relative=0.2378, max_abs_diff=2.786 |
| smolvla-450m | int4 | 25 | chunk_mse=0.1735, action_l2_mean=0.8306, action_l2_std=0.5926, action_l2_relative=0.2259, max_abs_diff=2.946 |
| smolvla-450m | int4 | 5 | chunk_mse=0.04256, action_l2_mean=0.4675, action_l2_std=0.1919, action_l2_relative=0.2229, max_abs_diff=1.092 |
| smolvla-450m | int4 | 50 | chunk_mse=0.2438, action_l2_mean=1.038, action_l2_std=0.6209, action_l2_relative=0.2047, max_abs_diff=2.979 |
| smolvla-450m | int8 | 1 | chunk_mse=0.002058, action_l2_mean=0.1024, action_l2_std=0.04304, action_l2_relative=0.05388, max_abs_diff=0.1763 |
| smolvla-450m | int8 | 10 | chunk_mse=0.005238, action_l2_mean=0.1525, action_l2_std=0.09041, action_l2_relative=0.06507, max_abs_diff=0.5174 |
| smolvla-450m | int8 | 25 | chunk_mse=0.02178, action_l2_mean=0.2899, action_l2_std=0.2159, action_l2_relative=0.08108, max_abs_diff=1.167 |
| smolvla-450m | int8 | 5 | chunk_mse=0.003161, action_l2_mean=0.1226, action_l2_std=0.06279, action_l2_relative=0.05687, max_abs_diff=0.5174 |
| smolvla-450m | int8 | 50 | chunk_mse=0.03521, action_l2_mean=0.3765, action_l2_std=0.2636, action_l2_relative=0.08058, max_abs_diff=1.18 |