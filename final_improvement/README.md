# Final Improvement

This experiment addresses the main limitations identified in the Week 1-10
review without replacing the original results.

## Changes

- Uses a chronological split: training through April 2026, validation on May
  2026, and the untouched final test on June 2026.
- Compares the original dollar target with `log1p(ClosePrice)`.
- Restores useful location categories such as ZIP code, city, county, school
  district, subdivision, and MLS area.
- Spatially assigns Elementary, High, and Unified school districts using the
  Week 6 boundary shapefile.
- Adds sale month, property age, HOA, garage, ratio, and log-scaled size
  features.
- Uses LightGBM native categorical handling instead of silently dropping string
  columns.
- Learns numeric medians and category levels from the training period only.
- Excludes final days-on-market and contract-to-close timing because those
  values are not available when estimating a home before its sale closes.

## Run

From the repository root:

```bash
python final_improvement/07_improved_model.py
```

For a faster diagnostic run without the spatial district join:

```bash
python final_improvement/07_improved_model.py --skip-district-join
```

The script writes its results to `final_improvement/outputs/`:

- `model_comparison.csv`
- `log_lightgbm_test_predictions.csv`
- `log_lightgbm_feature_importance.csv`
- `run_metadata.json`

The final June test set is used only for reporting. Model iteration selection is
based on the May validation period. After iteration selection, the model is
refitted through May with the fixed iteration count before the final June test.

## Final Result

The production-safe log-target model achieved the following result on the June
2026 test period:

| R2 | RMSE | MAE | MAPE | MdAPE | RMSLE |
|---:|---:|---:|---:|---:|---:|
| 0.8442 | $606,434 | $180,363 | 19.34% | 7.77% | 0.1834 |

The original Week 8 LightGBM result was R2 0.7650, RMSE $744,924, MAE $236,661,
and MdAPE 11.10%. The row counts differ by five because the new pipeline loads
and validates the raw monthly files directly, so the comparison is informative
but not a perfectly identical test cohort.
