# Section 4.5 New Attention Ablation Figure Notes

Final model: physical cubic degradation base + residual temporal network + degradation-state-guided curve attention v1 + life-cycle adaptive MMD.

## Figure contract

- Fig. 4.5a: effect-size map for MMD controls, attention-mechanism controls, and structural controls.
- Fig. 4.5b: radar view of stage-wise RMSE across early, middle, and late degradation stages.
- Fig. 4.5c: violin distribution of Batch10 cell-wise RMSE across seeds, with seed-level pooled RMSE overlaid.

## Key values

- Proposed three-seed mean RMSE: 0.991% SOH.
- Proposed three-seed mean MAE: 0.619% SOH.

## Writing guardrail

The figures support a mechanism-oriented ablation narrative. They should not be used to claim that every expanded curve token monotonically improves every metric. Voltage-only curve tokens are retained in the source data as a diagnostic control.