# vnncomp2021_results

https://github.com/stanleybak/vnncomp2021_results.git

results for vnncomp 2021. The csv files for all tools are in results_csv. The scores are computed using process_results.py, with stdout redirected to the output_*.txt files.

Summary scores are near the end of the file. You can check a specific benchmark by looking in the file. For example, to see network 4-2 property 2 of acasxu, I can look for the file for the following part:

```
Row: ['ACASXU_run2a_4_2_batch_2000-prop_2', '-', '6.4 (h)', '10.5 (h)', 'timeout', '41.1 (h)', 'timeout', 'timeout', '64.8 (h)', '62.5 (h)', 'timeout', 'timeout', 'timeout', '-']
73: nnv score: 0
73: nnenum score: 12
73: venus2 score: 11
73: NN-R score: 0
73: VeriNet score: 10
73: DNNF score: 0
73: Debona score: 0
73: a-b-CROWN score: 10
73: oval score: 10
73: Marabou score: 0
73: ERAN score: 0
73: NV.jl score: 0
73: randgen score: 0
```

The tools are listed in order, and row is the times and result for each tool. So nnenum should be holds with a time of 6.4 (after subtracting overhead). Sure enough, if you look in results_csv/nnenum.csv at the corresponding line you see:

`
acasxu,./benchmarks/acasxu/ACASXU_run2a_4_2_batch_2000.onnx,./benchmarks/acasxu/prop_2.vnnlib,.007249637,holds,7.374020909
`

The runtime was 7.374020909, which after subtracting the overhead of 1.0 secs you get 6.37 which roudns to 6.4.

The scores are also listed for each tool. Since nneum was the fastest, it got 12 points (10 for corrext + 2 for time bouns as fastest. Venus2, at 10.5 seconds, was the second fastest so it get 11 points. None of the remaining tools were within two seconds, so they all got 10 points.

You can adjust some parameters in the scoring defined in main(). Specifically, you can change how incorrect results are judged by changing `resolve_conflicts` in the code:

```
# how to resolve conflicts (some tools output holds others output violated)
# "voting": majority rules, tie = ighore
# "odd_one_out": only if single tool has mismatch, assume it's wrong
# "ignore": ignore all conflicts
resolve_conflicts = "voting"
```

### Changes from VNNCOMP Presentation Results
Based on additional feedback from tool authors, we made the following changes after the VNNCOMP presentation. All changes to scoring can be adjusted through flags in the `main()` function.
    
1. ERAN included optimized overhead for benchamarks that didn't use the GPU. This optimization had the side effect of reducing their overhead measurement for all benchmarks, inflating their times on GPU benchmarks by a few seconds (and thus reducing their score).

You can change how overhead is measured by modifying the `single_overhead` flag. If True, then a uniform overhead is used for all measurements per tool (original). If False, ERAN will use a separate overhead measurement for the acasxu and eran benchmarks.

2. Neural Network Reach renamed to RPM. Also RPM's mnistfc results have been removed due to a VNNLIB parsing bug on that benchmark found after the competition. This can be disabled by commenting out the line: `skip_benchmarks['RPM'] = ['minstfc']`
