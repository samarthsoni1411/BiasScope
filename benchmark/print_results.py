import pandas as pd, sys, os
sys.path.insert(0,'.')
from benchmark.run_experiments import generate_latex_table

df = pd.read_csv('benchmark/results/results.csv')
print('=== REAL EXPERIMENTAL RESULTS ===')
cols = ['dataset','acc_before','acc_after','ifs_before','ifs_after',
        'dp_diff_before','dp_diff_after','di_before','di_after',
        'ifs_ci_before_lower','ifs_ci_before_upper','ifs_ci_after_lower','ifs_ci_after_upper']
print(df[cols].to_string(index=False))

tex = generate_latex_table(df)
with open('benchmark/results/table_main.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
print('\nLaTeX table saved to benchmark/results/table_main.tex')
print('\n=== IFS Summary ===')
for _, row in df.iterrows():
    gain = row['ifs_after'] - row['ifs_before']
    ci_b = f"[{row['ifs_ci_before_lower']:.3f},{row['ifs_ci_before_upper']:.3f}]"
    ci_a = f"[{row['ifs_ci_after_lower']:.3f},{row['ifs_ci_after_upper']:.3f}]"
    print(f"  {row['dataset']:20s}: IFS {row['ifs_before']:.4f}{ci_b} -> {row['ifs_after']:.4f}{ci_a} ({gain:+.4f})")

print('\nAll done!')
