import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, f_oneway
import sys
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Function to check distribution overlap using 95% data coverage
def check_icmop_distribution_overlap(best_alg, alg2):
    best_95_data = np.mean(best_alg) + 2 * np.std(best_alg)
    alg2_95_data = np.mean(alg2) - 2 * np.std(alg2)

    best_max_value = np.max(best_alg)
    alg2_min_value = np.min(alg2)

    # Determine significant value boundaries
    best_value = max(best_max_value, best_95_data)
    alg2_value = min(alg2_min_value, alg2_95_data)

    # Check for overlap
    return alg2_value <= best_value

# Function to find the best algorithm based on the lowest median value
def find_best_alg(data):
    algs = data['Group'].unique()
    best_alg = None
    best_val = sys.maxsize

    # Iterate through each algorithm to determine the best
    for alg in algs:
        median_value = np.median(data[data['Group'] == alg]['Value'])
        if median_value < best_val:
            best_val = median_value
            best_alg = alg

    return best_alg

# Function to perform ANOVA and Tukey's HSD test
def check_algorithms_anova(data_orig, best_algs, best_alg):
    data = [data_orig[data_orig['Group'] == alg]['Value'].tolist() for alg in best_algs]
    groups = [data_orig[data_orig['Group'] == alg]['Group'].tolist() for alg in best_algs]

    # Perform ANOVA test
    anova_result = f_oneway(*data)
    if anova_result.pvalue < 0.002:
        # Tukey's HSD test
        values = [val for sublist in data for val in sublist]
        group_labels = [val for sublist in groups for val in sublist]
        tukey_result = pairwise_tukeyhsd(endog=values, groups=group_labels, alpha=0.0035)
        tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])

        # Filter relevant comparisons for the best algorithm
        tukey_df = tukey_df[(tukey_df['group1'] == best_alg) | (tukey_df['group2'] == best_alg)]
        tukey_df = tukey_df[tukey_df['reject'] == False]
        best_algs = list(set(tukey_df['group1'].tolist() + tukey_df['group2'].tolist()))

    return best_algs if best_algs else [best_alg]

# Main script
def main():
    dim = 5
    cuts = ['_1', '_2', '_3', '_4', '_5']
    algs = ['NSGA3', 'MOEAD', 'CTAEA', 'NSGA2', 'AGE', 'SPEA2', 'GDE3', 'NSDE', 'NSDER']
    cut = cuts[0]

    # Load data
    df = pd.read_csv(f'data/i_cmop_values_{dim}d.csv')

    best_algs_overall = []

    # Iterate over each problem
    for problem in df['problem'].unique():
        print(f'Processing problem: {problem}')
        problem_data = df[df['problem'] == problem]

        # Construct data for analysis
        data = {'Group': [], 'Value': []}
        for alg in algs:
            try:
                values = [float(x) for x in problem_data[alg + cut].values[0].replace('\n', '').strip('[]').split() if x]
                data['Group'].extend([alg + cut] * len(values))
                data['Value'].extend(values)
            except Exception as e:
                print(f'Error processing algorithm {alg}: {e}')

        data_df = pd.DataFrame(data)
        best_alg = find_best_alg(data_df)
        problem_best_algs = [best_alg]

        # Check overlap and perform ANOVA analysis
        for alg in algs:
            alg_cut = alg + cut
            if alg_cut == best_alg:
                continue
            overlap = check_icmop_distribution_overlap(data_df[data_df['Group'] == best_alg]['Value'].tolist(),
                                                       data_df[data_df['Group'] == alg_cut]['Value'].tolist())
            if overlap:
                problem_best_algs.append(alg_cut)

        if len(problem_best_algs) > 1:
            problem_best_algs = check_algorithms_anova(data_df, problem_best_algs, best_alg)

        best_algs_overall.append(problem_best_algs)

if __name__ == '__main__':
    main()
