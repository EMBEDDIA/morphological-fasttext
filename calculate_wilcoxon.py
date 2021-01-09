import argparse
import scipy.stats

def read_file(path):
    with open(path, "r") as file:
        predictions = []
        for line in file:
            predictions.append(float(line.replace('\n', '')))
    return predictions


def write_file(path, results):
    with open(path, "w") as file:
        for result in results:
            file.write(str(result) + '\n')


def main(args):
    base_pred = read_file(args.input1)
    othr_pred = read_file(args.input2)
    wilcoxon_result = scipy.stats.wilcoxon(base_pred, othr_pred)
    # wilcoxon_result = scipy.stats.mannwhitneyu(base_pred, othr_pred)
    write_file(args.output, [wilcoxon_result.pvalue, wilcoxon_result.statistic])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=str, required=True, help='Path to base language.')
    parser.add_argument("--input2", type=str, required=True, help='Path to comparing language.')
    parser.add_argument("--output", type=str, required=True, help='Path to results.')
    args = parser.parse_args()
    main(args)