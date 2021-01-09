import argparse
import os


def create_output_string(dictionary, headers, num_files, digits=4):
    '''
    Partially adapted sklearn.metrics.classification_report function for correct output
    :param dict:
    :return:
    '''
    target_names = dictionary.keys()
    longest_last_line_heading = 'micro avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
    for row_name in target_names:
        row = [row_name] + [round(dictionary[row_name][metric] / dictionary[row_name]['support'], 4) for metric in headers[:-1]] + [dictionary[row_name]['support']]
        if row_name == 'micro avg':
            report += '\n'
        report += row_fmt.format(*row, width=width, digits=digits)

    return report

def read_results(path):
    results = {}
    num_files = 0
    f1_scores = []
    for filename in sorted(os.listdir(path), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isnumeric() else -1):
        # if filename begins with results save results into final_results
        if filename != 'final_results.txt' and filename != 'wilcoxon_test.txt':
            num_files += 1
            with open(os.path.join(path, filename), "r") as file:
                metrics = []
                # read header
                for first_line in file:
                    split_fl = first_line.split()
                    for metric_name in split_fl:
                        metrics.append(metric_name)
                    break
                f1_score_index = metrics.index('f1-score')
                for line in file:
                    if line =='\n':
                        continue
                    split_line = line.split() if line.split()[1] != 'avg' else [' '.join(line.split()[:2])] + line.split()[2:]
                    if split_line[0] not in results:
                        results[split_line[0]] = {}
                    support = int(split_line[-1])
                    results[split_line[0]][metrics[-1]] = results[split_line[0]].get(metrics[-1], 0) + support
                    # last metric is support
                    for i, metric in enumerate(metrics[:-1]):
                        results[split_line[0]][metric] = results[split_line[0]].get(metric, 0) + float(split_line[i + 1]) * support
                    final_f1_score = float(split_line[f1_score_index + 1])
            f1_scores.append(str(final_f1_score))
    output_string = create_output_string(results, metrics, num_files)

    with open(os.path.join(path, 'wilcoxon_test.txt'), "w") as file:
        file.write('\n'.join(f1_scores))

    with open(os.path.join(path, 'final_results.txt'), "w") as file:
        file.write(output_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()
    read_results(args.results_path)
