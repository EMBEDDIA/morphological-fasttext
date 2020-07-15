import argparse
import os


def create_output_string(dictionary, headers, num_files, digits=4):
    '''
    Partially adapted sklearn.metrics.classification_report function for correct output
    :param dict:
    :return:
    '''
    target_names = dictionary.keys()
    longest_last_line_heading = 'weighted avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), digits)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
    for row_name in target_names:
        row = [row_name] + [round(dictionary[row_name][metric] / num_files, 4) for metric in headers]
        if row_name == 'avg / total':
            report += '\n'
        report += row_fmt.format(*row, width=width, digits=digits)

    return report

def read_results(path):
    results = {}
    num_files = 0
    for filename in os.listdir(path):
        # if filename begins with results save results into final_results
        if filename != 'final_results.txt':
            num_files += 1
            with open(os.path.join(path, filename), "r") as file:
                metrics = []
                # read header
                for first_line in file:
                    split_fl = first_line.split()
                    for metric_name in split_fl:
                        metrics.append(metric_name)
                    break
                for line in file:
                    if line =='\n':
                        continue
                    split_line = line.split() if line.split()[0] != 'avg' else ['avg / total'] + line.split()[3:]
                    for i, metric in enumerate(metrics):
                        if split_line[0] not in results:
                            results[split_line[0]] = {}
                        results[split_line[0]][metric] = results[split_line[0]].get(metric, 0) + float(split_line[i + 1])
                    pass

    output_string = create_output_string(results, metrics, num_files)

    with open(os.path.join(path, 'final_results.txt'), "w") as file:
        file.write(output_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()
    read_results(args.results_path)
