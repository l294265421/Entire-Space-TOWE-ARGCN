def read_all_lines(file_path, encoding='utf-8'):
    lines = []
    with open(file_path, encoding=encoding) as in_file:
        for line in in_file:
            lines.append(line.strip())
    return lines


def parse_a_metric_from_line(line: str, metric_str: str):
    """

    """
    start_index = line.index(metric_str) + len(metric_str) + 1
    end_index = start_index + 6
    result = line[start_index: end_index]
    return result


def parse_metric_line(line: str):
    """

    """
    metric_strs = ['precision:', 'recall:', 'f1:']
    result = [parse_a_metric_from_line(line, metric_str) for metric_str in metric_strs]
    return result


log_filepath = '/Users/yuncongli/PycharmProjects/towe-eacl-private/worker_0-stdout'
lines = read_all_lines(log_filepath)
target_lines = []
for i, line in enumerate(lines):
    if 'Total Metric' in line:
        target_lines.append(lines[i + 1])

runs = []
for line in target_lines:
    run = parse_metric_line(line)
    runs.append(run)

prf = [','.join([e[i] for e in runs]) for i in range(3)]
print('\t'.join(prf))

