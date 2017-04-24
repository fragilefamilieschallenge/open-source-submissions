import csv


header_row = [
    'Imputation', 'Scaling', 'Feature Selection', 'Model',
    'Mean MSE', 'Training Time (s)'
]
csv_writer = csv.writer(open('all_results.csv', 'wb'))
csv_writer.writerow(header_row)

with open('script_4_final_output.txt', 'r') as f:
    data = f.read()

csv_data = None

for line in data.split('\n'):
    if line.startswith('Imputer'):
        csv_data = line.split(',')
    elif line.startswith('Mean'):
        score = float(line.replace('Mean MSE score: ', ''))
        csv_data.append(score)
    elif line.startswith('Performance'):
        perf = line.replace('Performance: ', '')
        perf = float(perf.replace('s', ''))
        csv_data.append(perf)
        csv_writer.writerow(csv_data)
        csv_data = None

