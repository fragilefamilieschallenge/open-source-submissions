with open('script_4_final_output.txt', 'r') as f:
    data = f.read()

name = None

results = []
name = None
for line in data.split('\n'):
    if name is None:
        name = line
    elif line.startswith('Mean'):
        score_raw = line
        score = float(line.replace('Mean MSE score: ', ''))
    elif line.startswith('Performance'):
        perf = line
        results.append((score, name, score_raw, perf))
        name = None

for cnt, result in enumerate(sorted(results, reverse=True)):
    print '%d. %s' % (cnt+1, result[1])
    print result[2]
    print result[3]

