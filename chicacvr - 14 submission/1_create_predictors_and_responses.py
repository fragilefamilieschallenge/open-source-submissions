import csv


training_file = open('data/train.csv', 'r')
background_file = open('data/background.csv', 'r')
predictors_file = open('data/predictors.csv', 'w')
responses_file = open('data/responses.csv', 'w')

# Get all the Challenge IDs from the training set, excluding the ones with NA GPA
print 'Analyzing the training file'
reader = csv.reader(training_file)
writer = csv.writer(responses_file)
writer.writerow(['challengeID', 'gpa'])
next(reader)  # skip the header row
challenge_ids = set()
new_rows = []
for row in reader:
    gpa = row[1]
    if gpa.strip().lower() == 'na':
        continue

    gpa = float(gpa)
    if 4 >= gpa >= 1:
        challenge_ids.add(row[0])
        new_rows.append([row[0], row[1]])

# Write rows sorted by Challenge ID
new_rows.sort(key=lambda row: int(row[0]))
for new_row in new_rows:
    writer.writerow(new_row)

print 'Found %d valid challenge IDs' % len(challenge_ids)

print 'Writing matching predictors'
reader = csv.reader(background_file)
writer = csv.writer(predictors_file)
writer.writerow(next(reader)[1:-1])  # write the header row (minus the idnum and challengeID columns)
valid_cnt = 0
new_rows = []
for row in reader:
    challenge_id = row[-1]
    if challenge_id in challenge_ids:
        valid_cnt += 1

        valid_row = []
        for cell in row:
            try:
                value = float(cell)
                if value < 0:
                    raise ValueError
                val = cell
            except (TypeError, ValueError):
                val = 'NaN'

            valid_row.append(val)

        new_rows.append(valid_row)

        if not valid_cnt % 100:
            print valid_cnt

# Write rows sorted by Challenge ID (removing idnum and challengeID columns)
new_rows.sort(key=lambda row: int(row[-1]))
for new_row in new_rows:
    writer.writerow(new_row[1:-1])

print 'Done! Wrote %d matching predictors' % valid_cnt
