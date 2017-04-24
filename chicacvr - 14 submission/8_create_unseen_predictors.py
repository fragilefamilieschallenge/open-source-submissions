import csv


background_file = open('data/background.csv', 'r')
training_file = open('data/train.csv', 'r')
unseen_predictors_file = open('data/unseen_predictors.csv', 'w')


# Get all the Challenge IDs from the training set, excluding the ones with NA GPA
print 'Analyzing the training file'
reader = csv.reader(training_file)
next(reader)  # skip the header row
challenge_ids = set()
for row in reader:
    gpa = row[1]
    if gpa.strip().lower() == 'na':
        continue

    gpa = float(gpa)
    if 4 >= gpa >= 1:
        challenge_ids.add(row[0])

# Write rows from background.csv whose Challenge ID was *not* in the training
# set.
print 'Writing the unseen predictors file'
reader = csv.reader(background_file)
writer = csv.writer(unseen_predictors_file)
writer.writerow(next(reader)[1:])  # write the header row (minus the idnum column)
valid_cnt = 0
for row in reader:
    challenge_id = row[-1]
    if challenge_id not in challenge_ids:
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

        writer.writerow(valid_row[1:])  # remove idnum

        if not valid_cnt % 100:
            print valid_cnt

print 'Done! Wrote %d unseen rows' % valid_cnt
