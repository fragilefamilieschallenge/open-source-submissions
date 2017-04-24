"""Map feature codes to human readable strings, e.g.:

    k5conf2 -> CONF2. Child has seen his/her biofather in the last year.
"""

import glob
import pickle


def process_line(line):
    """Split a single line into its feature code and description.
    """
    line = line.rstrip()  # Remove trailing '\r\n'.
    parts = line.split(None, 1)  # Split on first whitespace.
    if len(parts) == 0:
        return None, None
    code, description = parts
    return code, description


def process_file(f):
    """Process file into dictionary mapping feature codes to descriptions.
    """
    data = {}
    fl = False  # Save next line flag.
    for line in f:
        if fl:
            code, description = process_line(line)
            if code and description:
                data[code] = description
            fl = False
        if line.startswith('----------'):
            fl = True
    return data


def save_data(data):
    """Save data as pickle file and plain text file (for quick searching).
    """
    with open('data/feature_codes_to_names.pck', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/feature_codes_to_names.txt', 'w+') as f:
        for k, v in data.items():
            line = '%s\t%s\n' % (k, v)
            f.write(line)


def process_feature_codes_and_pickle():
    data = {}
    for file_ in glob.glob('data/codebooks/*.txt'):
        with open(file_, 'r') as f:
            print('Processing %s' % file_)
            data.update(process_file(f))
    save_data(data)


if __name__ == '__main__':
    process_feature_codes_and_pickle()
