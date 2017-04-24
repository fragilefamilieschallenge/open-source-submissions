# Map codes to human readable strings
import os
import re
import glob
import pickle

def get_stuff_from_content(content):
    data = {}
    
    # read all lines into a list, skipping the first '-----' line
    lines = content.strip().split('\r\n')[1:]
    
    check_next_line = False
    for l in lines:
        
        # Every time you encounter a '-----' line, flip the switch. If that
        # switch is true, we'll load the next line into a dict. If the switch
        # was flipped to False, it means we encountered the end of the header.
        if re.match('^-+$', l):
            check_next_line = not check_next_line
        elif check_next_line:
            # Replace at least 4 spaces (or more) with "|", then split on
            # that value, loading the key on the left and the value on the
            # right.
            key, val = re.sub('\s\s\s\s+', '|', l).split('|')
            #print [key, val.replace('-->', '')]
            data[key] = val.replace('-->', '')

    return data


data = {}

os.chdir('codebooks')
for file in glob.glob("*.txt"):
    with open(file, 'r') as f:
        print 'Processing %s' % file
        content = f.read()
        data.update(get_stuff_from_content(content))

pickle.dump(data, open("Codebook_Mappings.pck", "wb" ))
