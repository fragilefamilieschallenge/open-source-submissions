#!/usr/bin/python


import argparse




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--continuous", help="CSV from which to extract continous predictions", required=True)
    parser.add_argument("-b", "--binary", help="CSV from which to extract binary predictions", required=True)
    parser.add_argument("-o", "--output", help="Path to output file", required=True)

    options = parser.parse_args()

    run(options)


def run(options):
    gpas = {}
    grits = {}
    hards = {}
    evicts = {}
    layoffs = {}
    jobtrains = {}

    c_cids = []
    cf = open(options.continuous)
    cf.readline() # Remove header
    for l in cf:
        cols = l.strip().split(",")
        cid = int(cols[0])
        gpas[cid] = cols[1]
        grits[cid] = cols[2]
        hards[cid] = cols[3]
        c_cids.append(int(cid))

    b_cids = []
    bf = open(options.binary)
    bf.readline() # Remove header
    for l in bf:
        cols = l.strip().split(",")
        cid = int(cols[0])
        evicts[cid] = cols[4]
        layoffs[cid] = cols[5]
        jobtrains[cid] = cols[6]
        b_cids.append(int(cid))

    if sorted(b_cids) != sorted(c_cids):
        print 'mismatched challengeIDs'
        return

    ofile = open(options.output, 'w')

    ofile.write('"challengeID","gpa","grit","materialHardship","eviction","layoff","jobTraining"\n')

    for cid in sorted(c_cids):
        ofile.write(str(cid))
        ofile.write(",")
        ofile.write(str(gpas[cid]))
        ofile.write(",")
        ofile.write(str(grits[cid]))
        ofile.write(",")
        ofile.write(str(hards[cid]))
        ofile.write(",")
        ofile.write(str(evicts[cid]))
        ofile.write(",")
        ofile.write(str(layoffs[cid]))
        ofile.write(",")
        ofile.write(str(jobtrains[cid]))
        ofile.write("\n")
        ofile.flush()

    ofile.close()


if __name__ == "__main__":
    main()
