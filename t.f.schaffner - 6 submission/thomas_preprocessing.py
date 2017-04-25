#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd


# From ``MissingDataScript.py``, provided as a resource for Princeton cos424, Spring 2017
def parse_data(inputcsv):
    df = pd.read_csv(inputcsv, low_memory=False)
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    return df


# From ``MissingDataScript.py``, provided as a resource for Princeton cos424, Spring 2017
def fillMissing(df):
    # replace NA's with mode
    # df = df.fillna(df.mode().iloc[0])
    # if still NA, replace with 1
    # df = df.fillna(value=1)
    # replace negative values with 1
    num = df._get_numeric_data()

    coltypes = num.dtypes

    # There should only be float64 and int64 types after _get_numeric_data()
    floatcols = (coltypes == np.float64)
    intcols = (coltypes == np.int64)

    means = num.mean(0)
    mean_num = pd.DataFrame([means for i in xrange(len(num.index))])
    modes = num.mode(0)
    mode_num = pd.DataFrame([modes for i in xrange(len(num.index))])

    num[num.loc[:,num.dtypes==np.float64].lt(0)] = mean_num
    num[num.loc[:,num.dtypes==np.int64].lt(0)] = mode_num

    num = num._get_numeric_data()

    num = num.fillna(value=1)

    return num


def column_na_portions(data):
    return (data.isnull().sum() / float(len(data.index)))


def column_ne_portions(data):
    return (data.lt(0).sum() / float(len(data.index)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--background", help="Path to background.csv", required=True)
    parser.add_argument("-o", "--output_filename", help="Path to output file", required=True)
    parser.add_argument("-v", "--verbose", help="[OPTIONAL] Verbosity (flag)", action="store_true")
    parser.add_argument("-nac", "--na_cutoff", help="[OPTIONAL] Remove columns that have more than cutoff portion (float) null values.", default=0, type=float)
    parser.add_argument("-nec", "--ne_cutoff", help="[OPTIONAL] Remove columns that have more than cutoff portion (float) negative values. Applied after na_cutoff, if specified.", default=0, type=float)


    options = parser.parse_args()

    run(options)


def run(options):
    if options.verbose:
        print "Loading data"
    # filled = fillMissing(options.background)
    data = parse_data(options.background)

    if options.verbose:
        print "Calculating portions of entries that are negative"
    ne_portions = column_ne_portions(data)

    if options.verbose:
        print "Calculating portions of entries that are NA"
    na_portions = column_na_portions(data)

    if options.verbose:
        print "Filling NA values"
    data = fillMissing(data)

    if options.verbose:
        print "Removing columns with NA portions that are too high"
    data = data.loc[:,~(na_portions.gt(float(options.na_cutoff)))]

    if options.verbose:
        print "Removing columns with negative portions that are too high"
    data = data.loc[:,~(ne_portions.gt(float(options.ne_cutoff)))]

    if options.verbose:
        print "Writing preprocessed data to " + str(options.output_filename)
    data.to_csv(options.output_filename, index=False)

    return data


if __name__ == "__main__":
    main()

