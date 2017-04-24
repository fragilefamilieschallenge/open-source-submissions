import pandas as pd
import numpy as np
import getopt
import sys, time


def fillMissing(inputcsv, outputcsv):

    #handpick some columns that we will drop
    cols_to_kill = ['hv4j5a_ot']
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)

    #Kill handpicked columns

    df.drop(cols_to_kill, inplace=True, axis=1)
    #Threshold columns to remove too many NAs
    value = 2000
    cols_to_drop =  df.isnull().sum() > value
    df.drop(cols_to_drop.loc[cols_to_drop].index, axis=1, inplace=True)

    # replace NA's with mode
    df = df.fillna(df.mode().iloc[0])

    # if still NA, delete
    df.dropna(axis=1, inplace=True)
    #df = df.fillna(value=1)
    #Drop columns with no variance (number unique values is 1)
    # from http://stackoverflow.com/questions/39658574/how-to-drop-columns-which-have-same-values-in-all-rows-via-pandas-or-spark-dataf
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)

    #Finally, drop all columns that aren't numeric



    # replace negative values with 1
    num = df._get_numeric_data()
    num[num < 0] = 1
    # write filled outputcsv
    #df.to_csv(outputcsv, index=False)
    num.to_csv(outputcsv, index=False)

# # Usage:
# fillMissing('background.csv', 'output.csv')
# filleddf = pd.read_csv('output.csv', low_memory=False)
#

def main(argv):
    start_time = time.time()

    path = './'
    outputf = 'output.csv'
    inputf = 'background.csv'
    usage_message = 'Usage: \n python MissingDataScript.py -p <path> -i <inputfile> -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv, "p:i:o:",
                                   ["path=", "ofile="])
    except getopt.GetoptError:
        print usage_message
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print usage_message
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg
        elif opt in ("-i", "--ifile"):
            inputf = arg
        elif opt in ("-o", "--ofile"):
            outputf = arg


    print 'Path:', path
    print 'Imputing values.'

    #Fill Missing Values
    infile = open(path + "/" + inputf, 'r')
    outfile = open(path + "/" + "imputed_" + outputf, 'w')
    fillMissing(infile, outfile)

    print 'Output files:', path + "/" + outputf + "*"

    # Runtime
    print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
    main(sys.argv[1:])
