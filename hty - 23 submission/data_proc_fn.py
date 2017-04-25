import math
import numpy as np

def nan_index(gpa):
    # remove NaN from vector
    gpa_no_NA = []
    for i in range (len(gpa)):
        if math.isnan(gpa[i]):
            gpa_no_NA.append(i)
    return gpa_no_NA

def detect_non_float_dataFrame(df):
    # detect non-float values:
    non_float_ind = []
    for i in range (np.shape(df)[1]):
        try:
            df[df.columns[i]].astype(np.float)
        except ValueError:
    #         print "error num:", i
            non_float_ind.append(i)
    return non_float_ind


def detect_non_number_df(df):
    # record columns with non-a-numbers
    non_number_items=[]
    for i in range(np.shape(df)[1]):
        if np.isfinite(df[df.columns[i]]).all() == False:
            non_number_items.append(i)
    return non_number_items


# get the indices in the X matrix that matchese with idy
# consider improving the algorithm. now its too slow.
# def get_matched_ind_for_X (idy_no_NaN_gpa):
#     id_index_df = []
#     for j in range (len(idy_no_NaN_gpa)):
#         for i in range (len(df['challengeID'])):
#             if df['challengeID'][i] == idy_no_NaN_gpa.iloc[j]:
#                 # print 'i: ', i
#                 id_index_df.append(i)
#                 break
#     return id_index_df



def get_clean_X (df):
    # find out columns containing non-float entries in the original dataframe
    non_float_ind = detect_non_float_dataFrame(df)

    # delete non-float values
    df_all_float = df.drop(df.columns[non_float_ind], 1)

    non_number_ind = detect_non_number_df(df_all_float)

    df_clean = df_all_float.drop(df_all_float.columns[non_number_ind], 1)
    return df_clean, non_float_ind, non_number_ind


def X_with_right_col(df_clean, idy, gpa_no_NA):
    # challengeID without NA corresponding to gpa, grit and material hardship
    idy_no_NaN_gpa = idy.drop(idy.index[gpa_no_NA])

    id_index_df = []
    for j in range (len(idy_no_NaN_gpa)):
        for i in range (len(df_clean['challengeID'])):
            if df_clean['challengeID'][i] == idy_no_NaN_gpa.iloc[j]:
                # print 'i: ', i
                id_index_df.append(i)
                break

    # id_index_df = get_matched_ind_for_X(idy_no_NaN_gpa)

    # this should be the final step. the X matrix is now ready to use for fitting.
    X_gpa = df_clean.loc[id_index_df]

    return X_gpa
