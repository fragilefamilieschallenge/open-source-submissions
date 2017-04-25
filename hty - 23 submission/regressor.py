
def processing (enet_gpa, df_clean, X_gpa, Y_gpa):
    enet_gpa.fit(X_gpa, Y_gpa)

    pred_gpa = enet_gpa.predict(df_clean)

    # sort the challengeID column in background.csv
    tp = df_clean['challengeID'].sort_values()

    # get the corresponding indices
    tp_ind = tp.index.values

    # this is the final array that can be outputted to predict.csv
    gpa_out = pred_gpa[tp_ind]

    return gpa_out
