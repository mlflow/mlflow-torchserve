import pandas as pd


def total(dataframe):
    if dataframe.Predicted == dataframe.Actual:
        return 1
    else:
        return 0


def scorer(df):
    dates_tested = pd.DataFrame(df.index)
    df["Score"] = df.apply(total, axis=1)
    avg_test_acc = df["Score"].sum() / len(df)
    tested_date = dates_tested["dates"].max()
    wrong_prediction = 1 - avg_test_acc
    test_results = str(tested_date) + "~" + str(avg_test_acc) + "~" + str(wrong_prediction)
    return test_results
