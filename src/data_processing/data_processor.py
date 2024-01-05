import pandas as pd

def session_data_aggregation(df_dict, file_name):
    df_dict_res = {}
    for session_df_key in df_dict.keys():
        session_df = df_dict[session_df_key]
        # session_df = session_df.iloc[:, 2:session_df.columns.get_loc("interventionDuration")].drop(["breathingGuide"], axis = 1).dropna(how="all")
        session_df = session_df.iloc[:121, 2:session_df.columns.get_loc("gamma")].dropna(how="all")
        session_df_mean = pd.DataFrame(session_df.mean()).T
        session_df_mean.columns = [f"{col}_mean" for col in session_df_mean.columns]
        session_df_std = pd.DataFrame(session_df.std()).T
        session_df_std.columns = [f"{col}_std" for col in session_df_std.columns]
        session_df_median = pd.DataFrame(session_df.
                                         median()).T
        session_df_median.columns = [f"{col}_median" for col in session_df_median.columns]
        # display(session_df_median)
        session_df_quantile10 = pd.DataFrame(session_df.quantile(0.1)).T.reset_index(drop=True)
        session_df_quantile10.columns = [f"{col}_quantile10" for col in session_df_quantile10.columns]
        # display(session_df_quantile10)
        session_df_quantile90 = pd.DataFrame(session_df.quantile(0.9)).T.reset_index(drop=True)
        session_df_quantile90.columns = [f"{col}_quantile90" for col in session_df_quantile90.columns]
        
        session_df_conc = pd.concat([session_df_mean, session_df_std, 
                                     session_df_median, 
                                     session_df_quantile10, 
                                     session_df_quantile90
                                     ], axis = 1)

        session_df_conc.columns = [f"{col}_{session_df_key}" for col in session_df_conc.columns]
        df_dict_res[session_df_key] = session_df_conc
    cdf = pd.concat(df_dict_res.values(), axis=1)  
    # cdf["origin"] = file_name
    return cdf


def session_data_aggregation_ratios_df(df_dict, file_name):
    df_dict_res = {}
    for session_df_key in df_dict.keys():
        session_df = df_dict[session_df_key]
        
        session_df_ratios = pd.DataFrame({"beta/theta": [(session_df["beta"]/session_df["theta"]).mean()],
            "gamma/theta" : [(session_df["gamma"]/session_df["theta"]).mean()],
            "smr/theta": [(session_df["smr"]/session_df["theta"]).mean()],
            "highAlpha/theta": [(session_df["highAlpha"]/session_df["theta"]).mean()],
            "beta/delta" : [(session_df["beta"]/session_df["delta"]).mean()],
            "theta/delta": [(session_df["theta"]/session_df["delta"]).mean()],
            "beta/lowAlpha": [(session_df["beta"]/session_df["lowAlpha"]).mean()]
        })
        
        session_df_ratios.columns = [f"{col}_{session_df_key}" for col in session_df_ratios.columns]
        # session_df_conc.columns = [f"{col}" for col in session_df_conc.columns]
        df_dict_res[session_df_key] = session_df_ratios
    cdf = pd.concat(df_dict_res.values(), axis=1)
    # cdf["origin"] = file
    return cdf


def session_data_NBRawBFV_df(df_dict, file_name):
    df_dict_res = {}
    for session_df_key in df_dict.keys():
        session_df = df_dict[session_df_key]
        session_df_ratios = pd.DataFrame({"NBRawBFV": [session_df["NBRawBFV"].mean()]})
        session_df_ratios.columns = [f"{col}_{session_df_key}" for col in session_df_ratios.columns]
        # session_df_conc.columns = [f"{col}" for col in session_df_conc.columns]
        df_dict_res[session_df_key] = session_df_ratios
    cdf = pd.concat(df_dict_res.values(), axis=1)
    # cdf["origin"] = file
    return cdf

