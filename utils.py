import pandas as pd


def create_dlc_points_2d_file(dlc_dataframe_filepaths):
    dfs = []
    for path in dlc_dataframe_filepaths:
        dlc_df = pd.read_hdf(path)
        dlc_df=dlc_df.droplevel([0], axis=1).swaplevel(0,1,axis=1).T.unstack().T.reset_index().rename({'level_0':'frame'}, axis=1)
        dlc_df.columns.name = ''
        dfs.append(dlc_df)
    #create new dataframe
    dlc_df = pd.DataFrame(columns=['frame', 'camera', 'label', 'x', 'y', 'likelihood'])
    for i, df in enumerate(dfs):
        df["camera"] = i
        df.rename(columns={"bodyparts":"label"}, inplace=True)
        dlc_df = pd.concat([dlc_df, df], sort=True, ignore_index=True)

    dlc_df = dlc_df[['frame', 'camera', 'label', 'x', 'y', 'likelihood']]
    return dlc_df
