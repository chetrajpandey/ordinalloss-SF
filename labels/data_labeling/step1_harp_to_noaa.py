import pandas as pd

# Your imports

def map_goes_NOAA_ARR_with_HARPNUM():
    goes = pd.read_csv('goes_flares_integrated.csv')
    harp = pd.read_csv('allharp2noaa.csv')

    harp['NOAA_ARS'] = harp['NOAA_ARS'].str.strip('""').str.split(',')
    harpdf = harp.explode('NOAA_ARS').drop_duplicates()

    goesdf = goes[goes['noaa_active_region'].notna()]
    harpdf = harpdf[harpdf['NOAA_ARS'].notna()]

    goesdf.loc[:, 'noaa_active_region'] = goesdf['noaa_active_region'].astype(int)

    harpdf['NOAA_ARS'] = harpdf['NOAA_ARS'].astype(int)
    harpdf['DEF_HARPNUM'] = harpdf['DEF_HARPNUM'].astype(int)

    df3 = goes.merge(harpdf, left_on='noaa_active_region', right_on='NOAA_ARS', how='inner')

    final_df = df3.sort_values(by=['DEF_HARPNUM'])
    final_df.loc[:,'DEF_HARPNUM'] = final_df['DEF_HARPNUM'].astype(int)
    final_df.drop_duplicates(inplace=True)


    final_df.to_csv('intermediates/harp2noaaAR_mappedGOES.csv', index=False, columns=['DEF_HARPNUM', 'NOAA_ARS', 'start_time', 'goes_class'])

def main():
    map_goes_NOAA_ARR_with_HARPNUM()

if __name__ == "__main__":
    main()
