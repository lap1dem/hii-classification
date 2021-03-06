import pandas as pd

def load_catalogue():
    # Classified HII regions
    hii_data = pd.read_csv('data/HII/hii_table_with_id.csv').dropna().set_index('objid')
    hii_data = hii_data[['u', 'g', 'r', 'ai', 'z', 'redsh', 'ra', 'dec']].rename(
        columns = {'ai': 'i', 'redsh': 'redshift'})
    hii_data = hii_data[
        (hii_data['u'] > 0) &
        (hii_data['g'] > 0) &
        (hii_data['r'] > 0) &
        (hii_data['i'] > 0) &
        (hii_data['z'] > 0)
        ]
    hii_data['class'] = 'HII'

    # SDSS classified objects
    sdss_data = pd.read_csv('data/SDSS/sdss-all-objects.csv')
    sdss_data.rename(columns = {'bestObjID': 'objid'}, inplace=True)
    sdss_data.set_index('objid', inplace=True)
    sdss_data = sdss_data[
        (sdss_data['zWarning'] == 0) &
        (sdss_data['redshift'] > 0) &
        (sdss_data['u'] > 0) &
        (sdss_data['g'] > 0) &
        (sdss_data['r'] > 0) &
        (sdss_data['i'] > 0) &
        (sdss_data['z'] > 0)
    ]
    sdss_data = sdss_data[['u', 'g', 'r', 'i', 'z', 'redshift', 'ra', 'dec', 'class']]

    # Similar objects
    sdss_data.drop(sdss_data.index[sdss_data.index.isin(hii_data.index)], inplace=True)
    return pd.concat([hii_data, sdss_data])


def gal_tvt_split(fracs = (0.8, 0.1, 0.1), random_state = 172):
    data = load_catalogue().drop(columns=['ra', 'dec', 'z']).reset_index(drop=True)
    data = data.sample(frac=1, random_state=random_state+42)
    data['class'] = pd.Categorical(data['class'])

    X_train = data.sample(frac=fracs[0], random_state=random_state)
    y_train = X_train.pop('class')
    X_rest = data.drop(X_train.index)
    X_val = X_rest.sample(frac=fracs[1]/(1-fracs[0]), random_state=random_state+42)
    y_val = X_val.pop('class')
    X_test = X_rest.drop(X_val.index)
    y_test = X_test.pop('class')

    return X_train, y_train, X_val, y_val, X_test, y_test

def gal_tt_split(frac = 0.9, random_state = 172):
    data = load_catalogue().drop(columns=['ra', 'dec', 'z']).reset_index(drop=True)
    data = data.sample(frac=1, random_state=random_state+42)
    data['class'] = pd.Categorical(data['class'])

    X_train = data.sample(frac=frac, random_state=random_state)
    y_train = X_train.pop('class')
    X_test = data.drop(X_train.index)
    y_test = X_test.pop('class')

    return X_train, y_train, X_test, y_test

