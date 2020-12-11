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
    sdss_data = pd.read_csv('data/SDSS/sdss-all-objects.csv', skiprows=1).set_index('objid')
    sdss_data = sdss_data[['u', 'g', 'r', 'i', 'z', 'redshift', 'ra', 'dec', 'class']]
    sdss_data = sdss_data[sdss_data['redshift'] > 0]
    sdss_data = sdss_data[
        (sdss_data['u'] > 0) &
        (sdss_data['g'] > 0) &
        (sdss_data['r'] > 0) &
        (sdss_data['i'] > 0) &
        (sdss_data['z'] > 0)
        ]

    # Similar objects
    sdss_data = sdss_data.drop(sdss_data.index.intersection(hii_data.index, sort=False))


    return pd.concat([hii_data, sdss_data])


def gal_tt_split(frac = 0.8, random_state = 172):
    data = load_catalogue().drop(columns=['ra', 'dec', 'z']).reset_index(drop=True)
    data = data.sample(frac=1, random_state=random_state+42)
    data['class'] = pd.Categorical(data['class'])

    X_train = data.sample(frac=0.8, random_state=random_state)
    X_test = data.drop(X_train.index)
    y_train = X_train.pop('class')
    y_test = X_test.pop('class')

    return X_train, y_train, X_test, y_test
