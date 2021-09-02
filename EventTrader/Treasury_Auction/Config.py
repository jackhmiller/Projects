from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
Auction_params = dict(
    date_dict={
        'train_start': '2010-01-01',
        'train_end': '2020-12-31',
        'test_start': '2021-01-01',
        'test_end': '2021-07-29'},
    event_path="Data/BH_data/US_5Y_Auction.csv",
    event_name="5Y Auction",
    Use_spreads=True,
    Only_spreads=False,
    assets=['Bond2Y', 'Bond5Y', 'Bond10Y', 'Bond30Y'],
    classifier_params={'Algorithm': XGBClassifier(use_label_encoder=False,
                                                  verbosity=0),  # GaussianProcessClassifier()
                       'Prob_thresh': 0.60,
                       'HP_opt': 'None',  # ['None', 'SK_CV', 'Hyperopt']
                       'Macro_features': ['vix'],
                       'Index_features': [],  # example: 'NQ1 Index', 'SP1 Index', 'VG1 Index'
                       'Features_window': 5,
                       'Folds': 5,
                       'Autoencoder': False,
                       'Feature_sel': 'Tree',
                       'Signal_encoding': False,
                       'Train_test': 0.80,
                       },
    vix_thresh=40,
    max_event_days_offset=5,
    exposure_templates=[(1, 0)],
    returns_file_path="Data/bonds_generic1_futures.csv",
    bh_returns_file_path="Data/BH_data/Historical_returns.csv",
    spread_weights_file_path="Data/forward_risk_bonds_data.csv",
    auction_data_file_path="Data/hy_competitive_bonds_data.csv",
    bond_futures_outrights_map={
            'TU1 Comdty': 'Bond2YR',
            'FV1 Comdty': 'Bond5YR',
            'US1 Comdty': 'Bond10YR',
            'WN1 Comdty': 'Bond30YR'},
    bond_futures_spreads_map={
            'Curve2YR5YR': ('TU1 Comdty', 'FV1 Comdty'),
            'Curve2YR10YR': ('TU1 Comdty', 'US1 Comdty'),
            'Curve2YR30YR': ('TU1 Comdty', 'WN1 Comdty'),
            'Curve5YR10YR': ('FV1 Comdty', 'US1 Comdty'),
            'Curve5YR30YR': ('FV1 Comdty', 'WN1 Comdty'),
            'Curve10YR30YR': ('US1 Comdty', 'WN1 Comdty')},
    auction_codes={
        '2Y Auction': {'HY': 'USB2YTA Index',
                       'CT': 'USB2YTC Index',
                       'CA': 'USB2YCA Index'},
        '5Y Auction': {'HY': 'USB5YTA Index',
                       'CT': 'USB5YTC Index',
                       'CA': 'USB5YCA Index'},
        '10Y Auction': {'HY': 'USN10YTA Index',
                        'CT': 'USN10YC Index',
                        'CA': 'USN10YCA Index'},
        '30Y Auction': {'HY': 'USBD30YT Index',
                        'CT': 'USBDYTC Index',
                        'CA': 'USBDYCA Index'}}
)

DATA_PATH = 'Sparks_alpha/Macro_Data'