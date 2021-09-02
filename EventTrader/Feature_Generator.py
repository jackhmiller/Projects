import pandas as pd
import numpy as np
import talib
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from Auction_model.Config import DATA_PATH, Auction_params
import requests


def create_encoder(df):
    X = df.values
    n_inputs = X.shape[1]
    visible = Input(shape=(n_inputs,))

    # Encoder level 1
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # Encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    n_bottleneck = round(float(n_inputs) / 2.0)
    bottleneck = Dense(n_bottleneck)(e)
    # Decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # Decoder level 2
    d = Dense(n_inputs * 2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    output = Dense(n_inputs, activation='linear')(d)

    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    autoencoder = Model(inputs=visible, outputs=bottleneck)

    # autoencoder.predict(np.hstack(predict_x_raw).reshape(1, -1))

    return autoencoder.predict(X)


def Gramian_Angular_Field(series):
    min_ = np.amin(series)
    max_ = np.amax(series)

    scaled_series = (2 * series - max_ - min_) / (max_ - min_)
    scaled_series = np.where(scaled_series >= 1., 1., scaled_series)
    scaled_series = np.where(scaled_series <= -1., -1., scaled_series)

    phi = np.arccos(scaled_series)
    r = np.linspace(0, 1, len(scaled_series))
    gaf = tabulate(phi, phi, cos_sum)

    return gaf


def get_macro_features(dat, features, live=False):

    macro_raw = {
        'vix': {'file': 'VIX_hist', 'col': 'vix_close'},
        'libor_3d': {'file': 'USD3MTD156N (1)', 'col': 'USD3MTD156N'},
        'TEDRATE': {'file': 'TEDRATE', 'col': 'TEDRATE'},
        'T10Y2Y': {'file': 'T10Y2Y', 'col': 'T10Y2Y'},
        'BAMLH': {'file': 'BAMLH0A0HYM2', 'col': 'BAMLH0A0HYM2'},
        'T5YIFR': {'file': 'T5YIFR', 'col': 'T5YIFR'},
        'T3MFF': {'file': 'T3MFF', 'col': 'T3MFF'},
        'T1YFF': {'file': 'T1YFF', 'col': 'T1YFF'}
    }

    read_macro_data = {}
    for i in features:
        read_macro_data[macro_raw[i]['col']] = pd.read_csv(f"../../{DATA_PATH}/{macro_raw[i]['file']}.csv", dayfirst=True, index_col=0, parse_dates=True)

    transformed_macro_feat = []
    for k, frame in read_macro_data.items():
        frame = frame[frame[k] != '.']
        ready = frame[k].apply(pd.to_numeric).fillna(method='ffill')
        transformed_macro_feat.append(ready.resample('D').asfreq().fillna(method='ffill').loc[dat.index])

    unscaled_macro = pd.DataFrame(transformed_macro_feat).T

    # Fed Account Balance
    if not live:
        fed_balance = pd.read_csv(f"../../{DATA_PATH}/DTS_OpCashBal_20091003_20210716.csv", dayfirst=True, index_col=0,
                                  parse_dates=True)
    else:
        for _ in range(0, 5):
            url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/dts_table_1"
            response = requests.get(url)
            while response.status_code != 200:
                response = requests.get(url)
            fed_balance = response.json()['data']

    fed_balance = fed_balance.iloc[::-1]
    fed_balance = fed_balance[fed_balance['Type of Account'] == 'Federal Reserve Account'][['Closing Balance Today', 'Opening Balance Today', 'Opening Balance This Month']]

    unscaled_features = pd.merge(unscaled_macro,
                                 fed_balance,
                                 how='left',
                                 left_index=True,
                                 right_index=True).fillna(method='ffill')
    scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(unscaled_features),
                        index=unscaled_features.index,
                        columns=unscaled_features.columns)


def gen_TIs(df, asset, signal_encoder=Auction_params['classifier_params']['Signal_encoding']):

    if signal_encoder:
        df['MOM'] = np.where(talib.MOM(df[asset], timeperiod=5) > 0, 1, 0)
        df['RSI'] = np.where((talib.RSI(df[asset]) <= 30) | (talib.RSI(df[asset]).shift(-1) > talib.RSI(df[asset])), 1, 0)
        df['PPO'] = np.where(talib.PPO(df[asset], fastperiod=12, slowperiod=26, matype=0).shift(-1) > talib.PPO(df[asset], fastperiod=12, slowperiod=26, matype=0), 1, 0)
        df['TRIX'] = np.where(talib.TRIX(df[asset], timeperiod=30).shift(-1) > talib.TRIX(df[asset], timeperiod=30), 1, 0)
        df['APO'] = np.where(talib.APO(df[asset], fastperiod=12, slowperiod=26, matype=0).shift(-1) > talib.APO(df[asset], fastperiod=12, slowperiod=26, matype=0), 1, 0)
        df['CMO'] = np.where(talib.CMO(df[asset], timeperiod=14).shift(-1) > talib.CMO(df[asset], timeperiod=14), 1, 0)
        # Fourier Transform
        df['FFT'] = np.where(pd.Series(np.real(np.fft.fft(df[asset]))).shift(-1) > pd.Series(np.real(np.fft.fft(df[asset]))), 1, 0)
        # Weighted MA
        df['WMA'] = np.where(df[asset] > talib.WMA(df[asset]), 1, 0)
        # Gramian Angular Field
        GAF = PCA(n_components=1).fit_transform(Gramian_Angular_Field(df[asset]))
        df['GAF'] = np.where(pd.Series(GAF.flatten()).shift(-1) > pd.Series(GAF.flatten()), 1, 0)
    else:
        df['MOM'] = talib.MOM(df[asset], timeperiod=5)
        df['RSI'] = talib.RSI(df[asset])
        df['PPO'] = talib.PPO(df[asset], fastperiod=12, slowperiod=26, matype=0)
        df['TRIX'] = talib.TRIX(df[asset], timeperiod=30)
        df['APO'] = talib.APO(df[asset], fastperiod=12, slowperiod=26, matype=0)
        df['CMO'] = talib.CMO(df[asset], timeperiod=14)
        # Fourier Transform
        df['FFT'] = np.real(np.fft.fft(df[asset]))
        # Weighted MA
        df['WMA'] = talib.WMA(df[asset])
        # Gramian Angular Field
        GAF = PCA(n_components=1).fit_transform(Gramian_Angular_Field(df[asset]))
        df['GAF'] = GAF.flatten()

    # Time Series Decomposition- Residual Extraction
    decomp = df[asset].resample('D').asfreq().fillna(method='ffill')
    df['residual'] = seasonal_decompose(decomp, model='additive').resid.loc[df.index]

    df['DoW'] = df.index.dayofweek
    df['WoY'] = df.index.weekofyear

    df = df.drop(asset, axis=1)

    scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(df),
                        index=df.index,
                        columns=df.columns)


def get_auction_data(event, last_index):
    df = pd.read_csv(Auction_params['auction_data_file_path'], dayfirst=True, parse_dates=True, index_col=0)
    auction_codes = Auction_params['auction_codes'][event]
    df = df[auction_codes.values()].dropna()
    df = df.rename(columns={v: k for k, v in auction_codes.items()})
    df['bid_to_cover'] = df['CT']/df['CA']

    df = df[['bid_to_cover', 'HY']].loc[:last_index]

    scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(df),
                        index=df.index,
                        columns=df.columns)

def get_stock_indexes():
    indexes = pd.read_csv(f"../../{DATA_PATH}/stock_index.csv", dayfirst=True, index_col=0, parse_dates=True)
    indexes = indexes[Auction_params['classifier_params']['Index_features']]
    indexes = indexes.pct_change().dropna()
    scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(indexes),
                        index=indexes.index,
                        columns=indexes.columns)


def tabulate(x, y, f):
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def cos_sum(a, b):
    return math.cos(a+b)


def add_features(df: pd.DataFrame, asset, encoder=Auction_params['classifier_params']['Autoencoder'],
                 macro_features=Auction_params['classifier_params']['Macro_features'],
                 indexes=Auction_params['classifier_params']['Index_features'],
                 event=Auction_params['event_name']):

    if 'Curve' in asset:
        auction_feats = []
    else:
        auction_feats = get_auction_data(event, df.index[-1])

    TIs = gen_TIs(df, asset)

    if len(indexes) > 0:
        index_df = get_stock_indexes()
    else:
        index_df = []

    if len(macro_features) > 0:
        macro_feats = get_macro_features(df, macro_features)
    else:
        macro_feats = []

    feature_frames = [TIs, macro_feats, index_df]

    if len(auction_feats) > 0:
        final = pd.concat([i for i in feature_frames if len(i) > 0], axis=1).join(auction_feats, how='left').fillna(method='ffill').fillna(method='bfill')
    else:
        final = pd.concat([i for i in feature_frames if len(i) > 0], axis=1).fillna(method='ffill').fillna(method='bfill')

    if not encoder:
        return final
    else:
        return create_encoder(final)
