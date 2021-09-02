import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import warnings
import argparse
from functools import partial
import statsmodels.api as sm
import ruptures as rpt
import Bayes_cpd as bcpd
import yaml
import datetime as dt
import boto3
import s3_utils
warnings.filterwarnings("ignore")

bucket = # your bucket name to store results in s3

# -------------------------------- Ensemble Classifier Classes ----------------------------------------

# Base Class
class Classifier_Base():

    """
    Helper class with pre- and post- classification processing functions.
    """

    def preprocess(self):
        df = self.data.dropna()
        try:
            assert 'value' in df.columns
        except AssertionError:
            print('Price column not labeled correctly')

        df['date'] = pd.to_datetime(df.index).weekday
        df = df[df['date'] < 5]

        if self.resample[-1] == 'D':
            return df[['value']].resample(self.resample).last().dropna()
        else:
            return df[['value']].resample(self.resample).mean().dropna()

    def classify_weeks(self, df):
        #df = df.resample('W-SUN').sum()
        df = df.resample('1D').sum()
        df.loc[df['Class'] < self.threshold, 'Class'] = 0
        df.loc[df['Class'] >= self.threshold, 'Class'] = 1
        return df, df.groupby('Class').size()

# Classifier #1
class Markov_Switching_AR(Classifier_Base):

    """
    Fits a dynamic regression model that exhibits different dynamics across unobserved states
    using state-dependent parameters to accommodate structural breaks or other multiple-state phenomena.
    Transitions between the unobserved states follow a Markov chain.
    A dynamic regression model (rather than an autoregressive one) is used to allow for a quick adjustment
    after the process changes state.
    Here it is implemented using price volatility.
    """

    _RESAMPLE_PARAMS = {'1D': {'threshold': 1, 'n_regimes': 2, 'window': 7},
                       '60min': {'threshold': 18, 'n_regimes': 3, 'window': 24}}

    def __init__(self, data, resample):
        self.data = data
        self.resample = resample
        try:
            self.threshold = self._RESAMPLE_PARAMS[self.resample]['threshold']
        except KeyError:
            print("Resampling frequency from the config file is incorrect")

    def calc_volatility(self, df):
        self.volatility = (df.rolling(window=self._RESAMPLE_PARAMS[self.resample]['window']).std()
                           * np.sqrt(self._RESAMPLE_PARAMS[self.resample]['window'])).dropna()
        self.start = pd.to_datetime(self.volatility.index[0]).date()
        self.end = pd.to_datetime(self.volatility.index[-1]).date()
        return

    def fit_model(self):
        mod = sm.tsa.MarkovRegression(endog=self.volatility.iloc[1:],
                                      k_regimes=self._RESAMPLE_PARAMS[self.resample]['n_regimes'],
                                      switching_variance=False,
                                      dates=pd.date_range(self.start, self.end))
        res_areturns = mod.fit()
        res_index = self._RESAMPLE_PARAMS[self.resample]['n_regimes'] - 1
        high_vix = pd.Series(res_areturns.smoothed_marginal_probabilities[res_index].values,
                             index=self.volatility.index.values[1:])
        self.plot_data = high_vix
        hv = pd.DataFrame(high_vix)
        hv = hv.rename(columns={0: 'prob'})
        hv['Class'] = 0
        hv.loc[hv['prob'] > 0.1, 'Class'] = 1
        return hv

    def run_classifier(self):
        df = super(Markov_Switching_AR, self).preprocess()
        self.calc_volatility(df)
        df2 = self.fit_model()
        final_df, aggregated_res = self.classify_weeks(df2) #super(Markov_Switching_AR, self).classify_weeks(df2)
        return final_df, aggregated_res

# Classifier #2
class BB_Classifier(Classifier_Base):

    """
    Bollinger Bands consist of
    1) An N-period moving average (MA)
    2) An upper band at K times an N-period standard deviation above the moving average (MA + Kσ),
    3) A lower band at K times an N-period standard deviation below the moving average (MA − Kσ).

    The classifier is rules-based.
    """

    _RESAMPLE_PARAMS = {'1D': {'threshold': 1, 'timeperiod': 10},
                        '240min': {'threshold': 4, 'timeperiod': 30}}

    def __init__(self, data, resample, nbdevup=2, nbdevdn=1):
        self.data = data
        self.resample = resample
        try:
            self.threshold = self._RESAMPLE_PARAMS[self.resample]['threshold']
        except KeyError:
            print("Resampling frequency from the config file is incorrect")
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn

    def _calculate_bands(self, df):
        return talib.BBANDS(df['value'],
                            timeperiod=self._RESAMPLE_PARAMS[self.resample]['timeperiod'],
                            nbdevup=self.nbdevup,
                            nbdevdn=self.nbdevdn)

    def plot_bands(self, df):
        plt.figure(figsize=(20, 10))
        plt.plot(df['value'])
        plt.plot(self.bands[0])
        plt.plot(self.bands[2])
        return

    def _create_classification(self, df):
        bands = self._calculate_bands(df)
        df['upper'] = bands[0]
        df['lower'] = bands[2]
        df['Class'] = 0
        df = df.dropna()
        df.loc[(df['value'] > df['upper']) | (df['value'] < df['lower']), 'Class'] = 1
        return df

    def run_classifier(self):
        df = super(BB_Classifier, self).preprocess()
        df2 = self._create_classification(df)
        final_df, aggregated_res = super(BB_Classifier, self).classify_weeks(df2)
        return final_df, aggregated_res

# Classifier #3
class NATR_Classifier(Classifier_Base):

    """
    Normalized Average True Range is a measure of volatility over temporal periods.
    It represents roughly how much you can expect a security to change in price on any given day.

    The classifier is rules-based.
    """
    _RESAMPLE_PARAMS = {'1D': {'NATR_period':14, 'window':180, 'threshold':1},
                        '60min': {'NATR_period': 14 * 22, 'window': 2800, 'threshold': 16},
                        '120min': {'NATR_period': 14 * 11, 'window': 1400, 'threshold': 8}}

    def __init__(self, data, resample='60min'):
        self.data = data
        self.resample = resample
        try:
            self.threshold = self._RESAMPLE_PARAMS[self.resample]['threshold']
        except KeyError:
            print("Resampling frequency from the config file is incorrect")


    def _OHLC_preprocess(self):
        df = self.data.dropna()
        df['date'] = pd.to_datetime(df.index)
        return df[['value']].resample(self.resample).ohlc().dropna()

    def _NATR(self):
        df = self._OHLC_preprocess()
        df['NATR'] = talib.NATR(high= df[('value', 'high')],
                                low = df[('value', 'low')],
                                close = df[('value', 'close')],
                                timeperiod=self._RESAMPLE_PARAMS[self.resample]['NATR_period'])
        df['NATR_threshold'] = df['NATR'].rolling(window=self._RESAMPLE_PARAMS[self.resample]['window']).mean() + 1.5 * (
            df['NATR'].rolling(window=self._RESAMPLE_PARAMS[self.resample]['window']).std())
        return df

    def _create_classification(self, df):
        df = df.dropna()
        df['Class'] = 0
        df.loc[df['NATR'] > df['NATR_threshold'], 'Class'] = 1
        return df[['Class']]

    def run_classifier(self):
        df = self._NATR()
        classified = self._create_classification(df)
        final_df, aggregated_res = self.classify_weeks(classified) #super(NATR_Classifier, self).classify_weeks(classified)

        return final_df, aggregated_res

# Classifier #4
class PELT_Classifier(Classifier_Base):

    """
    Pruned exact linear time algorithm- source:
    Killick Rebecca, Fearnhead P, Eckley Idris A(2012). Optimal detection of change points with a linear computational costs. JASA, 107, 1590-1598.

    Algorithm is implemented via the ruptures package- source code:
    https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/_modules/ruptures/detection/pelt.html
    """

    _RESAMPLE_PARAMS = {'60min': {'window':20},
                       '300min': {'window':10}}

    def __init__(self, data, resample, penalty='l2', breaks=1, threshold=1):
        self.data = data
        self.resample = resample
        self.penalty = penalty
        self.num_breakpoints = breaks
        self.threshold = threshold


    @staticmethod
    def _test_data_segment(df):
        """
        Ensures grouping method for weeks does not include data for same week number, but in a different year
        """
        index_diff = df.index.to_series().diff().dt.days.values
        bad_rows = np.where(index_diff > 5)
        if len(bad_rows[0]) == 0:
            return df
        else:
            print('Dropped', df.index[bad_rows].values)
            df = df.drop(df.index[bad_rows])
            return df

    def _run_PELT(self, df):
        """
        Implementation of the PELT algorithm, using a single changepoint fit to perform a binary segmentation of a
        week's worth of data.
        The algorithm computes the segmentation which minimizes the constrained sum of approximation errors, using
        the L2 penalty (detects mean-shifts in signal). Following the segmentation, the statistical moments of the two
        segments are compared to determine the relative significance of the changepoint.

        L1 cost function detects changes in the median of a signal. Overall, it is a more robust estimator of a shift
        in the central point (mean, median, mode) of a distribution, and less applicable to financial time series.

        Algorithm is implemented on price volatility.
        """
        df['Class'] = 0
        df['volatility'] = (df['value'].rolling(window=self._RESAMPLE_PARAMS[self.resample]['window']).std()
                            * np.sqrt(self._RESAMPLE_PARAMS[self.resample]['window'])).dropna()
        # results = {}
        for year in set(df.index.year):
            inner = {}
            for week in df[str(year)].index.week.unique():
                this_weeks_data = df[str(year)][df[str(year)].index.week == week]

                self.t = this_weeks_data
                this_weeks_data = self._test_data_segment(this_weeks_data)

                if len(this_weeks_data) < 2:
                    pass
                else:
                    algo = rpt.Dynp(model=self.penalty).fit(this_weeks_data['volatility'].values)
                    result = algo.predict(n_bkps=self.num_breakpoints)
                    before = this_weeks_data['volatility'].iloc[:result[0]].mean()
                    after = this_weeks_data['volatility'].iloc[result[0]:].mean()
                    after_std = this_weeks_data['volatility'].iloc[result[0]:].std()

                    if after - 1.5 * after_std > before:
                        # inner[week] = 1
                        df[str(year)].loc[this_weeks_data.index, 'Class'] = 1
                    # weeks += 1
        #                 else:
        #                     inner[week] = 0
        # results[str(year)] = inner
        return df

    def run_classifier(self):
        df = super(PELT_Classifier, self).preprocess()
        df2 = self._run_PELT(df)
        final_df, aggregated_res = self.classify_weeks(df2) #super(PELT_Classifier, self).classify_weeks(df2)
        return final_df, aggregated_res

# Classifier #5
class Bayes_CPD_Classifier(Classifier_Base):

    """
    Bayesian Changepoint detection algorithm- source:
    Fearnhead, Exact and Efficient Bayesian Inference for Multiple Changepoint problems, Statistics and computing 16.2 (2006).
    Offline algorithm is implemented via the bayesian_changepoint_detection package, with modifications.

    Source code:
    https://github.com/hildensia/bayesian_changepoint_detection)
    """

    _STEP_SIZES = {'60min': 700, '120min': 700}

    def __init__(self, data, resample, threshold=1, prob_threshold=0.40, truncate=-40):
        self.data = data
        self.resample = resample
        self.threshold = threshold
        self.truncate = truncate
        self.prob_threshold = prob_threshold

    def _detect_bayes_cp(self, df):
        """

        """
        data = df.diff().ffill().dropna().values
        step_size = self._STEP_SIZES[self.resample]
        n_start = 0
        n_end = step_size
        results = {}
        while n_end < len(df):
            start = df.index[n_start].date().strftime("%Y-%m-%d")
            end = df.index[n_end].date().strftime("%Y-%m-%d")
            print("Working on:", [start, end])
            _, _, Pcp = bcpd.offline_changepoint_detection(data=data[n_start:n_end],
                                                           prior_func=partial(bcpd.const_prior,l=(len(data[n_start:n_end]) + 1)),
                                                           observation_log_likelihood_function=bcpd.gaussian_obs_log_likelihood,
                                                           truncate=self.truncate)
            results[f"{start}, {end}"] = Pcp
            n_start += step_size
            n_end += step_size

        res_2 = {k: np.exp(v).sum(0).reshape(-1, 1) for k, v in results.items()}
        bayes_res = np.concatenate([i for i in res_2.values()], axis=0)
        prob_df = pd.DataFrame(data=bayes_res, index=df.iloc[:len(bayes_res)].index, columns=['Prob'])

        return prob_df

    def _create_classification(self, df):
        df['Class'] = 0
        df.loc[df['Prob'] > self.prob_threshold, 'Class'] = 1
        return df[['Class']]

    def run_classifier(self):
        df = super(Bayes_CPD_Classifier, self).preprocess()
        prob_df = self._detect_bayes_cp(df)
        df2 = self._create_classification(prob_df)
        final_df, aggregated_res = self.classify_weeks(df2) #super(Bayes_CPD_Classifier, self).classify_weeks(df2)
        return final_df, aggregated_res


def build_ensemble(results_dict: dict, percent=0.40):
    """
    Function that builds the ensemble by combining the results dataframes from the different classifiers
    into a single DF, and subsequently performs hard voting to classify weeks that exceed the voting threshold.
    :param frames: list of dataframes with weekly datetime index and single columnn with a classification value of 1 or 0
    :param column: name of classification label column for each frame
    :param percent: hard voting threshold
    :return: final DF with classification of 1 or 0, along with aggregated size of class
    """
    num_classifiers = len(results_dict.keys())
    min_votes = num_classifiers * percent
    #ensemble = (pd.concat([df[column] for df in frames], axis=1)).dropna()
    ensemble = pd.concat(results_dict, axis=1).dropna().astype(int)
    ensemble['Final_class'] = np.where(ensemble.sum(axis=1) >= min_votes, 1, 0)
    return ensemble, ensemble.groupby('Final_class').size()


def run_ensemble(components: dict, data: pd.DataFrame):
    """
    Instantiates and runs each ensemble component with the associated resampling frequency, and then combines the elements of the
    different classifiers to create the ensemble, which then votes using majority-rule for the final classification.
    """
    component_results = {}
    for key, values in components.items():
        for val in values:
            print(f'Running {key}, resampled at: {val}')
            try:
                results_df, _ = CLASSIFIER_DICT[key](data, val).run_classifier()
                #component_results.append(results_df)
            except AssertionError as e:
                print(f"{key}{val}: {e}")
                continue

            component_results[(key, val)] = results_df['Class']

    ensemble_df, ensemble_results = build_ensemble(component_results)

    return ensemble_df, ensemble_results

# -------------------------------- Post-ensemble Construction Utility Functions ----------------------------------

def calc_kpi(data: pd.DataFrame, ensemble: pd.DataFrame, kpi = 'returns'):
    """
    Accuracy KPI for the ensemble
    KPI 1) Average weekly price movement
    KPI 2) Avg High for the VIX resampled weekly
    """
    if kpi == 'returns':
        validation = data['value'].pct_change().abs().dropna().resample('W-SUN').sum()
        val_df = pd.DataFrame(validation)
        val_df['label'] = np.where(val_df['value'] > validation.mean(), 1, 0)
        ensemble['KPI'] = val_df['label']

        return ensemble['Final_class'].sum() / val_df['label'].sum(), validation, ensemble

    else:
        vix = pd.read_csv("VIX.csv", index_col=0, parse_dates=True)
        val_df = vix.resample('W-SUN').mean().loc[ensemble.index]
        val_df['label'] = np.where(val_df['High'] > 20, 1, 0)

        return ensemble['Final_class'].sum() / val_df['label'].sum(), val_df['High']


def plot_classification(ensemble, validation):
    """
    Optional function to output the classification of all historical data, overlaid on average price movements
    and absolute returns.
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(ensemble['Final_class'], linewidth=0.5)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Class", color="red", fontsize=20)
    ax.fill_between(ensemble['Final_class'].index, ensemble['Final_class'], alpha=0.5, color='green')

    ax2 = ax.twinx()
    ax2.plot(validation.loc[ensemble.index[0]:], color="blue", linewidth=2.0)
    ax2.set_ylabel("Percent Returns", color="blue", fontsize=20)

    ax2.axhline(y=validation.mean(), color='r', linestyle='-')

    return


def data_validation(df):
    try:
        assert 'value' in df.columns
    except AssertionError:
        print('Check the label for the price column')

    return


def read_config_file(flag):
    """
    Reads external YAML configuration file
    """
    yaml_file = open('config.yml')
    cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return cfg[flag]


def split_train_oos(df, start_oos='2020'):
    """
    Splits data into a "train" period, and a walk-forward out-of-sample period
    The ensemble is fed all of the train data, and walks forward one day, recievinng all of the new data
    associated with that new day.
    """
    oos_days = []
    for idx, day in df.loc[start_oos:].groupby(df.loc[start_oos:].index.date):
        oos_days.append(day)
    return df.loc[:start_oos], oos_days


def walk_forward(components_dict, train, oos):
    """
    Adds a days-worth of data for each iteration to the historical data, and runs the ensemble.
    The results are appended to a dataframe.
    """

    classification_results = {}
    for i in range(len(oos)):
        # Make sure only one days worth of data is passed
        assert len(set([i.strftime("%Y-%m-%d") for i in list(oos[i].index.date)])) == 1
        assert len(list(set(oos[i].index.date))) == 1

        # Get date for new data
        date = oos[i].index.date[0].strftime("%Y-%m-%d")

        # Add new data to training data
        train = pd.concat([train, oos[i]])

        # Main execution block
        DoW = list(set(oos[i].index.date))[0].weekday()

        # Run classifier if current DoW is not Sunday
        if DoW != 6:
            ensemble_df, ensemble_agg_res = run_ensemble(components_dict, train)
            classification_results[date] = ensemble_df[-1:]
        else:
            continue

    return pd.DataFrame(pd.concat(classification_results))


def main(args):
    config_flag = args['resample']
    components_dict = read_config_file(config_flag)

    # Read in data
    df = pd.read_csv("es_generic1_2017_until_today.csv", index_col=0, parse_dates=True)
    data_validation(df)

    # Split data for walk-forward implementation
    train_data, oos = split_train_oos(df)

    # Execution entry point
    oos_results = walk_forward(components_dict, train_data, oos)

    # Write to S3
    s3_utils.write_dataframe_to_csv(oos_results, bucket, 'results.csv')

    # Calc KPI
    # accuracy, validation_data, final_df = calc_kpi(df, ensemble_df)

    # Plot results
    # plot_classification(ensemble_df, validation_data)

    # print(ensemble_agg_res, accuracy)
    # final_df.to_csv('classification_df.csv')

    return


# ----------------------------------- Globals ---------------------------------------------------
CLASSIFIER_DICT = {'BB': BB_Classifier,
                   'NATR': NATR_Classifier,
                   'MK': Markov_Switching_AR,
                   'PELT': PELT_Classifier,
                   'Bayes': Bayes_CPD_Classifier
                   }
# ----------------------------------- Entry Point ---------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-resample",
                        "--resample",
                        help="Ensemble components by resampling freq",
                        type=str,
                        required=False,
                        default='all')

    args = parser.parse_args()
    main(vars(args))
