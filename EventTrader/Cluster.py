import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import talib
#import hdbscan
from ast import literal_eval
import datetime as dt
import re
from itertools import groupby
from Auction_model.Config import Auction_params as PARAMS

class ClusterAgents:

    def __init__(self, exposure_table, event, raw_returns, prices):
        self.outright_map = PARAMS['bond_futures_outrights_map'] #todo change
        self.prices = prices
        self.indicators = {'KAMA': talib.KAMA, 'CMO': talib.CMO, 'MOM': talib.MOM, 'ROC': talib.ROC}
        self.clustering_algo = SpectralClustering()
        self.exposure_table = exposure_table.iloc[np.where(exposure_table.index.date < event - dt.timedelta(days=25))]
        self.event_seq_matrix = self.get_event_seq_labels()
        self.event = event
        self.raw_returns = raw_returns
        self.cluster_matrix = self.gen_asset_exposures()
        #self.cluster_matrix = self.add_clustering_features()
        self.labels = None
        self.agent_labels = None
        self.best_cluster_SM = None
        self.final_agent_SM = None

    def gen_asset_exposures(self):
        asset_exposures = []
        for col in self.raw_returns.columns:
            asset_agent_exp = (self.exposure_table.mul(self.raw_returns[col], axis=0) + 0).fillna(0)
            asset_agent_exp = asset_agent_exp.add_prefix(col + ' ')
            asset_exposures.append(asset_agent_exp)

        exp_m = pd.concat(asset_exposures, axis=1)

        return exp_m + 0

    def get_event_seq_labels(self):
        exposure_abs = self.exposure_table.abs()
        df = exposure_abs.diff() * exposure_abs

        df.iloc[0, :] = exposure_abs.iloc[0, :]
        df = df.astype(int)
        df = df.cumsum() * exposure_abs
        #df = df.shift(1, fill_value=0)

        return df

    def add_clustering_features(self):
        df = self.sparse_matrix
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['DOW'] = df.index.dayofweek

        return df

    def add_clustering_TIs(self, window=10):
        df = self.prices
        df = df[:self.cluster_matrix.index[-1]]

        frames = [
            df.apply(talib.KAMA)[-window:].mean().to_frame(name='KAMA'),
            df.apply(talib.CMO)[-window:].mean().to_frame(name='CMO'),
            df.apply(talib.MOM)[-window:].mean().to_frame(name='MOM'),
            df.apply(talib.ROC)[-window:].mean().to_frame(name='ROC')
         ]

        return pd.concat(frames, axis=1)


    def agent_TIs(self):
        df = self.prices.rename(mapper=self.outright_map, axis=1)  # todo change
        df = df[:self.cluster_matrix.index[-1]]
        TI_list = []
        for ti in self.indicators.keys():
            di = []
            for i in df.columns:
                temp_df = self.get_event_metric(self.exposure_table.mul(
                    df.apply(self.indicators[ti])[i].dropna(), axis=0).fillna(method='ffill').dropna())
                temp_df = temp_df.set_index(i + ' ' + temp_df.index.astype(str))
                di.append(temp_df)

            concat_ti_df = pd.concat(di)
            concat_ti_df.columns = [ti]
            TI_list.append(concat_ti_df)

        return pd.concat([i for i in TI_list], axis=1)

    def create_clusters(self):

        event_stats_df = self.event_seq_pnl()
        #agent_TIs = self.agent_TIs()
        TIs = self.add_clustering_TIs()
        comb_df = self.add_join_merge(event_stats_df, TIs)
        #comb_df = event_stats_df.merge(agent_TIs, left_index=True, right_index=True)
        scaler = StandardScaler().fit_transform(comb_df.values)
        kpca = KernelPCA(n_components=3)
        f = kpca.fit_transform(scaler)
        self.clustering_algo.fit(f)

        self.labels = self.clustering_algo.labels_
        raw_agent_labels = sorted(list(zip([i for i in self.cluster_matrix.columns], self.labels)), key=lambda x: x[1])
        self.agent_labels = [i for i in raw_agent_labels if ('DOW' not in i) and ('year' not in i) and ('month' not in i)]

    def event_seq_pnl(self):
        temp_exp_df = self.get_event_seq_labels()
        temp_exp_df.columns = [str(i) for i in temp_exp_df.columns]
        res = {}
        for c, s in self.cluster_matrix.items():
            res[c] = s.groupby(temp_exp_df[c.split('YR ')[1]]).sum().loc[1:] #todo will cause issues

        return pd.DataFrame.from_dict(res).agg(['mean', 'sum', 'std'], axis=0).T

    def get_best_cluster(self, window=250):
        good_cluster = []
        for cluster in list(set(self.labels)):
            if cluster == -1:
                continue
            cluster_returns = self.cluster_matrix[[i[0] for i in self.agent_labels if i[1] == cluster]][-window:]
            event_r = self.get_event_metric(cluster_returns).mean().mean() #TODO MUST CHANGE TO OTHER FUNCTION
            if event_r > 0:
                good_cluster.append(cluster)

        self.best_cluster_SM = self.cluster_matrix[self.cluster_matrix[[i[0] for i in self.agent_labels if i[1] in good_cluster]].columns]
        print("{:.2%} Dimensionality Reduction".format(self.best_cluster_SM.shape[1] / self.cluster_matrix.shape[1]))
        top_agents = self.get_top_per_asset(self.get_event_metric(self.best_cluster_SM[-window:]))
        self.final_agent_SM = self.best_cluster_SM[top_agents]

        assets, agent_info = self.get_agent_info(top_agents)

        return assets, agent_info

    @staticmethod
    def add_join_merge(event_df, TI_df):
        event_df['asset'] = [i.split(' (')[0] for i in event_df.index.values]
        TI_df['asset'] = [i.split('_')[0] for i in TI_df.index.values]

        df = pd.merge(event_df, TI_df, how='left', on='asset', left_index=True).drop('asset', axis=1)

        return df

    @staticmethod
    def get_agent_info(good_cluster):
        asset_trade_combos = {}
        for col in good_cluster:
            # Auction Model
            if 'Curve' or 'Bond' in col:
                asset = col.split(" ", 1)[0]
            # ME Model
            else:
                regex = r'(\w*) '
                words = re.findall(regex, col)
                asset = ' '.join([i for i in words if len(i) > 0])

            agent_info_search = re.findall('\(.*?\)', col)
            direction = literal_eval(agent_info_search[0])[0]
            day_range = literal_eval(agent_info_search[1])
            if asset in asset_trade_combos.keys():
                asset_trade_combos[asset].append({'direction': direction, 'day_range': day_range})
            else:
                asset_trade_combos[asset] = {'direction': direction, 'day_range': day_range}

        assets = list(set([key for key in asset_trade_combos.keys()]))

        return assets, asset_trade_combos

    @staticmethod
    def get_event_metric(df, method='returns'):
        f = lambda x: x == 0
        res = {}
        for col in df.columns:
            lst = df[col].values
            vec = [i for k, g in groupby(lst, f) for i in (g if k else (sum(g),))]
            if method == 'returns':
                vec_sum = sum([i for i in vec if i != 0])
                vec_len = len([i for i in vec if i != 0])
                res[col] = vec_sum / vec_len
            if method == 'hit_ratio':
                vec_pnl = [i for i in vec if i != 0]
                hit_miss = [1 if i > 0 else -1 for i in vec_pnl]
                res[col] = hit_miss.count(1)/len(hit_miss)
        return pd.DataFrame(data=res.values(), index=res.keys())

    @staticmethod
    def get_top_per_asset(series):
        df = pd.DataFrame(series)
        df = df.rename(columns={df.columns[0]: 'actual'})
        df['agent'] = df.index
        df['asset'] = [i.split(" (")[0] for i in df.index]
        lookup_vals = df.groupby('asset')['actual'].max().values
        agents = list(df[df['actual'].isin(lookup_vals)].index.values)
        # agents = []
        # for asset in list(df['asset'].unique()):
        #     agents.append(df[df['asset'] == asset].sort_values(by='actual', ascending=False)['agent'][0])
        return agents

    def cluster_agents(self):
        self.create_clusters()
        assets, agent_info = self.get_best_cluster()
        return assets, agent_info


class ClusterPostProcessing:
    def __init__(self, cluster_instance):
        self.exposure_table = cluster_instance.exposure_table
        self.agents = list(cluster_instance.final_agent_SM.columns)
        self.agent_exposures = self.cluster_postprocess()
        self.sequential_labels = self.get_sequential_labels()
        self.agent_dates = self.get_agent_dates()

    def cluster_postprocess(self):
        et = self.exposure_table
        et.columns = [str(i) for i in et.columns]
        agent_cols = {i.split(' ', 1)[0]: i.split(' ', 1)[1] for i in self.agents}
        res = {}
        for k, v in agent_cols.items():
            res[k] = et[v]
        return pd.DataFrame(res)

    def get_sequential_labels(self):
        exposure_abs = self.agent_exposures.abs()
        df = exposure_abs.diff() * exposure_abs

        df.iloc[0, :] = exposure_abs.iloc[0, :]
        df = df.astype(int)
        df = df.cumsum() * exposure_abs
        #df = df.shift(1, fill_value=0)

        return df

    def get_agent_dates(self):
        df = self.sequential_labels
        result = {}
        for c, s in df.items():
            gb = s.reset_index().groupby(c)
            f, g = gb.first(), gb.last()
            result[c] = pd.DataFrame({'first': f.iloc[:, 0].values, 'last': g.iloc[:, 0].values},
                                     index=f.index.rename(None)).loc[1:]

        return result


#####################################################################################################
    # def agent_TIs(self):
    #     df = self.prices.rename(mapper=self.outright_map, axis=1)  # todo change
    #     df = df[:self.cluster_matrix.index[-1]]
    #     TI_dict = {}
    #     for ti in self.indicators.keys():
    #         di = []
    #         for i in df.columns:
    #             di.append(self.get_event_metric(self.exposure_table.mul(
    #                 df.apply(self.indicators[ti])[i].dropna(), axis=0).fillna(method='ffill').dropna()))
    #         TI_dict[ti] = np.concatenate(di)
    #
    #     return pd.DataFrame({k: pd.Series([i[0] for i in v]) for k, v in TI_dict.items()})