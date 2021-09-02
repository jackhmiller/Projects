import pandas as pd
import datetime as dt
from Auction_model.Config import Auction_params, DATA_PATH
from Utils.Pre_pro import Event_extraction, Agent_assembler
from Utils.Pre_pro.Exposure_table_creator import all_exposure_tables


class Preprocess:
    def __init__(self, config: dict):
        self.event_name = config['event_name']
        self.event_path = config['event_path']
        self.include_spreads = config['Use_spreads']
        self.only_spreads = config['Only_spreads']
        self.returns_file_path = config['returns_file_path']
        self.weights_file_path = config['spread_weights_file_path']
        self.spreads_map = config['bond_futures_spreads_map']
        self.outright_map = config['bond_futures_outrights_map']
        self.start_train = self.parse_train_test_dates(config['date_dict']['train_start'])
        self.end_train = self.parse_train_test_dates(config['date_dict']['train_end'])
        self.start_test = self.parse_train_test_dates(config['date_dict']['test_start'])
        self.end_test = self.parse_train_test_dates(config['date_dict']['test_end'])
        self.all_events = None
        self.prices = None
        self.events = self.load_events()
        self.outright_returns = self.get_outright_returns()

    @staticmethod
    def parse_train_test_dates(key):
        return dt.datetime.strptime(key, '%Y-%m-%d').date()

    def load_events(self):
        event = self.event_name
        events = pd.read_csv(self.event_path, parse_dates=True, index_col=0)
        events[event] = pd.to_datetime(events[event]).dt.date
        self.all_events = list(events[event].values)
        test_events = [i for i in self.all_events if self.start_test <= i <= self.end_test]
        last_train = [i for i in self.all_events if i not in test_events and i <= self.end_train][-1]
        test_events.insert(0, last_train)

        return test_events

    def get_outright_returns(self):
        csv_df = pd.read_csv(self.returns_file_path, index_col=0)
        csv_df.index = pd.to_datetime(csv_df.index, dayfirst=True)
        cols = list(self.outright_map.keys())
        self.prices = csv_df[cols].dropna().rename(mapper=self.outright_map, axis=1)
        df = self.prices.pct_change().dropna()
        mask = df.index > pd.to_datetime(self.start_train)
        return df.loc[mask]

    def get_spread_weights(self):
        weights_df = pd.read_csv(self.weights_file_path, index_col=0)
        weights_df.index = pd.to_datetime(weights_df.index, dayfirst=True)
        weights_df = weights_df.fillna(method='ffill')
        weights_df = weights_df.add_suffix('_weight')
        temp_weights_df = self.outright_returns.merge(weights_df, how='left', left_index=True, right_index=True)
        final_weights = temp_weights_df[[i for i in temp_weights_df.columns if '_weight' in i]].fillna(method='ffill')
        final_weights.columns = final_weights.columns.str.rstrip('_weight')
        return final_weights

    def calc_spread_returns(self):
        spread_dict = self.spreads_map
        all_returns_df = self.outright_returns
        weights_df = self.get_spread_weights()

        if self.include_spreads:
            for spread in spread_dict.keys():
                leg_1_com = spread_dict[spread][0]
                leg_2_com = spread_dict[spread][1]
                leg_1_bond = Auction_params['bond_futures_outrights_map'][leg_1_com]
                leg_2_bond = Auction_params['bond_futures_outrights_map'][leg_2_com]
                all_returns_df[spread] = all_returns_df[leg_2_bond]*(weights_df[leg_1_com]/weights_df[leg_2_com]) - all_returns_df[leg_1_bond]

            all_returns_df = all_returns_df.rename(columns=self.outright_map)
            if self.only_spreads:
                final_cols = list(self.spreads_map.keys())
            else:
                final_cols = list(self.outright_map.values()) + list(self.spreads_map.keys())

        else:
            all_returns_df = all_returns_df.rename(columns=self.outright_map)
            final_cols = list(self.outright_map.values())

        try:
            assert all_returns_df.isna().sum().all() == 0
        except AssertionError:
            print("Final returns data has NaNs")

        return all_returns_df[final_cols]



class AgentExposures:

    def __init__(self, config: dict, events: list):
        self._start = config['date_dict']['train_start']
        self._end = config['date_dict']['test_end'] #t
        self._offset = config['max_event_days_offset']
        self._exposure_t = config['exposure_templates']
        self.events = events

    def create_event_df(self):
        events = self.events
        index = pd.date_range(start=self._start, end=self._end, freq='B')
        df = pd.DataFrame(data=None, index=index)
        df['Event'] = [1 if i in events else 0 for i in df.index]
        df['Distance'] = Event_extraction.distance_to_event_numpy(df['Event'].values)

        val_distance = (df.index.month.value_counts()/len(df.index.year.unique())).max()
        val_start = df.index.get_loc(df.Event.ne(0).idxmax())
        assert df['Distance'].iloc[val_start:].min() > - val_distance
        assert df['Distance'].iloc[val_start:].max() < val_distance

        return df[['Distance']]

    def _get_agent_combos(self):
        return Agent_assembler.generate_trade_profiles(self._exposure_t, self._offset)

    def create_exposure_table(self):
        event_distance_vec = self.create_event_df()
        agents = self._get_agent_combos()
        exposure_tables = all_exposure_tables(agents, event_distance_vec)
        return exposure_tables


def load_vix():
    vix = pd.read_csv(f"../../{DATA_PATH}/VIX_hist.csv", dayfirst=True, index_col=0, parse_dates=True)
    vix = vix['vix_close'][Auction_params['date_dict']['train_start']:]
    vix = vix[vix != '.']

    return vix.apply(pd.to_numeric).fillna(method='ffill')


def get_data(config: dict):
    print("Loading data")
    prepro = Preprocess(config)
    test_events = prepro.events
    all_events = prepro.all_events
    prices = prepro.prices
    raw_returns = prepro.calc_spread_returns()

    try:
        assert raw_returns.index[0] < test_events[0]
    except AssertionError:
        print("Returns data begins after test start --> cutting events without data")
        test_events = [i for i in test_events if i > raw_returns.index[0]]

    print("Creating sparse exposure matrix for clustering")
    exp_generator = AgentExposures(config, all_events)
    exposure_template = exp_generator.create_exposure_table()

    vix = load_vix()

    return test_events, exposure_template, raw_returns[Auction_params['date_dict']['train_start']:], prices, vix


if __name__ == '__main__':
    get_data(Auction_params)
