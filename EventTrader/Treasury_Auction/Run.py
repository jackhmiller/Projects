from Sparks_alpha.Walk_OOS import run_model
import Preprocessing
from Auction_model.Config import Auction_params
from Logging import logger, path
import yaml


def save_config_file():
    dicts = [Auction_params['date_dict'],
             {'event_name': Auction_params['event_name']},
             {'Use_spreads': Auction_params['Use_spreads']},
             {'Only_spreads': Auction_params['Only_spreads']},
             Auction_params['classifier_params']]

    config_dict = {}
    for d in dicts:
        config_dict.update(d)

    with open('{}/config.yaml'.format(path), 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)


def main():
    save_config_file()

    test_events, exposure_template, hist_returns, prices, vix = Preprocessing.get_data(Auction_params)
    results_df, classification_metrics, yearly_pnl = run_model(test_events, hist_returns, exposure_template, vix)

    logger.info("Yearly breakdown: {}".format(yearly_pnl))
    logger.info("Classification metrics: {}".format(classification_metrics))
    results_df.to_csv('{}/OOS_results.csv'.format(path))

    return


if __name__ == '__main__':
    main()
