import Misc.Results_handling as ResultsAnalysis
from Cluster import ClusterAgents, ClusterPostProcessing
import Classifier
from Archive import Old_Forecast_2
import Monetization
from Auction_model.Config import Auction_params
from Logging import logger
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def vix_monitor(event, vix):
    vix_vals = list(vix.iloc[vix.index.get_loc(event) - 10: vix.index.get_loc(event) - 5])
    if any([i > Auction_params['vix_thresh'] for i in vix_vals]):
        return True
    else:
        return False


def walk_forward_predictions(test_events, hist_returns, exposure_template, vix):
    classification_matrix = {'TP': 0,
                             'FP': 0,
                             'TN': 0,
                             'FN': 0}
    all_event_results = {}
    test_events.pop(0)
    for event in test_events:
        if vix_monitor(event, vix):
            continue

        event_results = {}
        print(event)
        logger.info('\n')
        logger.info("{}  ******************************************************************************".format(event))
        spreads = list(Auction_params['bond_futures_spreads_map'].keys())
        outrights = [i + 'R' for i in list(Auction_params['assets'])]
        counter = 0
        for lst in [spreads, outrights]:
            cutoff = hist_returns.index.get_loc(event.strftime("%Y-%m-%d")) - 5
            clustering = ClusterAgents(exposure_template, event, hist_returns[lst][:cutoff], hist_returns[lst][:cutoff])
            assets, agent_info = clustering.cluster_agents()
            cluster_to_classifier = ClusterPostProcessing(clustering)

            for asset in agent_info.keys():
                logger.info("____________________")
                logger.info("{} {}".format(asset, agent_info[asset]))
                day_range = agent_info[asset]['day_range']
                direction = agent_info[asset]['direction']
                x_predict_end = hist_returns.iloc[hist_returns.index.get_loc(event.strftime("%Y-%m-%d")) + (day_range[0] + -1)].name.date()
                dates = cluster_to_classifier.agent_dates[asset]
                classifier = Classifier.create_classifier(dates, hist_returns[asset][:x_predict_end], x_predict_end, direction)
                label, label_probabilities = classifier.predict()

                if label is None:
                    continue

                real_returns, y_true_label = Monetization.get_real_returns(hist_returns[asset], event, day_range, direction)

                logger.info("y_pred: {}, y_true: {}, prob:{}".format(label, y_true_label, label_probabilities))
                logger.info("Training acc: {}".format(classifier.training_score))
                logger.info("ROC AUC: {}".format(classifier.roc_auc))

                #todo
                # if (classifier.training_score < 0.60) | (classifier.roc_auc < 0.55):
                #     continue

                if label_probabilities[1] < Auction_params['classifier_params']['Prob_thresh']:
                    if y_true_label < 0:
                        classification_matrix['TN'] += 1
                    else:
                        classification_matrix['FN'] += 1

                    continue

                logger.info("^TRADED^")
                counter += 1

                if y_true_label > 0:
                    classification_matrix['TP'] += 1
                else:
                    classification_matrix['FP'] += 1

                event_results[asset + " " + str(list(agent_info[asset].values()))] = {"actual": sum(real_returns) * direction,
                                                                                      'label_prob': label_probabilities}

        print(counter)
        logger.info("{}".format(counter))
        if counter > 1:
            top = Old_Forecast_2.get_top_forecasts(event_results)
            final_agents = list(top.keys())
            weighted_PNL = sum([i * (1 / len(top.keys())) for i in top.values()])
        elif counter == 1:
            final_agents = list(event_results.keys())[0]
            weighted_PNL = event_results[final_agents]['actual']
        else:
            final_agents = None
            weighted_PNL = 0

        logger.info("PNL: {}".format(weighted_PNL))

        all_event_results[event] = {'PNL': weighted_PNL,
                                    'Agents': final_agents,
                                    # 'Avg_train_acc': [mean([v['train_acc'] for k, v in event_results.items()])
                                    #                   if final_agents is not None else 0],
                                    'Probabilities': [sum([v['label_prob'] for k, v in event_results.items()])/len(event_results.keys())
                                                      if final_agents is not None else 0],
                                    'raw_results': event_results}

    return pd.DataFrame.from_dict(all_event_results).T, classification_matrix


def run_model(test_events, raw_returns, exposure_template, vix):
    oos_results, confusion_matrix = walk_forward_predictions(test_events, raw_returns, exposure_template, vix)
    total_PNL = oos_results['PNL'].sum()
    print(total_PNL)
    logger.info("PNL: {}".format(total_PNL))
    print(np.sign(oos_results['PNL']).value_counts())
    logger.info("Hit ratio by month: {}".format(np.sign(oos_results['PNL']).value_counts()))

    kpi_dict, f1 = ResultsAnalysis.confusion_matrix_metrics(confusion_matrix)
    yearly_pnl = ResultsAnalysis.calc_yearly_pnl(oos_results)
    print(yearly_pnl)

    _, _, _ = ResultsAnalysis.agent_result_breakdown(oos_results)

    return oos_results, kpi_dict, yearly_pnl

