import pandas as pd
import re
from collections import Counter
from ast import literal_eval


def agent_result_breakdown(df):
    agents = [i for i in df['Agents'].apply(pd.Series).values.flatten() if type(i) == str]
    asset_count = Counter([i.split(' [')[0] for i in agents])
    sorted_asset_count = {k: v for k, v in sorted(asset_count.items(), key=lambda item: item[1])}
    directions = Counter([i.split(' [')[1][0] for i in agents])
    range_tuples = Counter([literal_eval(i.split('R ')[1])[1] for i in agents])
    range_tuples_count = {k: v for k, v in sorted(range_tuples.items(), key=lambda item: item[1])}

    range_pnl = []
    for k, r in df.iterrows():
        range_pnl.append({re.search('\(.*?\)', k)[0]: round(v['actual'] * 100, 2) for k, v in r['raw_results'].items()})

    breakdown = {}
    for asset in list(set([i.split(' [')[0] + ' ' for i in agents])):
        asset_dict = Counter([literal_eval(i.split('R ')[1])[1] for i in agents if asset in i])
        breakdown[asset] = asset_dict

    return sorted_asset_count, range_tuples_count, breakdown


def calc_yearly_pnl(df):
    df.index = pd.to_datetime(df.index)
    df['year'] = df.index.year

    return df.groupby('year')['PNL'].sum()


def confusion_matrix_metrics(results: dict):
    precision = results['TP']/(results['TP'] + results['FP'])
    sensitivity = results['TP']/(results['TP'] + results['FN'])
    specificity = results['TN']/(results['TN'] + results['FP'])
    accuracy = (results['TP'] + results['TN'])/sum(results.values())

    kpi_dict = {'Precision': precision,
               'Sensitivity': sensitivity,
               'Specificity': specificity,
               'Accuracy': accuracy
               }

    f1 = 2*((precision*sensitivity)/(precision+sensitivity))

    return kpi_dict, f1
