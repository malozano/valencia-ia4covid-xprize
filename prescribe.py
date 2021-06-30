# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import argparse
import numpy as np
import pandas as pd

from copy import deepcopy
from datetime import datetime

import ast
import math
import glob
import time

# Function imports from utils
from pathlib import Path

from covid_xprize.examples.prescriptors.neat.utils import add_geo_id
from covid_xprize.examples.prescriptors.neat.utils import get_predictions
from covid_xprize.examples.prescriptors.neat.utils import load_ips_file
from covid_xprize.examples.prescriptors.neat.utils import prepare_historical_df

# Constant imports from utils
from covid_xprize.examples.prescriptors.neat.utils import CASES_COL
from covid_xprize.examples.prescriptors.neat.utils import IP_COLS
from covid_xprize.examples.prescriptors.neat.utils import IP_MAX_VALUES
from covid_xprize.examples.prescriptors.neat.utils import PRED_CASES_COL
from covid_xprize.examples.prescriptors.neat.utils import HIST_DATA_FILE_PATH


# Imports for fast version
from covid_xprize.validation.scenario_generator import get_raw_data, generate_scenario
from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor
from valencia_predictor import ValenciaPredictor


IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}

FEAT_IMPORTANCE = [0.16253311, 0.33043205, 0.02189676, 0.06947346, 0.05884782, 0.03694285,
 0.0307605,  0.07734699, 0.05564001, 0.10437437, 0.0270645,  0.02468759]


# Path to where this script lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Number of prescriptions to make per country.
# This can be set based on how many solutions in PRESCRIPTORS_FILE
# we want to run and on time constraints.
NB_PRESCRIPTIONS = 10
NB_GREEDY_FEAT_PRESCRIPTIONS = 10
NB_DYNAMIC_PRESCRIPTIONS = 10

# Number of days to fix prescribed IPs before changing them.
# This could be a useful toggle for decision makers, who may not
# want to change policy every day. Increasing this value also
# can speed up the prescriptor, at the cost of potentially less
# interesting prescriptions.
ACTION_DURATION = 15

WINDOW_SIZE = 7
COMBINED_NPI_FILES = os.path.join(ROOT_DIR, 'data/MasterYodaNPI.csv')
MAX_PREDICTION_DAYS = 30

DYNAMIC_SCENARIO_MIN_DAYS = 30

USE_VALENCIA_PREDICTOR = False
INCLUDE_CV_PREDICTION = False
CV_FILE = "data/OxfordComunitatValenciana.csv"
GEO_CV_FILE = "countries_regions_cv.csv"


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    pred_end_date_str = end_date_str
    prescription_days = (end_date - start_date).days

    # Limit to maximum prediction days
    if prescription_days > MAX_PREDICTION_DAYS:
        pred_end_date = start_date + np.timedelta64(MAX_PREDICTION_DAYS, 'D')
        pred_end_date_str = pred_end_date.strftime("%Y-%m-%d")
        
    
    # Load the past IPs data
    print("Loading past IPs data...")
    past_ips_df = load_ips_file(path_to_prior_ips_file)
    geos = past_ips_df['GeoID'].unique()

    # Preload model
    if USE_VALENCIA_PREDICTOR:
        predictor = ValenciaPredictor()
    else:
        predictor = XPrizePredictor()

    predictor_df = predictor.df[predictor.df.Date < start_date_str]
    # Load the historical data
    prepare_historical_df()
    hist_df = preload_historical_df(start_date_str, end_date_str, geos)

    # Load IP costs to condition prescriptions
    cost_df = pd.read_csv(path_to_cost_file)
    cost_df['RegionName'] = cost_df['RegionName'].fillna("")
    cost_df = add_geo_id(cost_df)
    geo_costs = {}
    for geo in geos:
        costs = cost_df[cost_df['GeoID'] == geo]
        cost_arr = np.array(costs[IP_COLS])[0]
        geo_costs[geo] = cost_arr

    combined_npi_df = load_npi_df()

    geo_front_dfs = dict()
    prescription_dfs = []
    for geo in geos:
        print('Prescribing for', geo)
        
        #country_name, region_name = geo.split('__')
        #geo_id = country_name if region_name=='' else country_name + ' / ' + region_name

        # Get combined pareto front (Rn, total_cases, t20_cases)

        combined_npi_df = apply_weights(combined_npi_df, geo_costs[geo])

        geo_front_df_1 = get_pareto_front(combined_npi_df, stringency_col="Stringency", cases_col="Rnavg")
        geo_front_df_2 = get_pareto_front(combined_npi_df, stringency_col="Stringency", cases_col="total_cases")
        geo_front_df_3 = get_pareto_front(combined_npi_df, stringency_col="Stringency", cases_col="t20_cases")
        geo_front_df_4 = get_pareto_front(combined_npi_df, stringency_col="FeatImportance", cases_col="Rnavg")

        geo_front_df = geo_front_df_1.merge(geo_front_df_2[IP_COLS], how='inner', on=IP_COLS)
        geo_front_df = geo_front_df.merge(geo_front_df_3[IP_COLS], how='inner', on=IP_COLS)
        geo_front_df = geo_front_df.merge(geo_front_df_4[IP_COLS], how='inner', on=IP_COLS)

        # Add feature greedy IPs

        geo_pres_front_df = geo_front_df[IP_COLS]
        geo_pres_greedy_df = feature_greedy_prescription(geo_costs[geo])
        geo_pres_df = pd.concat([geo_pres_front_df, geo_pres_greedy_df])
        geo_pres_df = geo_pres_df.drop_duplicates()
        
        # Evaluate all the candidates

        geo_hist_df = hist_df[hist_df.GeoID==geo]
        
        total_cases = []
        for npi_idx in range(geo_pres_df.shape[0]):
            npi_array = [int(geo_pres_df[col].values[npi_idx]) for col in IP_COLS]
            pres_df = make_prescriptions(geo, start_date_str, end_date_str, npi_array, npi_idx)
            pred_df = get_prediction(predictor, predictor_df, start_date_str, pred_end_date_str, geo_hist_df, pres_df)
            total_cases.append(pred_df['PredictedDailyNewCases'].sum())
        
        geo_pres_df = apply_weights(geo_pres_df, geo_costs[geo])
        geo_pres_df['TotalCases'] = total_cases
        geo_pres_df = get_pareto_front(geo_pres_df, stringency_col="Stringency", cases_col="TotalCases")

        # Select final static prescriptions

        geo_pres_df['Score'] = compute_scores(geo_pres_df)
        #geo_pres_df.to_csv("pareto-front-{}.csv".format(geo))

        geo_front_dfs[geo] = geo_pres_df
        
        selected_df = select_best_prescriptions(geo_pres_df)
        
        # Dynamic scenario
        
        if prescription_days > DYNAMIC_SCENARIO_MIN_DAYS:
            
            # Select NPI combinations
            
            selected_comb_df = get_combinations(selected_df)

            selected_comb_front_df = get_pareto_front(selected_comb_df, stringency_col="Stringency", 
                                                                        cases_col="TotalCases", group_by_ip=False)
            selected_comb_front_df['Score'] = compute_scores(selected_comb_front_df)

            selected_comb_single_df = selected_comb_front_df[(selected_comb_front_df.i==selected_comb_front_df.j) & 
                                                             (selected_comb_front_df.i==selected_comb_front_df.k)]
            selected_comb_multi_df = selected_comb_front_df.drop(index=selected_comb_single_df.index.values)
            
            nb_dynamic_prescriptions = min(NB_DYNAMIC_PRESCRIPTIONS, selected_comb_multi_df.shape[0])
            selected_comb_multi_df.sort_values(['Score'], ascending = False, inplace = True,ignore_index=True)
            selected_comb_multi_df = selected_comb_multi_df.iloc[0:nb_dynamic_prescriptions,:]
            
            # Evaluate selected dynamic prescriptions
            
            total_cases = []
            for idx in range(nb_dynamic_prescriptions):
                npi_array = [[int(selected_df[col].values[selected_comb_multi_df.i.values[idx]]) for col in IP_COLS],
                             [int(selected_df[col].values[selected_comb_multi_df.j.values[idx]]) for col in IP_COLS],
                             [int(selected_df[col].values[selected_comb_multi_df.k.values[idx]]) for col in IP_COLS]]
                pres_df = make_dynamic_prescriptions(geo, start_date_str, pred_end_date_str, npi_array, idx)
                pred_df = get_prediction(predictor, predictor_df, start_date_str, pred_end_date_str, geo_hist_df, pres_df)
                total_cases.append(pred_df['PredictedDailyNewCases'].sum())
            selected_comb_multi_df.TotalCases = total_cases

            # Select final dynamic prescriptions

            selected_comb_all_df = pd.concat([selected_comb_single_df, selected_comb_multi_df])
            selected_comb_all_df = get_pareto_front(selected_comb_all_df, stringency_col="Stringency", 
                                                                          cases_col="TotalCases", group_by_ip=False)            
            selected_comb_all_df['Score'] = compute_scores(selected_comb_all_df)
            selected_comb_final_df = select_best_prescriptions(selected_comb_all_df)
            
            # Generate final static prescriptions

            step = (selected_comb_final_df.shape[0]-1)/(NB_PRESCRIPTIONS-1)
            for npi_idx in range(NB_PRESCRIPTIONS):
                selected_npi = round(npi_idx*step)
                npi_array = [[int(selected_df[col].values[selected_comb_final_df.i.values[selected_npi]]) for col in IP_COLS],
                             [int(selected_df[col].values[selected_comb_final_df.j.values[selected_npi]]) for col in IP_COLS],
                             [int(selected_df[col].values[selected_comb_final_df.k.values[selected_npi]]) for col in IP_COLS]]
                pres_df = make_dynamic_prescriptions(geo, start_date_str, end_date_str, npi_array, npi_idx)
                prescription_dfs.append(pres_df)
            
        else:
            
            # Generate final static prescriptions

            step = (selected_df.shape[0]-1)/(NB_PRESCRIPTIONS-1)
            for npi_idx in range(NB_PRESCRIPTIONS):
                selected_npi = round(npi_idx*step)
                npi_array = [int(selected_df[col].values[selected_npi]) for col in IP_COLS]
                pres_df = make_prescriptions(geo, start_date_str, end_date_str, npi_array, npi_idx)
                prescription_dfs.append(pres_df)

    prescription_df = pd.concat(prescription_dfs)

    # Create the output directory if necessary.
    output_dir = os.path.dirname(output_file_path)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    # Save to a csv file
    prescription_df.to_csv(output_file_path, index=False)
    print('Prescriptions saved to', output_file_path)

    return geo_front_dfs


def make_dynamic_prescriptions(geo, start_date_str, end_date_str, npi_comb, pres_idx):
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    nb_prescription_days = (end_date - start_date).days
    nb_prescription_segments = len(npi_comb)
    nb_segment_days = math.ceil(nb_prescription_days / nb_prescription_segments)
    
    pres_dfs = []
    segment_start_date = start_date
    for i in range(nb_prescription_segments):
        segment_end_date = min(segment_start_date + np.timedelta64(nb_segment_days, 'D'), end_date)
        pres_df = make_prescriptions(geo, 
                                     segment_start_date.strftime("%Y-%m-%d"), 
                                     segment_end_date.strftime("%Y-%m-%d"), 
                                     npi_comb[i], 
                                     pres_idx)
        pres_dfs.append(pres_df)
        segment_start_date = segment_end_date + np.timedelta64(1, 'D')
        
    all_pres_df = pd.concat(pres_dfs)
    return all_pres_df
    

def make_prescriptions(geo, start_date_str, end_date_str, npi_comb, pres_idx):

    country_name, region_name = geo.split('__')
    if region_name == 'nan':
        region_name = np.nan

    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    dates = []
    for date in pd.date_range(start_date, end_date):
        dates.append(date)

    prescription_df = pd.DataFrame({'Date': dates})
    prescription_df['CountryName'] = country_name
    prescription_df['RegionName'] = region_name
    prescription_df['PrescriptionIndex'] = pres_idx

    for ip_col, prescribed_ip in zip(IP_COLS, npi_comb):
        prescription_df[ip_col] = prescribed_ip

    return prescription_df


def select_best_prescriptions(front_df, nb_prescriptions=NB_PRESCRIPTIONS):
    if front_df.shape[0] <= NB_PRESCRIPTIONS:
        return front_df

    df = front_df.reset_index(drop=True)
    df.Stringency = df.Stringency.apply(lambda x: (x - df.Stringency.min()) / (df.Stringency.max() - df.Stringency.min()) )
    original_points = df.copy()

    df = df.drop(df[df.Stringency == df.Stringency.min()].index).reset_index(drop=True)
    his = np.histogram(df.Stringency)

    w =0.1
    a_window_size = w * 1 / (his[0] + 1)
    n_points = df.shape[0]
    drop_list = set()
    i = 0
    while n_points > nb_prescriptions-1:
        for _, row in df.iterrows():
            window_size = a_window_size[int(row["Stringency"] // 0.1)]        
            window_lower = row["Stringency"] - window_size / 2
            window_upper = row["Stringency"] + window_size / 2
            selection = df[(df.Stringency >= window_lower) & (df.Stringency <= window_upper)]
            if selection.shape[0] > 1:
                to_drop = selection[selection.Score != selection.Score.max()].index.values
            else:
                to_drop = set()
            drop_list.update(to_drop)
        drop_list = list(drop_list)
        df = df.drop(index=drop_list)
        df = df.reset_index(drop=True)
        drop_list = set()
        n_points = df.shape[0]
        w += 0.01
        a_window_size = w * 1 / (his[0] + 1)
        i += 1
        
    # Identify selected 
    df3 = pd.concat([df,original_points]).reset_index(drop=True)
    df3["front"] = False
    df3.loc[df3.Stringency == df3.Stringency.min(),"front"] = True
    df3.loc[df3.duplicated(subset=["TotalCases","Score"],keep=False),"front"] = True
    df3 = df3.drop_duplicates(subset=["TotalCases","Score"],keep="last").reset_index(drop=True)
    df3.loc[(df3.Score < df3.Score.quantile(0.25)) & (df3.Stringency < 0.6),"front"] = False

    selected_df = df3[df3.front].reset_index(drop=True)

    return selected_df


def get_combinations(front_df):
    num_npi = front_df.shape[0]

    total_cases = front_df.TotalCases.values
    stringency = front_df.Stringency.values

    comb_dict = {
        'i' : [],
        'j' : [],
        'k' : [],
        'TotalCases' : [],
        'Stringency' : []
    }

    for i in range(num_npi):
        for j in range(num_npi):
            for k in range(num_npi):
                comb_dict['i'].append(i)
                comb_dict['j'].append(j)
                comb_dict['k'].append(k)
                mean_total_cases = (total_cases[i] + total_cases[j] + total_cases[k]) / 3.0
                mean_stringency = (stringency[i] + stringency[j] + stringency[k]) / 3.0
                comb_dict['TotalCases'].append(mean_total_cases)
                comb_dict['Stringency'].append(mean_stringency)

    return pd.DataFrame(comb_dict)
                


def compute_scores(pareto_front_df, order_column='Stringency'):
    
    pareto_front_df.sort_values([order_column], ascending = True, inplace = True,ignore_index=True)
    points = np.array(list(zip(pareto_front_df.Stringency.values, pareto_front_df.TotalCases.values)))

    min_str = np.min(points[:,0])
    max_str = np.max(points[:,0])
    min_cases = np.min(points[:,1])
    max_cases = np.max(points[:,1])

    height = max_str-min_str
    width = max_cases-min_cases

    if width<=0 or height<=0:
        return [1] * len(points)

    points[:,0] = (points[:,0] - min_str) / height
    points[:,1] = (points[:,1] - min_cases) / width

    curr_point = points[0]
    next_point = points[1]
    scores = [ 1 ]

    for index in range(2, len(points)):
        prev_point = curr_point
        curr_point = next_point
        next_point = points[index]

        v0 = next_point - curr_point
        v1 = prev_point - curr_point

        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)) 

        if angle<0: angle = angle + 2*math.pi
        score = 1.0 - (angle - math.pi/2) / math.pi
        scores.append(score)

    scores.append(1)

    return scores

    
def load_npi_df():
    df = pd.read_csv(COMBINED_NPI_FILES)    
    return df


def apply_weights(npi_df, ip_weights):
    ip_weights_feat = ip_weights/np.asarray(FEAT_IMPORTANCE)

    weight_df = pd.DataFrame(pd.Series(ip_weights, index=IP_COLS, name=0))
    weight_feat_df = pd.DataFrame(pd.Series(ip_weights_feat, index=IP_COLS, name=0))

    npi_df['Stringency'] = npi_df.loc[:,IP_COLS].dot(weight_df)
    npi_df['FeatImportance'] = npi_df.loc[:,IP_COLS].dot(weight_feat_df)

    return npi_df


def load_raw_data(data_file):
    """
    Returns the raw data from which to generate scenarios.
    Args:
        cache_file: the file to use to cache the data
        latest: True to force a download of the latest data and update cache_file,
                False to get the data from cache_file

    Returns: a Pandas DataFrame

    """
    latest_df = pd.read_csv(data_file,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    # Fill any missing NPIs by assuming they are the same as previous day, or 0 if none is available
    latest_df.update(latest_df.groupby(['CountryName', 'RegionName'])[IP_COLS].ffill().fillna(0))
    return latest_df


def preload_historical_df(start_date_str, end_date_str, geos):
    # Preload historical IPS
    raw_df = get_raw_data(HIST_DATA_FILE_PATH)
    if INCLUDE_CV_PREDICTION:
        raw_cv_df = load_raw_data(CV_FILE)
        raw_df = raw_df.append(raw_cv_df)

    hist_df = generate_scenario(start_date_str, end_date_str, raw_df,
                                    countries=None, scenario='Freeze')
    hist_df = hist_df[hist_df.Date < start_date_str]
    hist_df["GeoID"] = hist_df["CountryName"] + '__' + hist_df["RegionName"]
    hist_df = hist_df[hist_df.GeoID.isin(geos)]

    return hist_df


def get_prediction(predictor, predictor_df, start_date_str, end_date_str, hist_df, pres_df):
    ips_df = pd.concat([hist_df, pres_df])
    ips_df['GeoID'] = np.where(ips_df["RegionName"]=='',
                                ips_df["CountryName"],
                                ips_df["CountryName"] + ' / ' + ips_df["RegionName"])

    predictor.df = predictor_df.copy()
    pred_df = predictor.predict_from_df(start_date_str, end_date_str, ips_df)

    return pred_df


# Get feature greedy NPIs
def feature_greedy_prescription(ip_weights):

    # Multiply ip_weights by FEAT_IMPORTANCE
    ip_weights = ip_weights/np.asarray(FEAT_IMPORTANCE)
    sorted_ips = [ip for _, ip in sorted(zip(ip_weights, IP_COLS))]

    # Initialize the IPs to all turned off
    curr_ips = {ip: 0 for ip in IP_COLS}

    # Initialize empty dict for IPs
    prescription_dict = dict()
    for ip in IP_COLS:
        prescription_dict[ip] = []

    for prescription_idx in range(NB_GREEDY_FEAT_PRESCRIPTIONS):

        # Turn on the next IP
        next_ip = sorted_ips[prescription_idx]
        curr_ips[next_ip] = IP_MAX_VALUES[next_ip]

        # Use curr_ips for all dates for this prescription
        for ip in IP_COLS:
            prescription_dict[ip].append(curr_ips[ip])

    prescription_df = pd.DataFrame(prescription_dict)
    return prescription_df


# Calculate pareto front
def get_pareto_front(npi_df, stringency_col='Stringency', cases_col = "Rnavg", group_by_ip=True):
    
    if group_by_ip:
        # Group by NPI - Stringency and compute the mean
        df = npi_df.groupby(IP_COLS, as_index=False).median()
    else:
        df = npi_df.copy()

    # Sort df by Stringency
    df.sort_values([stringency_col, cases_col], ascending = True, inplace = True,ignore_index=True)

    min_value = math.inf
    selected = []
    for value in df[cases_col].values:
        if value < min_value:
            min_value = value
            selected.append(True)
        else:
            selected.append(False)
    
    df["Selected"] = selected

    return df[df.Selected==True]


    
# Calculates simulated cases
def simulate(actual_df: pd.DataFrame, start_date_str: str, end_date_str: str, Ravg: float, geos=None) -> pd.DataFrame:
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    nb_days = (end_date - start_date).days + 1

    # Prepare the output
    forecast = {"CountryName": [],
                  "RegionName": [],
                  "GeoID": [],
                  "Date": [],
                  "Rn": [],
                  "NewCases": [],
                  "ConfirmedCases": [],
                  "Population": []}

    # For each requested geo
    if geos is None:
        geos = actual_df.GeoID.unique()
    for g in geos:
        cdf = actual_df[actual_df["GeoID"] == g]
        if len(cdf) == 0:
            # we don't have historical data for this geo: return zeroes
            simulated_new_cases = [0] * nb_days
            geo_start_date = start_date
        else:
            last_known_date = cdf.Date.max()
            # Start predicting from start_date, unless there's a gap since last known date
            geo_start_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)
            # Simulate with the formula (list)
            nb_sim_days = (end_date - geo_start_date).days + 1
            simulated_new_cases = get_new_cases_preds(cdf, [Ravg] * nb_sim_days, geo_start_date, end_date)

        # Append forecast data to results to return
        country = cdf["CountryName"].values[0]
        region = cdf["RegionName"].values[0]
        population = cdf["Population"].values[0]
        cases = cdf["ConfirmedCases"].values[-1]
        
        for i,pred in enumerate(simulated_new_cases):
            forecast["CountryName"].append(country)
            forecast["RegionName"].append(region)
            forecast["GeoID"].append(g)
            current_date = geo_start_date + pd.offsets.Day(i)
            forecast["Date"].append(current_date)
            forecast["Rn"].append(Ravg)
            forecast["NewCases"].append(pred)
            forecast["Population"].append(population)
            cases = cases + pred
            forecast["ConfirmedCases"].append(cases)

    forecast_df = pd.DataFrame.from_dict(forecast)
    # Return only the requested predictions
    return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]
            
    
def get_new_cases_preds(c_df, ratios, start_date, end_date):
    #cdf = c_df[c_df.ActualDailyNewCases.notnull() & c_df.ActualDailyNewCases7DMA.notnull() ]
    cdf = c_df[c_df.NewCases.notnull() & c_df.ConfirmedCases.notnull() ]    
    # Gather info to convert to total cases
    # Comfirmed: Accumulated
    prev_confirmed_cases = np.array(cdf.ConfirmedCases)
    prev_new_cases = np.array(cdf.NewCases)

    if(len(prev_confirmed_cases) < 1 or len(prev_new_cases) < WINDOW_SIZE):
        return [0] * len(ratios)

    initial_total_cases = prev_confirmed_cases[-1]
    pop_size = np.array(cdf.Population)[-1]  # Population size doesn't change over time
    # Compute predictor's forecast
    pred_new_cases = convert_ratios_to_total_cases(
        ratios,
        WINDOW_SIZE,
        prev_new_cases,
        initial_total_cases,
        pop_size)

    return pred_new_cases


# From Rns to cases
def convert_ratio_to_new_cases(ratio, window_size, prev_new_cases_list, prev_pct_infected):
    return (ratio * (1 - prev_pct_infected) - 1) * \
               (window_size * np.mean(prev_new_cases_list[-window_size:])) \
               + prev_new_cases_list[-window_size]


def convert_ratios_to_total_cases(ratios, window_size, prev_new_cases, initial_total_cases, pop_size):
    new_new_cases = []
    prev_new_cases_list = list(prev_new_cases)
    curr_total_cases = initial_total_cases
    # Prepare the loop
    for ratio in ratios:
        new_cases = convert_ratio_to_new_cases(ratio, window_size, prev_new_cases_list, curr_total_cases / pop_size)
        # new_cases can't be negative!
        new_cases = max(0, new_cases)
        # Which means total cases can't go down
        curr_total_cases += new_cases
        # Update prev_new_cases_list for next iteration of the loop
        prev_new_cases_list.append(new_cases)
        new_new_cases.append(new_cases)
    return new_new_cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prior_ips_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prior_ips_file, args.cost_file, args.output_file)
    print("Done!")
