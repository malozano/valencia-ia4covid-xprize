from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import Constraint
from keras.constraints import NonNeg

import os
import csv 
import ast
import pandas as pd
import numpy as np

import datetime


# Suppress noisy Tensorflow debug logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

INCLUDE_CV_PREDICTION = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')
DATA_FILE_CV_PATH = os.path.join(DATA_PATH, 'OxfordComunitatValenciana.csv')
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = os.path.join(DATA_PATH, "brazil_populations.csv")

MODEL_PATH = os.path.join(ROOT_DIR, 'models')
MODEL_WEIGHTS_CLUSTER_FILE = os.path.join(MODEL_PATH, "weightscluster{}_280traineddays.h5")
MODEL_WEIGHTS_SCO_V0_FILE = os.path.join(MODEL_PATH, "sco_v0_trained_model_weights.h5")
MODEL_WEIGHTS_SCO_V1_FILE = os.path.join(MODEL_PATH, "sco_v1_trained_model_weights.h5")
MODEL_WEIGHTS_SCO_V2_FILE = os.path.join(MODEL_PATH, "sco_v2_trained_model_weights.h5")

ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COL = ['NewCases']
CONTEXT_COLUMNS = ['CountryName',
                   'RegionName',
                   'GeoID',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population']
NPI_COLUMNS = ["C1_School closing",
            "C2_Workplace closing",
            "C3_Cancel public events",
            "C4_Restrictions on gatherings",
            "C5_Close public transport",
            "C6_Stay at home requirements",
            "C7_Restrictions on internal movement",
            "C8_International travel controls"]
NB_LOOKBACK_DAYS = 21
WINDOW_SIZE = 7
LSTM_SIZE = 32
US_PREFIX = "United States / "


Cluster_1 = [('Central African Republic', ''),('Chile', ''),('China', ''),('Lithuania', ''),('Niger', ''),('Panama', ''),
             ('Sweden', ''),('Switzerland', ''),('United States', 'Arizona'),('United States', 'Hawaii'),
             ('United States', 'Maine'),('United States', 'Rhode Island')]
Cluster_2 = [('Bahrain', ''),('Bangladesh', ''),('El Salvador', ''),('Estonia', ''),('Japan', ''),('Kosovo', ''),
             ('Luxembourg', ''),('Moldova', ''),('Peru', ''),('Vietnam', '')]
Cluster_3 = [('Andorra', ''),('Aruba', ''),('Australia', ''),('Belarus', ''),('Belgium', ''),('Bolivia', ''),
             ('Bulgaria', ''),('Burkina Faso', ''),('Croatia', ''),("Cote d'Ivoire", ''),('Czech Republic', ''),
             ('Dominican Republic', ''),('Finland', ''),('France', ''),('Greece', ''),('Guatemala', ''),('Iceland', ''),
             ('India', ''),('Ireland', ''),('Israel', ''),('Kosovos', ''),('Latvia', ''),('Mongolia', ''),('Myanmar', ''),
             ('Nepal', ''),('Norway', ''),('Oman', ''),('Puerto Rico', ''),('Romania', ''),('Russia', ''),('Saudi Arabia', ''),
             ('Slovenia', ''),('Tajikistan', ''),('Trinidad and Tobago', ''),('Uganda', ''),('Ukraine', ''),
             ('United Arab Emirates', ''),('United States', 'California'),('United States', 'Georgia'),
             ('United States', 'Idaho'),('United States', 'New Hampshire'),('United States', 'North Carolina'),('Uruguay', ''),
             ('Venezuela', ''),('Zambia', '')] 
Cluster_4 = [('United States', 'South Carolina')]
Cluster_6 = [('Cameroon', ''),('Ethiopia', ''),('Jordan', ''),('Uzbekistan', ''),('Zimbabwe', '')]
Cluster_7 = [('Eswatini', ''),('Kenya', ''),('Libya', ''),('Singapore', ''),('Suriname', ''),('United States', 'Illinois')]
Cluster_10 = [('Algeria', ''), ('Iran', ''), ('Morocco', ''), ('United States', 'Texas')]
Cluster_11 = [('United States', 'Florida')]
Cluster_v0 = [ ('Afghanistan', ''), ('Bahamas', ''), ('Azerbaijan', ''), ('Burundi', ''), ('Comoros', ''), 
            ('Democratic Republic of Congo', ''), ('Hong Kong', ''), ('Indonesia', ''), ('Kazakhstan', ''), 
            ('Kyrgyz Republic', ''), ('Mauritius', ''), ('New Zealand', ''), ('Nicaragua', ''), ('Sudan', ''), 
            ('Taiwan', '')]


class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)

class ValenciaPredictor(object):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self):
        # Carga el modelo y sus pesos
        # self.model = self._create_model_default(MODEL_WEIGHTS_DEFAULT_FILE)
        nb_context = 1  # Only time series of new cases rate is used as context
        nb_action = len(NPI_COLUMNS)    
        self.model_v0 = self._create_model_sco_v0(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
        self.model_v0.load_weights(MODEL_WEIGHTS_SCO_V0_FILE)
        self.model_v1 = self._create_model_sco_v1(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
        self.model_v1.load_weights(MODEL_WEIGHTS_SCO_V1_FILE)
        self.model_v2 = self._create_model_sco_v2(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
        self.model_v2.load_weights(MODEL_WEIGHTS_SCO_V2_FILE)
        self.cluster_dict = self._load_clusters()
        self.df = self._prepare_dataframe()


    def predict_df(self, start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
        # Load historical intervention plans, since inception
        hist_ips_df = self._load_original_data(path_to_ips_file)
        return self.predict_from_df(start_date_str, end_date_str, hist_ips_df, verbose=verbose)


    def predict_from_df(self,
                        start_date_str: str,
                        end_date_str: str,
                        npis_df: pd.DataFrame, 
                        verbose=False) -> pd.DataFrame:
        """
        Generates a file with daily new cases predictions for the given countries, regions and npis, between
        start_date and end_date, included.
        :param start_date_str: day from which to start making predictions, as a string, format YYYY-MM-DDD
        :param end_date_str: day on which to stop making predictions, as a string, format YYYY-MM-DDD
        :param path_to_ips_file: path to a csv file containing the intervention plans between inception_date and end_date
        :param verbose: True to print debug logs
        :return: a Pandas DataFrame containing the predictions
        """
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        nb_days = (end_date - start_date).days + 1

        # Load historical intervention plans, since inception
        hist_ips_df = npis_df
        # Fill any missing NPIs by assuming they are the same as previous day
        for npi_col in NPI_COLUMNS:
            hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

        # Intervention plans to forecast for: those between start_date and end_date
        ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]
        
        # Make predictions for each country,region pair
        geo_pred_dfs = []
        for g in ips_df.GeoID.unique():
            if verbose:
                print('\nPredicting for', g)
                        
            # Pull out all relevant data for country c
            ips_gdf = ips_df[ips_df.GeoID == g]
            hist_ips_gdf = hist_ips_df[hist_ips_df.GeoID == g]
            hist_cases_gdf = self.df[self.df.GeoID == g]
            last_known_date = hist_cases_gdf.Date.max()
            
            # Start predicting from start_date, unless there's a gap since last known date
            current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)

            past_cases_gdf = hist_cases_gdf[hist_cases_gdf.Date < current_date]
            past_ips_gdf = hist_ips_gdf[hist_ips_gdf.Date < current_date]
            future_ips_gdf = hist_ips_gdf[(hist_ips_gdf.Date >= current_date) & (hist_ips_gdf.Date <= end_date)]

            past_cases = np.array(past_cases_gdf[CASES_COL]).flatten()
            past_npis = np.array(past_ips_gdf[NPI_COLUMNS])
            future_npis = np.array(future_ips_gdf[NPI_COLUMNS])

            pop_size = hist_cases_gdf.Population.max()
            
            past_cum_cases = np.cumsum(past_cases)
            zn = np.array(compute_7days_mean(past_cases))
            rn = np.array(compute_rns(past_cum_cases, zn, pop_size))

            # Loads custom model
            cluster_id = self.cluster_dict.get(g)
            if cluster_id is None:
                current_model = self.model_v1
            elif cluster_id == -1:
                current_model = self.model_v0
            else:
                file_name = MODEL_WEIGHTS_CLUSTER_FILE.format(cluster_id)
                current_model = self._create_model(file_name)

            # Make prediction for each day
            geo_preds = []
            geo_ratios = []
            days_ahead = 0
            while current_date <= end_date:
                # Prepare data
                X_rns = rn[-NB_LOOKBACK_DAYS:].reshape(1, 21, 1)
                X_npis = past_npis[-NB_LOOKBACK_DAYS:].reshape(1, 21, 8)
                
                # Make the prediction (reshape so that sklearn is happy)
                pred_rn = current_model.predict([X_rns, X_npis])[0][0]
                pred_cases = int(((((pred_rn * ((pop_size - past_cum_cases[-1]) / pop_size)) - 1.0) * 7.0 * zn[-1])) + past_cases[-7])

                pred = max(0, pred_cases)  # Do not allow predicting negative cases
                # Add if it's a requested date
                if current_date >= start_date:
                    geo_preds.append(pred)
                    geo_ratios.append(pred_rn)
                    if verbose:
                        print(f"{current_date.strftime('%Y-%m-%d')}: {pred}")
                else:
                    if verbose:
                        print(f"{current_date.strftime('%Y-%m-%d')}: {pred} - Skipped (intermediate missing daily cases)")

                # Append the prediction and npi's for next day
                # in order to rollout predictions for further days.
                past_cases = np.append(past_cases, pred)
                past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + 1], axis=0)
                past_cum_cases = np.append(past_cum_cases, past_cum_cases[-1] + pred)
                zn = np.append(zn, compute_last_7days_mean(past_cases))
                rn = np.append(rn, pred_rn) # compute_last_rn(past_cum_cases, zn, pop_size)

                # Move to next day
                current_date = current_date + np.timedelta64(1, 'D')
                days_ahead += 1

            # we don't have historical data for this geo: return zeroes
            if len(geo_preds) != nb_days:
                geo_preds = [0] * nb_days
                geo_ratios = [0] * nb_days
 
            if g=='Mauritania':
                geo_preds = [140] * nb_days
                geo_ratios = [0] * nb_days
            if g=='Yemen':
                geo_preds = [5] * nb_days
                geo_ratios = [0] * nb_days

            # Create geo_pred_df with pred column
            geo_pred_df = ips_gdf[ID_COLS].copy()
            geo_pred_df['PredictedDailyNewCases'] = geo_preds
            geo_pred_df['PredictedDailyNewRatios'] = geo_ratios
            geo_pred_dfs.append(geo_pred_df)

            
        # Combine all predictions into a single dataframe
        pred_df = pd.concat(geo_pred_dfs)

        # Drop GeoID column to match expected output format
        pred_df = pred_df.drop(columns=['GeoID'])

        return pred_df


    def _load_cluster(self, country_list, cluster_id, cluster_dict):
        for country, region in country_list:
            geo_id = country if region=='' else "{} / {}".format(country, region)
            cluster_dict[geo_id] = cluster_id
 

    def _load_clusters(self):
        cluster_dict = dict()
        self._load_cluster(Cluster_1, 1, cluster_dict)
        self._load_cluster(Cluster_2, 2, cluster_dict)
        self._load_cluster(Cluster_3, 3, cluster_dict)
        self._load_cluster(Cluster_4, 4, cluster_dict)
        self._load_cluster(Cluster_6, 6, cluster_dict)
        self._load_cluster(Cluster_7, 7, cluster_dict)
        self._load_cluster(Cluster_10, 10, cluster_dict)
        self._load_cluster(Cluster_11, 11, cluster_dict)
        self._load_cluster(Cluster_v0, -1, cluster_dict)

        return cluster_dict
    
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df1 = self._load_original_data(DATA_FILE_PATH)

        if INCLUDE_CV_PREDICTION:
            # Adds CV data
            df1_cv = self._load_original_data(DATA_FILE_CV_PATH)
            df1 = df1.append(df1_cv)

        # Additional context df (e.g Population for each country)
        df2 = self._load_additional_context_df()

        # Merge the 2 DataFrames
        df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))

        # Drop countries with no population data
        df.dropna(subset=['Population'], inplace=True)

        #  Keep only needed columns
        columns = CONTEXT_COLUMNS + NPI_COLUMNS
        df = df[columns]

        # Fill in missing values
        self._fill_missing_values(df)

        # Compute number of new cases and deaths each day
        df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)

        # Replace negative values (which do not make sense for these columns) with 0
        df['NewCases'] = df['NewCases'].clip(lower=0)
        df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

        # Compute smoothed versions of new cases and deaths each day
        df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
        df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

        # Compute percent change in new cases and deaths each day
        df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1
        df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1

        # Add column for proportion of population infected
        df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

        # Create column of value to predict
        df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

        return df

    @staticmethod
    def _load_original_data(data_url):
        latest_df = pd.read_csv(data_url,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
        # GeoID is CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
        return latest_df

    @staticmethod
    def _fill_missing_values(df):
        """
        # Fill missing values by interpolation, ffill, and filling NaNs
        :param df: Dataframe to be filled
        """
        df.update(df.groupby('GeoID').ConfirmedCases.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of cases is available
        df.dropna(subset=['ConfirmedCases'], inplace=True)
        df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of deaths is available
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)
        for npi_column in NPI_COLUMNS:
            df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

    @staticmethod
    def _load_additional_context_df():
        # File containing the population for each country
        # Note: this file contains only countries population, not regions
        additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                            usecols=['CountryName', 'Population'])
        additional_context_df['GeoID'] = additional_context_df['CountryName']

        # US states population
        additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                              usecols=['NAME', 'POPESTIMATE2019'])
        # Rename the columns to match measures_df ones
        additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
        # Prefix with country name to match measures_df
        additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_us_states_df)

        # UK population
        additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_uk_df)

        # Brazil population
        additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_brazil_df)

        if INCLUDE_CV_PREDICTION:
            additional_cv_df = pd.DataFrame(data = {'CountryName': ['Spain'], 'Population': [5003769], 'GeoID': ['Spain / ComunidadValenciana']})
            additional_context_df = additional_context_df.append(additional_cv_df)
        
        return additional_context_df


    @staticmethod
    def _create_model_sco_v0(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((nb_lookback_days, nb_context))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _lstm_rn = LSTM(units=lstm_size)(_conv1d_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm_rn)
        model_rn = Model(_input_rn, _output_rn)
        #print(model_rn.summary())
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])

        _input_hn = Input((nb_lookback_days, nb_action))
        _conv1d_hn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_hn)
        _lstm_hn = LSTM(units=lstm_size, kernel_constraint=Positive(), recurrent_constraint=Positive(), bias_constraint=Positive(),return_sequences=False)(_conv1d_hn)
        _output_hn_med = Dense(10, activation="sigmoid")(_lstm_hn)
        _output_hn = Dense(1, activation="sigmoid")(_output_hn_med)
        model_hn = Model(_input_hn, _output_hn)
        #print(model_hn.summary())
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])

        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])
    
        return combined


    @staticmethod
    def _create_model_sco_v1(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((nb_lookback_days, nb_context))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _lstm_rn = Bidirectional(LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(), return_sequences=False))(_conv1d_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm_rn)
        model_rn = Model(_input_rn, _output_rn)
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        _input_hn = Input((nb_lookback_days, nb_action))
        _lstm_hn = LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(),return_sequences=False)(_input_hn)
        _output_hn_med = Dense(10, activation="sigmoid")(_lstm_hn)
        _output_hn = Dense(1, activation="sigmoid")(_output_hn_med)
        model_hn = Model(_input_hn, _output_hn)
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])
    
        return combined

    @staticmethod
    def _create_model_sco_v2(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((nb_lookback_days, nb_context))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _lstm_rn = Bidirectional(LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(), return_sequences=False))(_conv1d_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm_rn)
        model_rn = Model(_input_rn, _output_rn)
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        _input_hn = Input((nb_lookback_days, nb_action))
        _lstm_hn = LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(),return_sequences=False)(_input_hn)
        _output_hn_med = Dense(10, activation="sigmoid")(_lstm_hn)
        _output_hn = Dense(1, activation="sigmoid")(_output_hn_med)
        model_hn = Model(_input_hn, _output_hn)
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])
    
        return combined

    @staticmethod
    def _create_model(path_to_model_weights):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((21, 1))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _conv2d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_conv1d_rn)
        _lstm_rn = Bidirectional(LSTM(32, return_sequences=True))(_conv2d_rn)
        _lstm2_rn = Bidirectional(LSTM(32))(_lstm_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm2_rn)

        model_rn = Model(_input_rn, _output_rn)
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])
        _input_hn = Input((21, 8))
        _lstm_hn = LSTM(32)(_input_hn)
        _output_hn = Dense(1, activation="sigmoid")(_lstm_hn)

        model_hn = Model(_input_hn, _output_hn)
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])
        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])

        combined.load_weights(path_to_model_weights)

        return combined
    

def compute_7days_mean(casos_diarios):
    '''
    Function to compute the 7 days mean for daily cases (Zt), to smooth the reporting anomalies.
    Input:
        - daily cases list
    Output:
        - daily Zt (7 days mean, current day and the 6 prior it)
    '''
    zn = []
    for i in range(len(casos_diarios)):
        if i == 0:
            zn.append(casos_diarios[i])
        elif i > 0 and i < 6:
            acc = 0
            for j in range(i):
                acc += casos_diarios[i-j]
            zn.append(acc/i)
        else:
            acc = 0
            for j in range(7):
                acc += casos_diarios[i-j]
            zn.append(acc/7)
    return zn

def compute_last_7days_mean(casos_diarios):
    '''
    Function to compute the 7 days mean for the last day (Zt), to smooth the reporting anomalies.
    Input:
        - daily cases list
    Output:
        - last day Zt (7 days mean, current day and the 6 prior it)
    '''
    i = len(casos_diarios) - 1
    if i == 0:
        zn=casos_diarios[i]
    elif i > 0 and i < 6:
        acc = 0
        for j in range(i):
            acc += casos_diarios[i-j]
        zn=acc/i
    else:
        acc = 0
        for j in range(7):
            acc += casos_diarios[i-j]
        zn=acc/7

    return zn



def compute_rns(casos_acumulados, zn, population):
    '''
    Function to take into account population size and immunity when calculating Rt.
    Input:
        - cummulated cases list
        - daily Zt means over 7 days (Zn)
        - population for the given country or region
    Output:
        - Rns list
    '''
    rn = []
    for i in range(len(casos_acumulados)):
        if i == 0:
            num = population * zn[i]
            denom = population
            rn.append(num/denom)
        else:
            if zn[i-1] == 0:
                rn.append(0)
            else:
                num = population * zn[i]
                denom = (population - casos_acumulados[i-1]) * zn[i-1]  # En xprize utilizan casos_acumulados[i]
                rn.append(num/denom)
    rn = [2 if x>2 else x for x in rn] # En xprize no acotan a 2 en la prediccion, solo en el train
    return rn


def compute_last_rn(casos_acumulados, zn, population):
    '''
    Function to take into account population size and immunity when calculating Rt.
    Input:
        - cummulated cases list
        - daily Zt means over 7 days (Zn)
        - population for the given country or region
    Output:
        - Last Rn
    '''
    i = len(casos_acumulados) - 1
    if i == 0:
        num = population * zn[i]
        denom = population
        rn=num/denom
    else:
        if zn[i-1] == 0:
            rn=0
        else:
            num = population * zn[i]
            denom = (population - casos_acumulados[i-1]) * zn[i-1]  # En xprize utilizan casos_acumulados[i]
            rn=num/denom
    rn = 2 if rn>2 else rn # En xprize no acotan a 2 en la prediccion, solo en el train
    return rn


