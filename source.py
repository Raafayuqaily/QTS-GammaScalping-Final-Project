"""
Source file containing all functions used in the notebook, imported directly.

"""
# ============================================

# Standard
import bisect
import collections
import datetime
import gzip
import io
import json
import math
import os
import re
import sys
import urllib.request
import zipfile

# Third-party
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import psycopg2
import requests
import scipy
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.graphics.tsaplots
import statsmodels.regression.linear_model
import statsmodels.stats.stattools
import statsmodels.tools
import wrds
import quandl

# Specific
from bisect import insort, bisect_left
from collections import deque
from datetime import datetime
from itertools import islice
from matplotlib.ticker import PercentFormatter
from plotnine import (aes, geom_bar, geom_hline, geom_line, geom_point, geom_ribbon, ggplot, 
                      guide_legend, guides, labs, scale_color_identity, scale_color_manual, 
                      scale_size_manual, scale_x_date, scale_y_continuous, theme, theme_minimal,
                      element_blank, element_text, ggtitle)
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import acf, coint

# Warnings
import warnings
warnings.filterwarnings("ignore")


# ============================================

# Quandl data retrieval functions


def grab_quandl_table(
    table_path,
    quandl_api_key,
    avoid_download=False,
    replace_existing=False,
    date_override=None,
    allow_old_file=False,
    **kwargs,
):
    root_data_dir = os.path.join(os.getcwd(), "quandl_data_table_downloads")
    data_symlink = os.path.join(root_data_dir, f"{table_path}_latest.zip")
    if avoid_download and os.path.exists(data_symlink):
        print(f"Skipping any possible download of {table_path}")
        return data_symlink
    
    table_dir = os.path.dirname(data_symlink)
    if not os.path.isdir(table_dir):
        print(f'Creating new data dir {table_dir}')
        os.makedirs(table_dir)

    if date_override is None:
        my_date = datetime.datetime.now().strftime("%Y%m%d")
    else:
        my_date = date_override
    data_file = os.path.join(root_data_dir, f"{table_path}_{my_date}.zip")

    if os.path.exists(data_file):
        file_size = os.stat(data_file).st_size
        if replace_existing or not file_size > 0:
            print(f"Removing old file {data_file} size {file_size}")
        else:
            print(
                f"Data file {data_file} size {file_size} exists already, no need to download"
            )
            return data_file

    dl = quandl.export_table(
        table_path, filename=data_file, api_key=quandl_api_key, **kwargs
    )
    file_size = os.stat(data_file).st_size
    if os.path.exists(data_file) and file_size > 0:
        print(f"Download finished: {file_size} bytes")
        if not date_override:
            if os.path.exists(data_symlink):
                print(f"Removing old symlink")
                os.unlink(data_symlink)
            print(f"Creating symlink: {data_file} -> {data_symlink}")
            os.symlink(
                data_file, data_symlink,
            )
    else:
        print(f"Data file {data_file} failed download")
        return
    return data_symlink if (date_override is None or allow_old_file) else "NoFileAvailable"

def fetch_quandl_table(table_path, api_key, avoid_download=True, **kwargs):
    return pd.read_csv(
        grab_quandl_table(table_path, api_key, avoid_download=avoid_download, **kwargs)
    )


# ============================================

# Data Restructuring

# Checkpoint2

def prepare_ticker_data(ticker_data, start_date, end_date):
    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
    filtered_data = ticker_data[(ticker_data['date'] >= start_date) & (ticker_data['date'] <= end_date)]
    sorted_data = filtered_data.sort_values(by='date')
    reduced_data = sorted_data[['date', 'close', 'adj_open', 'adj_close', 'adj_volume']]
    return reduced_data

def prepare_option_data(option_data):
    option_data['date'] = pd.to_datetime(option_data['date'])
    option_data['strike_price'] = option_data['strike_price'] / 1000.0
    return option_data

def enrich_option_data(option_data, ticker_data_reduced):
    enriched_data = pd.merge(option_data, ticker_data_reduced, on='date', how='left')
    return enriched_data

# Checkpoint3
def merge_dataframes(original_dfs, result_dfs):
    filtered_dfs = {}
    for key in original_dfs.keys():
        df = original_dfs[key]
        result_df = result_dfs[key]
        filtered_df = df.merge(result_df, on=['date', 'TTE'])
        filtered_dfs[key] = filtered_df
    return filtered_dfs

def filter_dfs_on_criteria(filtered_dfs, select_row_with_smallest_diff):
    final_dfs = {}
    for key, df in filtered_dfs.items():
        final_df = df.groupby('date', as_index=False).apply(select_row_with_smallest_diff).reset_index(drop=True)
        final_df = final_df.drop(columns=['abs_diff'])
        final_dfs[key] = final_df
    return final_dfs

def calculate_closest_dates(final_dfs):
    for key, df in final_dfs.items():
        df['date'] = pd.to_datetime(df['date'])
        dates_np = df['date'].values
        target_dates = dates_np + np.timedelta64(21, 'D')
        abs_diff_matrix = np.abs(dates_np[:, None] - target_dates)
        min_diff_indices = np.argmin(abs_diff_matrix, axis=0)
        closest_dates = dates_np[min_diff_indices]
        df['close_date'] = closest_dates
        final_dfs[key] = df
    return final_dfs

def merge_with_options(final_dfs, options_df1):
    options_df1['date'] = pd.to_datetime(options_df1['date'])
    options_df1['exdate'] = pd.to_datetime(options_df1['exdate'])
    
    for key, df in final_dfs.items():
        df['close_date'] = pd.to_datetime(df['close_date'])
        df['exdate'] = pd.to_datetime(df['exdate'])
        merged_df = pd.merge(df, options_df1, left_on=['close_date', 'cp_flag', 'strike_price', 'exdate'],
                             right_on=['date', 'cp_flag', 'strike_price', 'exdate'],
                             how='left', indicator=True)
        
        merged_df['is_present'] = merged_df['_merge'] == 'both'
        columns_to_drop = ['_merge'] + [col for col in merged_df.columns if col.endswith('_y')]
        merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        final_dfs[key] = merged_df
    return final_dfs

def find_closest_tte(group, days = 30):
    group['diff'] = (group['TTE'] - days).abs() - group['TTE'].lt(days) * 0.1
    return group.loc[group['diff'].idxmin()]

def select_row_with_smallest_diff(group):
    group['abs_diff'] = (group['close'] - group['strike_price']).abs()
    return group.loc[group['abs_diff'].idxmin()]

def check_dates_in_calls_and_puts(calls_key, puts_key, final_dfs):
    calls_df = final_dfs[calls_key]
    puts_df = final_dfs[puts_key]

    filtered_calls = calls_df[calls_df['is_present'] == False]
    filtered_puts = puts_df[puts_df['is_present'] == False]

    calls_dates = set(filtered_calls['date_x'].unique())
    puts_dates = set(filtered_puts['date_x'].unique())

    return calls_dates.issubset(puts_dates)

def count_dates_comparison(calls_key, puts_key, final_dfs):
    calls_df = final_dfs[calls_key][final_dfs[calls_key]['is_present'] == False]
    puts_df = final_dfs[puts_key][final_dfs[puts_key]['is_present'] == False]

    calls_dates = set(calls_df['date_x'].unique())
    puts_dates = set(puts_df['date_x'].unique())

    unique_in_calls = calls_dates.difference(puts_dates)
    unique_in_puts = puts_dates.difference(calls_dates)

    overlapping_dates = calls_dates.intersection(puts_dates)

    count_unique_in_calls = len(unique_in_calls)
    count_unique_in_puts = len(unique_in_puts)
    count_overlapping = len(overlapping_dates)

    return count_unique_in_calls, count_unique_in_puts, count_overlapping

# ============================================

# Further Data restructuring

def preprocess_options_data(calls_df, puts_df):
    calls_df.rename(columns={'date_x': 'date'}, inplace=True)
    puts_df.rename(columns={'date_x': 'date'}, inplace=True)

    calls_df['date'] = pd.to_datetime(calls_df['date'])
    puts_df['date'] = pd.to_datetime(puts_df['date'])

    calls_df['exdate'] = pd.to_datetime(calls_df['exdate'])
    puts_df['exdate'] = pd.to_datetime(puts_df['exdate'])

    return calls_df, puts_df


def compare_strike_prices(calls_df, puts_df):
    calls_grouped = calls_df.groupby('date')['strike_price'].apply(list).reset_index(name='calls_strike_prices')
    puts_grouped = puts_df.groupby('date')['strike_price'].apply(list).reset_index(name='puts_strike_prices')

    merged_df = pd.merge(calls_grouped, puts_grouped, on='date')
    merged_df['strike_prices_match'] = merged_df.apply(lambda row: set(row['calls_strike_prices']) == set(row['puts_strike_prices']), axis=1)

    return merged_df[merged_df['strike_prices_match'] == False]['date']

def compare_strike_prices_and_exdates(calls_df, puts_df):
    calls_grouped = calls_df.groupby('date').apply(lambda x: list(zip(x['strike_price'], x['exdate']))).reset_index(name='calls_data')
    puts_grouped = puts_df.groupby('date').apply(lambda x: list(zip(x['strike_price'], x['exdate']))).reset_index(name='puts_data')

    merged_df = pd.merge(calls_grouped, puts_grouped, on='date')
    merged_df['data_match'] = merged_df.apply(lambda row: set(row['calls_data']) == set(row['puts_data']), axis=1)

    return merged_df[merged_df['data_match'] == False]['date']


# ============================================

# Simulation Code

# Gives more metrics, but we will generally use the later, streamlined version of creating simulations in the actual implementation.

def create_simulations_original(options_subset, data, dropna_greeks=False):
    simulations = {}

    for index, row in options_subset.iterrows():
        strikeID = row['exdate'].strftime('%Y%m%d') + '_' + str(row['strike_price'])
        mask = (data['strikeID'] == strikeID) & (data['date'] >= row['date']) & (data['date'] <= row['close_date'])
        temp_df = data[mask].sort_values(by=['date', 'cp_flag'])

        shared_cols = ['date', 'exdate', 'strike_price', 'expiry_indicator', 'close', 'adj_open', 'adj_close', 'adj_volume', 'strikeID']
        greeks_cols = ['impl_volatility', 'delta', 'gamma', 'vega', 'theta']
        call_specific_cols = ['cp_flag', 'best_bid', 'best_offer', 'volume', 'open_interest'] + greeks_cols
        put_specific_cols = call_specific_cols

        calls = temp_df[temp_df['cp_flag'] == 'C'][shared_cols + call_specific_cols].rename(columns={col: col + '_c' for col in call_specific_cols})
        puts = temp_df[temp_df['cp_flag'] == 'P'][shared_cols + put_specific_cols].rename(columns={col: col + '_p' for col in put_specific_cols})

        merged_df = pd.merge(calls, puts, on=shared_cols, how='outer')

        if dropna_greeks:
            greeks_cols_c = [col + '_c' for col in greeks_cols]
            greeks_cols_p = [col + '_p' for col in greeks_cols]
            merged_df = merged_df.dropna(subset=greeks_cols_c + greeks_cols_p, how='any')

        merged_df['delta_sum'] = merged_df['delta_c'].fillna(0) + merged_df['delta_p'].fillna(0)
        merged_df['shares_held'] = -1 * merged_df['delta_sum']

        merged_df = merged_df.sort_values(by='date')
        merged_df['sharechange'] = merged_df['shares_held'].diff()

        simulations[row['date'].strftime('%Y-%m-%d')] = merged_df

    return simulations

# Current version of create_simulations

def create_simulations(options_subset, data, dropna_greeks=False):
    simulations = {}

    for index, row in options_subset.iterrows():
        strikeID = row['exdate'].strftime('%Y%m%d') + '_' + str(row['strike_price'])
        mask = (data['strikeID'] == strikeID) & (data['date'] >= row['date']) & (data['date'] <= row['close_date'])
        temp_df = data[mask].sort_values(by=['date', 'cp_flag'])

        shared_cols = ['date', 'exdate', 'strike_price', 'close', 'strikeID'] # 'expiry_indicator',  'adj_open', 'adj_close', 'adj_volume',
        greeks_cols = ['impl_volatility', 'delta'] # , 'gamma', 'vega', 'theta'
        call_specific_cols = ['cp_flag', 'best_bid', 'best_offer'] + greeks_cols # , 'volume', 'open_interest'
        put_specific_cols = call_specific_cols

        calls = temp_df[temp_df['cp_flag'] == 'C'][shared_cols + call_specific_cols].rename(columns={col: col + '_c' for col in call_specific_cols})
        puts = temp_df[temp_df['cp_flag'] == 'P'][shared_cols + put_specific_cols].rename(columns={col: col + '_p' for col in put_specific_cols})

        merged_df = pd.merge(calls, puts, on=shared_cols, how='outer')

        if dropna_greeks:
            greeks_cols_c = [col + '_c' for col in greeks_cols]
            greeks_cols_p = [col + '_p' for col in greeks_cols]
            merged_df = merged_df.dropna(subset=greeks_cols_c + greeks_cols_p, how='any')

        merged_df['delta_sum'] = merged_df['delta_c'].fillna(0) + merged_df['delta_p'].fillna(0)
        merged_df['shares_held'] = -1 * merged_df['delta_sum']

        merged_df = merged_df.sort_values(by='date')
        merged_df['sharechange'] = merged_df['shares_held'].diff()

        simulations[row['date'].strftime('%Y-%m-%d')] = merged_df

    return simulations

# ============================================

# IV calculations

def load_and_transform_tbills(file_path):
    tbills = pd.read_csv(file_path)
    tbills = tbills.rename(columns={
        'TMATDT': 'maturity_date',
        'CALDT': 'quote_date',
        'TDNOMPRC': 'price',
        'TDDURATN': 'dte'
    })
    tbills = tbills.iloc[:, [2, 6, 9, 10]]
    tbills['maturity_date'] = pd.to_datetime(tbills['maturity_date'])
    tbills['quote_date'] = pd.to_datetime(tbills['quote_date'])
    return tbills

def load_and_transform_calls(file_path):
    calls = pd.read_csv(file_path)
    calls['date_x'] = pd.to_datetime(calls['date_x'])
    calls['exdate'] = pd.to_datetime(calls['exdate'])
    calls['dte'] = (calls['exdate'] - calls['date_x']).dt.days
    return calls

def load_and_transform_puts(file_path):
    puts = pd.read_csv(file_path)
    puts['date_x'] = pd.to_datetime(puts['date_x'])
    puts['exdate'] = pd.to_datetime(puts['exdate'])
    puts['dte'] = (puts['exdate'] - puts['date_x']).dt.days
    return puts

def load_and_transform_option_data(file_path):
    option_data = pd.read_csv(file_path)
    option_data['midpt'] = (option_data['best_bid'] + option_data['best_offer']) / 2
    option_data['date'] = pd.to_datetime(option_data['date'])
    option_data['exdate'] = pd.to_datetime(option_data['exdate'])
    option_data['dte'] = (option_data['exdate'] - option_data['date']).dt.days
    return option_data




def find_closest_index(val, col2):
    return np.abs(col2 - val).idxmin()

def calculate_iv_calls(df, delta_k, s_0, zcb_price):
    options_df = df.copy()
    options_df['adjusted_strike'] = options_df['strike_price'] / (zcb_price/100)

    options_df['closest_price_index'] = options_df['adjusted_strike'].apply(lambda x: find_closest_index(x, options_df['strike_price']))

    options_prices = options_df.loc[options_df['closest_price_index'].values, 'midpt']

    underlying_minus_strike = s_0 - (options_df['strike_price']/1000)
    underlying_minus_strike[underlying_minus_strike < 0] = 0
    options_prices.index = underlying_minus_strike.index
    options_df['g'] = (options_prices - underlying_minus_strike)/((options_df['strike_price']/1000)**2)
    
    options_df.iloc[1:-2, -1] = options_df.iloc[1:-2, -1] * 2
    return options_df.iloc[:, -1].sum() * delta_k

def calc_s0(dt, zcb_price, dte, spydata):
    unadjusted_price = spydata[spydata['date'] == dt]['close'].iloc[0]
    dividend_window_end = dt + pd.Timedelta(days=dte)

    dividends_to_be_paid = spydata[(spydata['date'] <= dividend_window_end) & (spydata['date'] >= dt)]['dividend'].sum()
    adjusted_price = unadjusted_price - (dividends_to_be_paid * zcb_price)
    return adjusted_price

def calculate_iv_for_calls(calls, option_data, tbills, spydata):
    our_ivs = pd.DataFrame(columns=['iv'])
    sizes = pd.Series()

    for ind, dt in enumerate(calls['date_x']):
        call_df = option_data[(option_data['date'] == dt) & (option_data['cp_flag'] == 'C')]
        call_df = call_df[call_df['dte'] == calls.loc[ind, 'dte']]
        call_df = call_df.sort_values('strike_price')

        call_df = call_df.reset_index(drop=True)

        call_df = call_df[call_df['midpt'] > 0.375]
        atm_strike = calls.loc[ind, 'strike_price']

        call_df = call_df[call_df['strike_price'] > 0.97 * atm_strike]

        call_df['increments'] = call_df['strike_price'].diff().bfill()

        mode = call_df['increments'].mode().iloc[0]
        inds_to_drop = pd.Series(call_df[call_df['increments'] > mode].index)
        midpoint = call_df.shape[0] / 2
        lower_inds_to_drop = inds_to_drop[inds_to_drop < midpoint]
        upper_inds_to_drop = inds_to_drop[inds_to_drop > midpoint]

        if not lower_inds_to_drop.empty:
            call_df = call_df.iloc[lower_inds_to_drop.max():]

        if not upper_inds_to_drop.empty:
            call_df = call_df.iloc[:upper_inds_to_drop.min()]

        valid_strikes = np.arange(start=int(call_df['strike_price'].min()), stop=call_df['strike_price'].max() + mode, step=mode)
        call_df = call_df[call_df['strike_price'].isin(valid_strikes)]

        # Look at how many options are being used to find IV
        sizes.loc[ind] = call_df.shape[0]

        delta_k = mode / 1000
        dte = call_df['dte'].iloc[0]

        tbills_today = tbills[(tbills['quote_date'] == dt)]
        days_back = -1
        while tbills_today.empty:
            tbills_today = tbills[tbills['quote_date'] == dt + pd.Timedelta(days=days_back)]
            days_back -= 1
        zcb_price = tbills_today[abs(tbills_today['dte'] - dte) == abs(tbills_today['dte'] - dte).min()]['price'].iloc[0]

        s_0 = calc_s0(dt, zcb_price, dte, spydata)

        iv = calculate_iv_calls(call_df, delta_k, s_0, zcb_price)

        our_ivs.loc[dt] = [iv]
    
    return our_ivs, sizes

# ============================================

# Strategy Simulation Part 1

def prepare_dataframes(data_file_path, options_file_path):
    data = pd.read_csv(data_file_path)
    options = pd.read_csv(options_file_path)

    data['exdate'] = pd.to_datetime(data['exdate'])
    options['exdate'] = pd.to_datetime(options['exdate'])

    data['exdate_str'] = data['exdate'].dt.strftime('%Y%m%d')
    data['strikeID'] = data['exdate_str'] + '_' + data['strike_price'].astype(str)
    data.drop(columns=['exdate_str'], inplace=True)

    options['exdate_str'] = options['exdate'].dt.strftime('%Y%m%d')
    options['strikeID'] = options['exdate_str'] + '_' + options['strike_price'].astype(str)
    options.drop(columns=['exdate_str'], inplace=True)

    options['date'] = pd.to_datetime(options['date'])
    data['date'] = pd.to_datetime(data['date'])

    return data, options

# dividend pay dates
def find_pay_date(end_of_month, trading_days):
    if end_of_month in trading_days:
        return end_of_month
    else:
        eligible_days = trading_days[trading_days <= end_of_month]
        return eligible_days.max() 
    
def process_spy_dividends(ticker_data_path, start_date, end_date):
    spy_divdata = pd.read_csv(ticker_data_path)[['date', 'dividend']].sort_values(by='date').reset_index(drop=True)
    spy_divdata = spy_divdata.loc[(spy_divdata['date'] >= start_date) & (spy_divdata['date'] <= end_date)].copy().reset_index(drop=True)
    spy_divdata['date'] = pd.to_datetime(spy_divdata['date'])
    
    trading_days = spy_divdata['date']
    
    spy_divdata = spy_divdata.loc[spy_divdata['dividend'] != 0]

    spy_divdata['end_of_next_month'] = spy_divdata['date'] + pd.offsets.MonthEnd(2)
    spy_divdata['pay_date'] = spy_divdata['end_of_next_month'].apply(lambda date: find_pay_date(date, trading_days))
    spy_divdata.drop(columns=['end_of_next_month'], inplace=True)

    return spy_divdata, trading_days
    
def filter_simulations(simulations, trading_days):
    filtered_simulations = {}

    for key, df in simulations.items():
        # Ensure 'date' column is in datetime64 dtype
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the range of trading days for each simulation
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        # Filter the trading_days series to get the expected range of dates
        expected_trading_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]
        
        # Get unique trading days from the simulation
        actual_trading_days = df['date'].unique()
        actual_trading_days = pd.to_datetime(actual_trading_days)
        
        # Check if all expected trading days are present in the actual trading days
        if expected_trading_days.isin(actual_trading_days).all():
            filtered_simulations[key] = df
    
    return filtered_simulations

# ============================================

# Strategy simulation part 2: PL/metrics

def calculate_realized_PL(df, long_op=True):
    df = df.reset_index(drop=True)
    
    # Vectorized initial operations for stock
    df['stock_pos'] = np.where(long_op, df['shares_held'], -df['shares_held'])
    df = df.drop(columns=['shares_held'])
    df['pos_change'] = np.where(long_op, df['sharechange'], -df['sharechange'])
    df = df.drop(columns=['sharechange'])
    df.loc[0, 'pos_change'] = df.loc[0, 'stock_pos']
    
    df['change_cost_basis'] = df['pos_change'] * df['close']
    df['stock_cost_basis'] = df['change_cost_basis'].cumsum()
    df['daily_stock_value'] = df['stock_pos'] * df['close']
    df['stock_PL'] = df['daily_stock_value'] - df['stock_cost_basis']

    # Initial option value and vectorized daily option value calculation
    df['option_cost_basis'] = df.loc[0, 'best_offer_c'] + df.loc[0, 'best_offer_p'] if long_op else -df.loc[0, 'best_bid_c'] - df.loc[0, 'best_bid_p']
    df['change_cost_basis_op'] = 0.0
    df.loc[0, 'change_cost_basis_op'] = df.loc[0, 'option_cost_basis']
    df['daily_option_value'] = np.where(long_op, df['best_bid_c'] + df['best_bid_p'], -(df['best_offer_c'] + df['best_offer_p']))
    df['option_PL'] = df['daily_option_value']- df['option_cost_basis']

    # Column to track total positions, PL, and cash flow after positions are closed
    df['total_cost_basis'] = df['stock_cost_basis'] + df['option_cost_basis']
    df['total_pos_value'] = df['daily_stock_value'] + df['daily_option_value']
    df['total_PL'] = df['stock_PL'] + df['option_PL']
    df['realized_stock_PL'] = 0.0
    df['realized_option_PL'] = 0.0
    df['realized_PL'] = 0.0

    # Misc
    df['UID'] = df['strikeID'] + '_' + str(df.loc[0, 'date'].date())
    df['to_open'] = 0
    df.loc[0, 'to_open'] = 1
#    df['gross_trades_value'] = abs(df['to_open'] * df['option_cost_basis']) + abs(df['change_cost_basis']) # Need to do this at end

    # Close positions on final day
    final_row_index = len(df) - 1
    df.loc[final_row_index, 'realized_stock_PL'] = df.loc[final_row_index, 'stock_PL'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'realized_option_PL'] = df.loc[final_row_index, 'option_PL'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'realized_PL'] = df.loc[final_row_index, 'total_PL'] if final_row_index > 0 else 0
#    df.loc[final_row_index, 'gross_trades_value'] = abs(df.loc[final_row_index, 'daily_option_value']) + abs(df.loc[final_row_index - 1, 'stock_pos']) * df.loc[final_row_index, 'close'] 

    final_close_price = df.loc[final_row_index, 'close']
    df.loc[final_row_index, 'stock_pos'] = 0
    df.loc[final_row_index, 'pos_change'] = - df.loc[final_row_index - 1, 'stock_pos'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'change_cost_basis'] = df.loc[final_row_index, 'pos_change'] * final_close_price
    df.loc[final_row_index, 'stock_cost_basis'] = 0
    df.loc[final_row_index, 'daily_stock_value'] = 0
    df.loc[final_row_index, 'stock_PL'] = 0

    df.loc[final_row_index, 'option_cost_basis'] = 0
    df.loc[final_row_index, 'change_cost_basis_op'] = -df.loc[final_row_index, 'daily_option_value'] if final_row_index > 0 else 0
    df.loc[final_row_index, 'daily_option_value'] = 0
    df.loc[final_row_index, 'option_PL'] = 0

    df.loc[final_row_index, 'total_cost_basis'] = 0
    df.loc[final_row_index, 'total_pos_value'] = 0
    df.loc[final_row_index, 'total_PL'] = 0
    
    return df

def compare_iv(filtered_simulations, iv_data):
    temp_data = []

    for key, df in filtered_simulations.items():
        temp_data.append({'date': key, 'BS_Call_IV': df.loc[0, 'impl_volatility_c']})

    BS_Call_IV = pd.DataFrame(temp_data)

    iv_data['date'] = pd.to_datetime(iv_data['date'])
    BS_Call_IV['date'] = pd.to_datetime(BS_Call_IV['date'])

    IV_compare = pd.merge(BS_Call_IV, iv_data[['date', 'iv']], on='date', how='left')
    IV_compare.rename(columns={'iv': 'MF_Call_IV'}, inplace=True)
    IV_compare['IV_diff'] = IV_compare['MF_Call_IV'] - IV_compare['BS_Call_IV']

    return IV_compare

# ============================================

# Strategies

# Long-Short
def trade_strategy_1(x):
    if x > 0.25:
        return 1
    elif x < -0.10:
        return -1
    else:
        return 0

# Long Only
def trade_strategy_2(x):
    if x > 0.35:
        return 1
    else:
        return 0

# Short Only
def trade_strategy_3(x):
    if x < -0.08:
        return -1
    else:
        return 0
    

def generate_trades_dfs(strat_dict, initial_df, simulations_long, simulations_short):
    trades_dfs = {}
    
    for key in strat_dict.keys():
        dfs_to_combine = []
        
        for index, row in initial_df.iterrows():
            date = row['date']
            trade = row[key]
            iv_diff = row['IV_diff']
            
            if trade == 1 and date in simulations_long:
                df_to_add = simulations_long[date].copy()
            elif trade == -1 and date in simulations_short:
                df_to_add = simulations_short[date].copy()
            else:
                # Skip if 'trade' is 0 or the date is not in the dictionaries
                continue
            
            # Add 'trade' & 'IV_diff' column
            df_to_add['IV_diff'] = iv_diff  # Needed for position calculation
            df_to_add[key] = trade  # Include the 'trade' value
            dfs_to_combine.append(df_to_add)
        
        # Concatenate all collected DataFrames
        trades_dfs[key] = pd.concat(dfs_to_combine, ignore_index=True)
        trades_dfs[key] = trades_dfs[key].sort_values(by=['date', 'exdate', 'strike_price', 'to_open']).reset_index(drop=True)

    return trades_dfs

def preprocess_options(options_df):
    options_df['UID'] = options_df['strikeID'] + '_' + options_df['date'].dt.date.astype(str)
    volumes = options_df[['date', 'volume_c', 'volume_p', 'adj_volume', 'UID']].copy()
    volumes['date'] = volumes['date'].dt.strftime('%Y-%m-%d')
    volumes['volume_med'] = (volumes['volume_c'] + volumes['volume_p']) / 2
    return volumes

def pos_size(IV_diff, strike_price, option_cost_basis, UID, key, volumes_df, trades_dfs, KAPITAL = 1e7):
    volume = min(volumes_df.loc[volumes_df['UID'] == UID, 'volume_med'].item(), 50)

    # Calculating position size based on attractiveness while ensuring risk stays within limits and capital remains bounded
    factor = min(min(abs(IV_diff), 0.8) * volume * strike_price / 10, KAPITAL / 10)

    if option_cost_basis == 0:
        filtered_df = trades_dfs[key].loc[trades_dfs[key]['UID'] == UID, 'option_cost_basis']
        option_cost_basis = filtered_df.iloc[0] if not filtered_df.empty else 0

    # Requires a whole number of options contracts
    return round(factor / abs(option_cost_basis)) if option_cost_basis != 0 else 0

def update_trades_with_pos_size(trades_dfs, volumes, KAPITAL):
    for key, df in trades_dfs.items():
        df = df.drop(columns=[col for col in df.columns if col.endswith('_p') or col.endswith('_c')]).copy()

        df['pos_size'] = df.apply(lambda row: pos_size(row['IV_diff'], row['strike_price'], row['option_cost_basis'], row['UID'], key, volumes, trades_dfs, KAPITAL = 1e7), axis=1)
        lot_size = 100 * df['pos_size']

        for col in ['stock_pos', 'pos_change', 'change_cost_basis', 'stock_cost_basis', 'daily_stock_value', 'stock_PL', 'option_cost_basis',
                    'change_cost_basis_op', 'daily_option_value', 'option_PL', 'total_cost_basis', 'total_pos_value', 'total_PL', 'realized_stock_PL',
                    'realized_option_PL', 'realized_PL']:
            df['sized_' + col] = lot_size * df[col]
        
        trades_dfs[key] = df
    
    return trades_dfs


# ============================================

# dataframe metrics

def summarize_pl_by_date(trades_dfs, trading_days):
    PL_temp_dfs = {}
    for key, df in trades_dfs.items():
        columns_to_sum = ['sized_' + col for col in ['stock_pos', 'change_cost_basis', 'stock_cost_basis', 'daily_stock_value', 'stock_PL', 'option_cost_basis', 'change_cost_basis_op',
                        'daily_option_value', 'option_PL', 'total_cost_basis', 'total_pos_value', 'total_PL', 'realized_stock_PL', 'realized_option_PL', 'realized_PL']]

        grouped_df = df[['date'] + columns_to_sum].groupby('date').sum().reset_index()
        pl_df = grouped_df.set_index('date').reindex(trading_days).fillna(0).reset_index()
        pl_df.rename(columns={'index': 'date'}, inplace=True)
        PL_temp_dfs[key] = pl_df
    
    return PL_temp_dfs

def calculate_dividends(PL_temp_dfs, spy_divdata):
    divvies = {}
    for key, df in PL_temp_dfs.items():
        df['date'] = pd.to_datetime(df['date'])
        #df['pay_date']  = pd.to_datetime(df['pay_date'])
        temp_merged = pd.merge(spy_divdata, df[['date', 'sized_stock_pos']], how='left', on='date')
        temp_merged['div'] = temp_merged['sized_stock_pos'] * temp_merged['dividend']
        divvies[key] = temp_merged
    
    return divvies

def merge_dividends_with_pl(PL_temp_dfs, divvies):
    for key, pl_df in PL_temp_dfs.items():
        div_df = divvies[key]
        pl_df['date'] = pd.to_datetime(pl_df['date'])
        merged_df = pd.merge(pl_df, div_df[['pay_date', 'div']], how='left', left_on='date', right_on='pay_date')
        merged_df.drop(columns=['pay_date'], inplace=True)
        merged_df['div'] = merged_df['div'].fillna(0)
        merged_df['sized_realized_stock_PL'] += merged_df['div']
        merged_df['sized_realized_PL'] += merged_df['div']
        PL_temp_dfs[key] = merged_df
    
    return PL_temp_dfs


def process_tbills_data(tbill_data_path, start_date, end_date, trading_days):
    tbills_data = pd.read_csv(tbill_data_path)[['CALDT', 'TDDURATN', 'TMATDT', 'TDNOMPRC']].sort_values(by=['CALDT', 'TDDURATN']).reset_index(drop=True)
    tbills_data = tbills_data.rename(columns={
        'TMATDT': 'maturity_date',
        'CALDT': 'date',
        'TDNOMPRC': 'price',
        'TDDURATN': 'dte'
    })
    tbills_data['maturity_date'] = pd.to_datetime(tbills_data['maturity_date'])
    tbills_data['date'] = pd.to_datetime(tbills_data['date'])
    tbills_data = tbills_data.loc[(tbills_data['date'] >= start_date) & (tbills_data['date'] <= end_date)].copy().reset_index(drop=True)
    tbills_data = tbills_data.drop_duplicates(subset='date', keep='first').reset_index(drop=True)
    tbills_data = tbills_data[tbills_data['date'].isin(trading_days)].copy().reset_index(drop=True)
    tbills_data['rate'] = (100 / tbills_data['price']) ** (1 / tbills_data['dte']) - 1
    tbills_data['leverage_rate'] = ((tbills_data['rate'] + 1) ** 365 + 25 / 100 / 100) ** (1 / 365) - 1  # 25 bps to loan anything (leverage)
    
    return tbills_data

def calculate_rfr(trading_days, tbills_data):
    if not isinstance(trading_days, pd.DataFrame):
        trading_days_df = pd.DataFrame({'date': trading_days})
    else:
        trading_days_df = trading_days.copy()

    trading_days_df['date'] = pd.to_datetime(trading_days_df['date'])
    tbills_data['date'] = pd.to_datetime(tbills_data['date'])

    rfr = trading_days_df.merge(tbills_data[['date', 'rate', 'leverage_rate']], on='date', how='left', sort=True)

    rfr['rate'] = rfr['rate'].ffill()
    rfr['leverage_rate'] = rfr['leverage_rate'].ffill()

    return rfr

# ============================================

# ALL METRICS - CORRESPONDING TO "Final Dataframes"

# Function calculates all relevant metrics for our trading analysis
# Due to the smaller data, we went for a less vectorized approach here to get more metrics easily.

def process_pl_dfs(PL_temp_dfs, rfr, INITIAL, KAPITAL):
    PL_dfs = {}
    
    for key, df in PL_temp_dfs.items():
        pl_df = pd.DataFrame(index=df.index)
        pl_df['date'] = df['date']

        # Trading costs and values
        pl_df['gross_stock_trades'] = abs(df['sized_change_cost_basis'])
        pl_df['gross_option_trades'] = abs(df['sized_change_cost_basis_op'])
        pl_df['gross_trades_value'] = pl_df['gross_stock_trades'] + pl_df['gross_option_trades']
        pl_df['stock_trading_costs'] = 1/100/100 * pl_df['gross_stock_trades']
        pl_df['option_trading_costs'] = 1/100/100 * pl_df['gross_option_trades']
        pl_df['net_trading_costs'] = 1/100/100 * pl_df['gross_trades_value']

        # Position values
        pl_df['stock_pos_value'] = df['sized_daily_stock_value']
        pl_df['option_pos_value'] = df['sized_daily_option_value']
        pl_df['gross_pos_value'] = pl_df['stock_pos_value'] + pl_df['option_pos_value']

        # Realized P&L calculations
        real_stock_PL = df['sized_realized_stock_PL'] - pl_df['stock_trading_costs']
        real_option_PL = df['sized_realized_option_PL'] - pl_df['option_trading_costs']
        real_net_PL = df['sized_realized_PL'] - pl_df['net_trading_costs']
        pl_df['stock_PL'] = real_stock_PL.cumsum()
        pl_df['option_PL'] = real_option_PL.cumsum()
        pl_df['net_PL'] = real_net_PL.cumsum()

        # Initial cash and capital calculations
        pl_df['start_cash'] = 0.0
        pl_df['initial_kapital'] = INITIAL
        pl_df['short_fee'] = 0.0
        pl_df['initial_cash'] = 0.0
        pl_df['interest'] = 0.0
        pl_df['lever_cash'] = 0.0
        pl_df['leverage_fee'] = 0.0
        pl_df['end_kapital'] = 0.0
        pl_df.loc[0, 'start_cash'] = INITIAL

        # Iterate over each day to calculate fees and interest
        for i in range(0, len(pl_df)):
            if i > 0:
                pl_df.loc[i, 'start_cash'] = pl_df.loc[i - 1, 'end_kapital']
            pl_df.loc[i, 'short_fee'] = - min(0, df.loc[i, 'sized_daily_stock_value']) * rfr.loc[i, 'leverage_rate']
            pl_df.loc[i, 'initial_kapital'] = pl_df.loc[i, 'start_cash'] + real_net_PL[i] - pl_df.loc[i, 'short_fee']
            pl_df.loc[i, 'initial_cash'] = max(pl_df.loc[i, 'initial_kapital'] - df.loc[i, 'sized_total_cost_basis'], 0)
            pl_df.loc[i, 'interest'] = pl_df.loc[i, 'initial_cash'] * rfr.loc[i, 'rate']
            pl_df.loc[i, 'lever_cash'] = max(df.loc[i, 'sized_total_cost_basis'] - pl_df.loc[i, 'initial_kapital'], 0)
            pl_df.loc[i, 'leverage_fee'] = pl_df.loc[i, 'lever_cash'] * rfr.loc[i, 'leverage_rate']
            pl_df.loc[i, 'end_kapital'] = pl_df.loc[i, 'initial_kapital'] + pl_df.loc[i, 'interest'] - pl_df.loc[i, 'leverage_fee']

        # Cumulative fees and interest
        pl_df['net_short_fees'] = pl_df['short_fee'].cumsum()
        pl_df['net_interest_paid'] = pl_df['interest'].cumsum()
        pl_df['net_interest_earned'] = pl_df['leverage_fee'].cumsum()

        # Final position value and total cash calculations
        pl_df['net_pos_value'] = pl_df['end_kapital'] - df['sized_total_cost_basis'] + pl_df['gross_pos_value']
        pl_df['tot_cash'] = KAPITAL - df['sized_total_cost_basis'] + pl_df['net_PL'] - pl_df['net_interest_paid'] + pl_df['net_interest_earned']

        PL_dfs[key] = pl_df

    return PL_dfs

# ============================================

# Original calculate PL function for mass simulation
# NOT USED LATER - for doc purposes only (in the appendix)

def calculate_realized_PL1(df, long_op=True):
    df = df.reset_index(drop=True)  
    df['stock_pos'] = df['shares_held'] if long_op else -df['shares_held']
    df['avg_cost'] = np.nan
    df['realized_PL'] = 0.0
    df['option_PL'] = 0.0
    df['stock_PL'] = 0.0  

    initial_option_value = df.loc[0, 'best_offer_c'] + df.loc[0, 'best_offer_p'] if long_op else -df.loc[0, 'best_bid_c'] - df.loc[0, 'best_bid_p']

    for i in range(len(df)):
        close_price = df.loc[i, 'close']
        df.loc[i, 'option_PL'] = (df.loc[i, 'best_bid_c'] + df.loc[i, 'best_bid_p'] - initial_option_value) if long_op else (-df.loc[i, 'best_offer_c'] - df.loc[i, 'best_offer_p'] - initial_option_value)

        if i == 0:
            df.loc[i, 'avg_cost'] = close_price
            continue

        prev_pos = df.loc[i - 1, 'stock_pos']
        current_pos = df.loc[i, 'stock_pos']
        pos_change = current_pos - prev_pos

        if not pd.isna(df.loc[i - 1, 'avg_cost']):
            df.loc[i, 'stock_PL'] = df.loc[i - 1, 'realized_PL'] + (close_price - df.loc[i - 1, 'avg_cost']) * prev_pos

        daily_option_value = df.loc[i, 'best_bid_c'] + df.loc[i, 'best_bid_p'] if long_op else -df.loc[i, 'best_offer_c'] - df.loc[i, 'best_offer_p']
        df.loc[i, 'option_PL'] = daily_option_value - initial_option_value

        if pos_change != 0:
            if np.sign(pos_change) == np.sign(prev_pos) or prev_pos == 0:
                total_shares = abs(prev_pos) + abs(pos_change)
                total_cost = df.loc[i - 1, 'avg_cost'] * abs(prev_pos) + close_price * abs(pos_change)
                df.loc[i, 'avg_cost'] = total_cost / total_shares if total_shares != 0 else close_price
            else:
                closed_shares = min(abs(prev_pos), abs(pos_change))
                realized_PL_this_step = (close_price - df.loc[i - 1, 'avg_cost']) * closed_shares * np.sign(prev_pos)
                df.loc[i, 'realized_PL'] = df.loc[i - 1, 'realized_PL'] + realized_PL_this_step
                if abs(pos_change) > abs(prev_pos):
                    excess_shares = abs(pos_change) - abs(prev_pos)
                    df.loc[i, 'avg_cost'] = close_price
                    df.loc[i, 'stock_pos'] = excess_shares * np.sign(pos_change)
                else:
                    df.loc[i, 'avg_cost'] = np.nan
        else:
            df.loc[i, 'avg_cost'] = df.loc[i - 1, 'avg_cost']
            df.loc[i, 'stock_pos'] = prev_pos

        df['avg_cost'].ffill(inplace=True)

    final_row_index = len(df) - 1
    final_pos = df.loc[final_row_index, 'stock_pos']
    final_close_price = df.loc[final_row_index, 'close']
    final_avg_cost = df.loc[final_row_index, 'avg_cost']
    final_realized_PL = (final_close_price - final_avg_cost) * final_pos
    df.loc[final_row_index, 'realized_PL'] += final_realized_PL
    df.loc[final_row_index, 'stock_PL'] = df.loc[final_row_index, 'realized_PL']
    df.loc[final_row_index, 'stock_pos'] = 0

    return df


# ============================================

# Analysis Helpers

def performance_summary(cumulative_pl):
    """
    Returns the Performance Stats for a given set of data.
    
    Inputs: 
        data - DataFrame with Date index and corresponding financial data.
    
    Output:
        summary_stats - DataFrame with summary statistics.
    """
    daily_returns = cumulative_pl
    
    summary_stats = pd.DataFrame()
    summary_stats['Mean'] = daily_returns.mean()
    summary_stats['Median'] = daily_returns.median()
    summary_stats['Volatility'] = daily_returns.std() 
    summary_stats['Sharpe Ratio'] = summary_stats['Mean'] / summary_stats['Volatility']
    summary_stats['Skewness'] = daily_returns.skew()
    summary_stats['Excess Kurtosis'] = daily_returns.kurtosis()
    summary_stats['Min'] = daily_returns.min()
    summary_stats['Max'] = daily_returns.max()

    wealth_index = 1000 * (1 + daily_returns)
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    summary_stats['Max Drawdown'] = drawdowns.min()
    
    return summary_stats

def plot_trades_over_time(*cumulative_pls):
    """
    Plots gross option trades and gross stock trades over time for multiple trading strategies.
    
    Inputs:
        cumulative_pls - Tuple of DataFrames, each with Date index and financial data including
                         gross_option_trades and gross_stock_trades.
    """
    fig, axs = plt.subplots(2, len(cumulative_pls), figsize=(12 * len(cumulative_pls), 8))  

    for i, cumulative_pl in enumerate(cumulative_pls):
        axs[0, i].plot(cumulative_pl.index, cumulative_pl['gross_option_trades'], color='blue', marker='o', linestyle='-')
        axs[0, i].set_title(f'Gross Option Trades Over Time: Strategy {i+1}')  
        axs[0, i].set_xlabel('Date')  # X-axis label
        axs[0, i].set_ylabel('Gross Option Trades')  
        axs[0, i].grid(True)  

        axs[1, i].plot(cumulative_pl.index, cumulative_pl['gross_stock_trades'], color='red', marker='o', linestyle='-')
        axs[1, i].set_title(f'Gross Stock Trades Over Time: Strategy {i+1}')  
        axs[1, i].set_xlabel('Date')  
        axs[1, i].set_ylabel('Gross Stock Trades')  
        axs[1, i].grid(True)  

    plt.tight_layout()
    plt.show()

def plot_position_values_over_time(*cumulative_pls):
    """
    Plots option position value and stock position value over time for multiple trading strategies.
    
    Inputs:
        cumulative_pls - Tuple of DataFrames, each with Date index and financial data including
                         option_pos_value and stock_pos_value.
    """
    # Determine the number of trading strategies (dataframes)
    num_strategies = len(cumulative_pls)
    fig, axs = plt.subplots(2, num_strategies, figsize=(12 * num_strategies, 8), squeeze=False)  

    for i, cumulative_pl in enumerate(cumulative_pls):
        axs[0, i].plot(cumulative_pl.index, cumulative_pl['option_pos_value'], color='green', marker='o', linestyle='-')
        axs[0, i].set_title(f'Option Position Value Over Time: Strategy {i+1}')  
        axs[0, i].set_xlabel('Date')  # X-axis label
        axs[0, i].set_ylabel('Option Position Value ($)')  
        axs[0, i].grid(True)  

        axs[1, i].plot(cumulative_pl.index, cumulative_pl['stock_pos_value'], color='purple', marker='o', linestyle='-')
        axs[1, i].set_title(f'Stock Position Value Over Time: Strategy {i+1}')  
        axs[1, i].set_xlabel('Date')  
        axs[1, i].set_ylabel('Stock Position Value ($)')  
        axs[1, i].grid(True)  

    plt.tight_layout()
    plt.show()

def plot_trading_costs_over_time(*cumulative_pls):
    """
    Plots stock trading costs, option trading costs, and net trading costs over time for multiple trading strategies.
    
    Inputs:
        cumulative_pls - Tuple of DataFrames, each with Date index and financial data including
                         stock_trading_costs, option_trading_costs, and net_trading_costs.
    """
    # Determine the number of trading strategies (dataframes)
    num_strategies = len(cumulative_pls)
    fig, axs = plt.subplots(3, num_strategies, figsize=(12 * num_strategies, 12), squeeze=False)  # Adjust for three rows

    for i, cumulative_pl in enumerate(cumulative_pls):
        # Plot stock trading costs
        axs[0, i].plot(cumulative_pl.index, cumulative_pl['stock_trading_costs'], color='blue', marker='o', linestyle='-')
        axs[0, i].set_title(f'Stock Trading Costs Over Time: Strategy {i+1}')  
        axs[0, i].set_xlabel('Date')  
        axs[0, i].set_ylabel('Stock Trading Costs ($)')  
        axs[0, i].grid(True)  

        # Plot option trading costs
        axs[1, i].plot(cumulative_pl.index, cumulative_pl['option_trading_costs'], color='red', marker='o', linestyle='-')
        axs[1, i].set_title(f'Option Trading Costs Over Time: Strategy {i+1}')  
        axs[1, i].set_xlabel('Date')  
        axs[1, i].set_ylabel('Option Trading Costs ($)')  
        axs[1, i].grid(True)

        # Plot net trading costs
        axs[2, i].plot(cumulative_pl.index, cumulative_pl['net_trading_costs'], color='green', marker='o', linestyle='-')
        axs[2, i].set_title(f'Net Trading Costs Over Time: Strategy {i+1}')  
        axs[2, i].set_xlabel('Date')  
        axs[2, i].set_ylabel('Net Trading Costs ($)')  
        axs[2, i].grid(True)  

    plt.tight_layout()
    plt.show()

def calculate_daily_pl(df):
    """
    Calculates daily P&L by taking the difference of cumulative P&L values and adds them to the DataFrame.

    Inputs:
        df - DataFrame with 'option_PL', 'stock_PL', and 'net_PL' columns.

    Returns:
        DataFrame with added 'daily_option_PL', 'daily_stock_PL', and 'daily_net_PL' columns.
    """
    df['daily_option_PL'] = df['option_PL'].diff().fillna(0)  # Using fillna(0) for the first difference which will be NaN
    df['daily_stock_PL'] = df['stock_PL'].diff().fillna(0)
    df['daily_net_PL'] = df['net_PL'].diff().fillna(0)
    return df

def plot_pl_over_time(*cumulative_pls):
    """
    Plots option, stock, and net P&L values over time for multiple trading strategies.
    
    Inputs:
        cumulative_pls - Tuple of DataFrames, each with Date index and financial data.
    """
    # Determine the number of trading strategies (dataframes)
    num_strategies = len(cumulative_pls)
    fig, axs = plt.subplots(3, num_strategies, figsize=(12 * num_strategies, 12), squeeze=False)

    for i, cumulative_pl in enumerate(cumulative_pls):
        axs[0, i].plot(cumulative_pl.index, cumulative_pl['daily_option_PL'], color='orange', marker='o', linestyle='-')
        axs[0, i].set_title(f'Option P&L Over Time: Strategy {i+1}')
        axs[0, i].set_xlabel('Date')
        axs[0, i].set_ylabel('Option P&L ($)')
        axs[0, i].grid(True)
        
        axs[1, i].plot(cumulative_pl.index, cumulative_pl['daily_stock_PL'], color='cyan', marker='o', linestyle='-')
        axs[1, i].set_title(f'Stock P&L Over Time: Strategy {i+1}')
        axs[1, i].set_xlabel('Date')
        axs[1, i].set_ylabel('Stock P&L ($)')
        axs[1, i].grid(True)
        
        axs[2, i].plot(cumulative_pl.index, cumulative_pl['daily_net_PL'], color='magenta', marker='o', linestyle='-')
        axs[2, i].set_title(f'Net P&L Over Time: Strategy {i+1}')
        axs[2, i].set_xlabel('Date')
        axs[2, i].set_ylabel('Net P&L ($)')
        axs[2, i].grid(True)

    plt.tight_layout()
    plt.show()

def plot_cumulative_pl_over_time(*cumulative_pls):
    """
    Plots cumulative option, stock, and net P&L values over time for multiple trading strategies.
    
    Inputs:
        cumulative_pls - Tuple of DataFrames, each with Date index and financial data.
    """
    # Determine the number of trading strategies (dataframes)
    num_strategies = len(cumulative_pls)
    fig, axs = plt.subplots(3, num_strategies, figsize=(12 * num_strategies, 12), squeeze=False)

    for i, cumulative_pl in enumerate(cumulative_pls):
        axs[0, i].plot(cumulative_pl.index, cumulative_pl['option_PL'], color='orange', marker='o', linestyle='-')
        axs[0, i].set_title(f'Cumulative Option P&L Over Time: Strategy {i+1}')
        axs[0, i].set_xlabel('Date')
        axs[0, i].set_ylabel('Cumulative Option P&L ($)')
        axs[0, i].grid(True)
        
        axs[1, i].plot(cumulative_pl.index, cumulative_pl['stock_PL'], color='cyan', marker='o', linestyle='-')
        axs[1, i].set_title(f'Cumulative Stock P&L Over Time: Strategy {i+1}')
        axs[1, i].set_xlabel('Date')
        axs[1, i].set_ylabel('Cumulative Stock P&L ($)')
        axs[1, i].grid(True)
        
        axs[2, i].plot(cumulative_pl.index, cumulative_pl['net_PL'], color='magenta', marker='o', linestyle='-')
        axs[2, i].set_title(f'Cumulative Net P&L Over Time: Strategy {i+1}')
        axs[2, i].set_xlabel('Date')
        axs[2, i].set_ylabel('Cumulative Net P&L ($)')
        axs[2, i].grid(True)

    plt.tight_layout()
    plt.show()

def plot_kapital_over_time(*cumulative_pls):
    """
    Plots initial capital and end capital over time for multiple trading strategies.
    
    Inputs:
        cumulative_pls - Tuple of DataFrames, each with Date index and financial data including
                         initial_kapital and end_kapital.
    """
    # Determine the number of trading strategies (dataframes)
    num_strategies = len(cumulative_pls)
    fig, axs = plt.subplots(2, num_strategies, figsize=(12 * num_strategies, 8), squeeze=False)  # Adjust for two rows

    for i, cumulative_pl in enumerate(cumulative_pls):
        # Plot initial capital
        axs[0, i].plot(cumulative_pl.index, cumulative_pl['initial_kapital'], color='blue', marker='o', linestyle='-')
        axs[0, i].set_title(f'Initial Capital Over Time: Strategy {i+1}')  
        axs[0, i].set_xlabel('Date')  
        axs[0, i].set_ylabel('Initial Capital ($)')  
        axs[0, i].grid(True)  

        # Plot end capital
        axs[1, i].plot(cumulative_pl.index, cumulative_pl['end_kapital'], color='red', marker='o', linestyle='-')
        axs[1, i].set_title(f'End Capital Over Time: Strategy {i+1}')  
        axs[1, i].set_xlabel('Date')  
        axs[1, i].set_ylabel('End Capital ($)')  
        axs[1, i].grid(True)

    plt.tight_layout()
    plt.show()

def plot_trades_and_test_stationarity(*cumulative_pls):
    """
    This function creates subplots for 'gross_option_trades' and 'gross_stock_trades' from multiple 'cumulative_pl' DataFrames.
    It plots the original data along with their 5-day, 30-day, and 100-day rolling averages for each trading strategy.
    Additionally, it performs the Augmented Dickey-Fuller (ADF) test on both datasets to test for stationarity and 
    displays the test results (Test Statistic and P-Value) below each subplot.

    Inputs:
        cumulative_pls - Tuple of DataFrames with columns including 'gross_option_trades' and 'gross_stock_trades'.
    """
    num_strategies = len(cumulative_pls)
    fig, axes = plt.subplots(2, num_strategies, figsize=(20, 10), squeeze=False)

    for i, cumulative_pl in enumerate(cumulative_pls):
        # Plot for gross_option_trades
        axes[0, i].plot(cumulative_pl['gross_option_trades'], label="Gross Option Trades")
        axes[0, i].plot(cumulative_pl['gross_option_trades'].rolling(5).mean(), label="5-day MA")
        axes[0, i].plot(cumulative_pl['gross_option_trades'].rolling(30).mean(), label="30-day MA")
        axes[0, i].plot(cumulative_pl['gross_option_trades'].rolling(100).mean(), label="100-day MA")
        axes[0, i].set_title(f"Strategy {i+1}: Gross Option Trades", fontsize=18)
        axes[0, i].legend(fontsize=14)

        # ADF test on gross_option_trades
        adf_result_option = adfuller(cumulative_pl['gross_option_trades'].dropna(), maxlag=1)
        axes[0, i].text(0.01, -0.3, f'ADF Statistic: {adf_result_option[0]:.2f}\nP-Value: {adf_result_option[1]:.4f}', 
                        transform=axes[0, i].transAxes, fontsize=14)

        # Plot for gross_stock_trades
        axes[1, i].plot(cumulative_pl['gross_stock_trades'], label="Gross Stock Trades")
        axes[1, i].plot(cumulative_pl['gross_stock_trades'].rolling(5).mean(), label="5-day MA")
        axes[1, i].plot(cumulative_pl['gross_stock_trades'].rolling(30).mean(), label="30-day MA")
        axes[1, i].plot(cumulative_pl['gross_stock_trades'].rolling(100).mean(), label="100-day MA")
        axes[1, i].set_title(f"Strategy {i+1}: Gross Stock Trades", fontsize=18)
        axes[1, i].legend(fontsize=14)

        # ADF test on gross_stock_trades
        adf_result_stock = adfuller(cumulative_pl['gross_stock_trades'].dropna(), maxlag=1)
        axes[1, i].text(0.01, -0.3, f'ADF Statistic: {adf_result_stock[0]:.2f}\nP-Value: {adf_result_stock[1]:.4f}', 
                        transform=axes[1, i].transAxes, fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_position_values_and_test_stationarity(*cumulative_pls):
    """
    This function creates subplots for 'option_pos_value' and 'stock_pos_value' from multiple 'cumulative_pl' DataFrames.
    It plots the original data along with their 5-day, 30-day, and 100-day rolling averages for each trading strategy.
    Additionally, it performs the Augmented Dickey-Fuller (ADF) test on both datasets to test for stationarity and 
    displays the test results (Test Statistic and P-Value) below each subplot.

    Inputs:
        cumulative_pls - Tuple of DataFrames with columns including 'option_pos_value' and 'stock_pos_value'.
    """
    num_strategies = len(cumulative_pls)
    fig, axes = plt.subplots(2, num_strategies, figsize=(20 * num_strategies, 10), squeeze=False)

    for i, cumulative_pl in enumerate(cumulative_pls):
        # Plot for option_pos_value
        axes[0, i].plot(cumulative_pl['option_pos_value'], label="Option Position Value")
        axes[0, i].plot(cumulative_pl['option_pos_value'].rolling(5).mean(), label="5-day MA")
        axes[0, i].plot(cumulative_pl['option_pos_value'].rolling(30).mean(), label="30-day MA")
        axes[0, i].plot(cumulative_pl['option_pos_value'].rolling(100).mean(), label="100-day MA")
        axes[0, i].set_title(f"Strategy {i+1}: Option Position Value", fontsize=18)
        axes[0, i].legend(fontsize=14)

        # Perform and display ADF test results for option_pos_value
        adf_result_option = adfuller(cumulative_pl['option_pos_value'].dropna(), maxlag=1)
        axes[0, i].text(0.01, -0.3, f'ADF Statistic: {adf_result_option[0]:.2f}\nP-Value: {adf_result_option[1]:.4f}', 
                        transform=axes[0, i].transAxes, fontsize=14)

        # Plot for stock_pos_value
        axes[1, i].plot(cumulative_pl['stock_pos_value'], label="Stock Position Value")
        axes[1, i].plot(cumulative_pl['stock_pos_value'].rolling(5).mean(), label="5-day MA")
        axes[1, i].plot(cumulative_pl['stock_pos_value'].rolling(30).mean(), label="30-day MA")
        axes[1, i].plot(cumulative_pl['stock_pos_value'].rolling(100).mean(), label="100-day MA")
        axes[1, i].set_title(f"Strategy {i+1}: Stock Position Value", fontsize=18)
        axes[1, i].legend(fontsize=14)

        # Perform and display ADF test results for stock_pos_value
        adf_result_stock = adfuller(cumulative_pl['stock_pos_value'].dropna(), maxlag=1)
        axes[1, i].text(0.01, -0.3, f'ADF Statistic: {adf_result_stock[0]:.2f}\nP-Value: {adf_result_stock[1]:.4f}', 
                        transform=axes[1, i].transAxes, fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_pl_values_and_test_stationarity(*cumulative_pls):
    """
    This function creates subplots for 'daily_option_PL', 'daily_stock_PL', and 'daily_net_PL' from multiple 'cumulative_pl' DataFrames.
    It plots the original data along with their 5-day, 30-day, and 100-day rolling averages for each trading strategy.
    Additionally, it performs the Augmented Dickey-Fuller (ADF) test on all three datasets to test for stationarity and 
    displays the test results (Test Statistic and P-Value) below each subplot.

    Inputs:
        cumulative_pls - Tuple of DataFrames with columns including 'daily_option_PL', 'daily_stock_PL', and 'daily_net_PL'.
    """
    num_strategies = len(cumulative_pls)
    fig, axes = plt.subplots(3, num_strategies, figsize=(20 * num_strategies, 15), squeeze=False)

    for i, cumulative_pl in enumerate(cumulative_pls):
        # Plot for daily_option_PL
        axes[0, i].plot(cumulative_pl['daily_option_PL'], label="Daily Option P&L")
        axes[0, i].plot(cumulative_pl['daily_option_PL'].rolling(5).mean(), label="5-day MA")
        axes[0, i].plot(cumulative_pl['daily_option_PL'].rolling(30).mean(), label="30-day MA")
        axes[0, i].plot(cumulative_pl['daily_option_PL'].rolling(100).mean(), label="100-day MA")
        axes[0, i].set_title(f"Strategy {i+1}: Option P&L", fontsize=18)
        axes[0, i].legend(fontsize=14)

        # Perform and display ADF test results for daily_option_PL
        adf_result_option = adfuller(cumulative_pl['daily_option_PL'].dropna(), maxlag=1)
        axes[0, i].text(0.01, -0.3, f'ADF Statistic: {adf_result_option[0]:.2f}\nP-Value: {adf_result_option[1]:.4f}', 
                        transform=axes[0, i].transAxes, fontsize=14)

        # Plot for daily_stock_PL
        axes[1, i].plot(cumulative_pl['daily_stock_PL'], label="Daily Stock P&L")
        axes[1, i].plot(cumulative_pl['daily_stock_PL'].rolling(5).mean(), label="5-day MA")
        axes[1, i].plot(cumulative_pl['daily_stock_PL'].rolling(30).mean(), label="30-day MA")
        axes[1, i].plot(cumulative_pl['daily_stock_PL'].rolling(100).mean(), label="100-day MA")
        axes[1, i].set_title(f"Strategy {i+1}: Stock P&L", fontsize=18)
        axes[1, i].legend(fontsize=14)

        # Perform and display ADF test results for daily_stock_PL
        adf_result_stock = adfuller(cumulative_pl['daily_stock_PL'].dropna(), maxlag=1)
        axes[1, i].text(0.01, -0.3, f'ADF Statistic: {adf_result_stock[0]:.2f}\nP-Value: {adf_result_stock[1]:.4f}', 
                        transform=axes[1, i].transAxes, fontsize=14)

        # Plot for daily_net_PL
        axes[2, i].plot(cumulative_pl['daily_net_PL'], label="Daily Net P&L")
        axes[2, i].plot(cumulative_pl['daily_net_PL'].rolling(5).mean(), label="5-day MA")
        axes[2, i].plot(cumulative_pl['daily_net_PL'].rolling(30).mean(), label="30-day MA")
        axes[2, i].plot(cumulative_pl['daily_net_PL'].rolling(100).mean(), label="100-day MA")
        axes[2, i].set_title(f"Strategy {i+1}: Net P&L", fontsize=18)
        axes[2, i].legend(fontsize=14)

        # Perform and display ADF test results for daily_net_PL
        adf_result_net = adfuller(cumulative_pl['daily_net_PL'].dropna(), maxlag=1)
        axes[2, i].text(0.01, -0.3, f'ADF Statistic: {adf_result_net[0]:.2f}\nP-Value: {adf_result_net[1]:.4f}', 
                        transform=axes[2, i].transAxes, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_correlations(*cumulative_pls):
    """
    Creates subplots for each trading strategy, each displaying the correlation between two specific variables:
    1. Gross option trades vs. gross stock trades
    2. Option position value vs. stock position value
    3. Option P&L vs. Stock P&L

    Inputs:
        cumulative_pls - Tuple of DataFrames with required financial columns.
    """
    num_strategies = len(cumulative_pls)
    fig, axes = plt.subplots(3, num_strategies, figsize=(8 * num_strategies, 15), squeeze=False)

    for i, cumulative_pl in enumerate(cumulative_pls):
        # Correlation and plot for gross_option_trades and gross_stock_trades
        correlation1 = cumulative_pl['gross_option_trades'].corr(cumulative_pl['gross_stock_trades'])
        axes[0, i].scatter(cumulative_pl['gross_option_trades'], cumulative_pl['gross_stock_trades'],
                           label=f'Correlation: {correlation1:.2f}', color='blue', alpha=0.7)
        axes[0, i].set_xlabel('Gross Option Trades')
        axes[0, i].set_ylabel('Gross Stock Trades')
        axes[0, i].set_title(f'Strategy {i+1}: Gross Option vs. Stock Trades')
        axes[0, i].legend()
        axes[0, i].grid(True)

        # Correlation and plot for option_pos_value and stock_pos_value
        correlation2 = cumulative_pl['option_pos_value'].corr(cumulative_pl['stock_pos_value'])
        axes[1, i].scatter(cumulative_pl['option_pos_value'], cumulative_pl['stock_pos_value'],
                           label=f'Correlation: {correlation2:.2f}', color='red', alpha=0.7)
        axes[1, i].set_xlabel('Option Position Value')
        axes[1, i].set_ylabel('Stock Position Value')
        axes[1, i].set_title(f'Strategy {i+1}: Option vs. Stock Position Values')
        axes[1, i].legend()
        axes[1, i].grid(True)

        # Correlation and plot for option_PL and stock_PL
        correlation3 = cumulative_pl['daily_option_PL'].corr(cumulative_pl['daily_stock_PL'])
        axes[2, i].scatter(cumulative_pl['daily_option_PL'], cumulative_pl['daily_stock_PL'],
                           label=f'Correlation: {correlation3:.2f}', color='green', alpha=0.7)
        axes[2, i].set_xlabel('Daily Option P&L')
        axes[2, i].set_ylabel('Daily Stock P&L')
        axes[2, i].set_title(f'Strategy {i+1}: Daily Option vs. Stock P&L')
        axes[2, i].legend()
        axes[2, i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrices(*cumulative_pls, fff_data):
    """
    Merges each cumulative_pl DataFrame from multiple trading strategies with fff_data DataFrame based on date indices,
    and creates subplots showing correlation matrices between selected financial metrics from each cumulative_pl
    and Fama-French factors from fff_data.

    Parameters:
        cumulative_pls (tuple of DataFrames): Financial metrics data from multiple strategies.
        fff_data (DataFrame): Fama-French factors data.
    """
    num_strategies = len(cumulative_pls)
    fig, axes = plt.subplots(num_strategies, 3, figsize=(30, 10 * num_strategies), squeeze=False)

    for i, cumulative_pl in enumerate(cumulative_pls):
        combined_df = cumulative_pl.merge(fff_data, left_index=True, right_index=True, how='left')

        subset1 = combined_df[['gross_option_trades', 'option_pos_value', 'daily_option_PL', 'Mkt-RF', 'SMB', 'HML']]
        subset2 = combined_df[['gross_stock_trades', 'stock_pos_value', 'daily_stock_PL', 'Mkt-RF', 'SMB', 'HML']]
        subset3 = combined_df[['gross_trades_value', 'daily_net_PL', 'Mkt-RF', 'SMB', 'HML']]

        # Correlation matrix for the first subset
        correlation_matrix1 = subset1.corr()
        sns.heatmap(correlation_matrix1, annot=True, fmt=".2f", cmap='coolwarm',
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axes[i, 0])
        axes[i, 0].set_title(f'Strategy {i+1}: Options Data vs Fama-French Factors')

        # Correlation matrix for the second subset
        correlation_matrix2 = subset2.corr()
        sns.heatmap(correlation_matrix2, annot=True, fmt=".2f", cmap='coolwarm',
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axes[i, 1])
        axes[i, 1].set_title(f'Strategy {i+1}: Stocks Data vs Fama-French Factors')

        # Correlation matrix for the third subset
        correlation_matrix3 = subset3.corr()
        sns.heatmap(correlation_matrix3, annot=True, fmt=".2f", cmap='coolwarm',
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axes[i, 2])
        axes[i, 2].set_title(f'Strategy {i+1}: Gross Trades & Net P&L vs Fama-French Factors')

    plt.tight_layout()
    plt.show()

def run_regression_option_PL(*combined_dfs):
    """
    Performs linear regression using the Fama-French three factors as independent variables
    and option_PL from each combined_df as the dependent variable.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include option_PL and the Fama-French factors.
    """
    for i, combined_df in enumerate(combined_dfs):
        X = combined_df[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        y = combined_df['option_PL']

        model = sm.OLS(y, X).fit()

        print(f'Strategy {i+1}: R-squared of the regression of option_PL on the Fama-French factors: {round(model.rsquared, 6)}\n')
        print(f'Strategy {i+1} regression summary:\n{model.summary()}\n{"-"*100}\n')

def run_regression_stock_PL(*combined_dfs):
    """
    Performs linear regression using the Fama-French three factors as independent variables
    and stock_PL from each combined_df as the dependent variable.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include stock_PL and the Fama-French factors.
    """
    for i, combined_df in enumerate(combined_dfs):
        X = combined_df[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        y = combined_df['stock_PL']

        model = sm.OLS(y, X).fit()

        print(f'Strategy {i+1}: R-squared of the regression of stock_PL on the Fama-French factors: {round(model.rsquared, 6)}\n')
        print(f'Strategy {i+1} regression summary:\n{model.summary()}\n{"-"*100}\n')

def run_regression_net_PL(*combined_dfs):
    """
    Performs linear regression using the Fama-French three factors as independent variables
    and net_PL from each combined_df as the dependent variable.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include net_PL and the Fama-French factors.
    """
    for i, combined_df in enumerate(combined_dfs):
        X = combined_df[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        y = combined_df['net_PL']

        model = sm.OLS(y, X).fit()

        print(f'Strategy {i+1}: R-squared of the regression of net_PL on the Fama-French factors: {round(model.rsquared, 6)}\n')
        print(f'Strategy {i+1} regression summary:\n{model.summary()}\n{"-"*100}\n')

def plot_rolling_volatility(*combined_dfs, window_size=30):
    """
    Calculates and plots the rolling standard deviation of returns (volatility)
    from the net profit and loss (net_PL) of multiple trading strategies.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include the net_PL column.
        window_size (int): The window size for calculating rolling volatility (default is 30 days).
    """
    fig, axes = plt.subplots(len(combined_dfs), 1, figsize=(14, 7 * len(combined_dfs)), squeeze=False)

    for i, combined_df in enumerate(combined_dfs):
        if 'net_PL' not in combined_df.columns:
            raise ValueError("DataFrame must contain a 'net_PL' column")

        combined_df['daily_returns'] = combined_df['net_PL'].pct_change()
        combined_df['rolling_volatility'] = combined_df['daily_returns'].rolling(window=window_size).std()

        axes[i, 0].plot(combined_df.index, combined_df['rolling_volatility'], label=f'{window_size}-Day Rolling Volatility')
        axes[i, 0].set_title(f'Strategy {i+1}: Volatility Clustering - {window_size}-Day Rolling Volatility of Returns')
        axes[i, 0].set_xlabel('Date')
        axes[i, 0].set_ylabel('Rolling Volatility')
        axes[i, 0].legend()
        axes[i, 0].grid(True)

    plt.tight_layout()
    plt.show()

def plot_pnl_distribution(*combined_dfs):
    """
    Creates histograms and density plots for the distribution of daily and monthly P&L 
    from multiple trading strategies, based on the net_PL column of each combined_df DataFrame.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include the net_PL column.
    """
    for i, combined_df in enumerate(combined_dfs):
        if 'net_PL' not in combined_df.columns:
            raise ValueError("DataFrame must contain a 'net_PL' column")

        combined_df['daily_pnl'] = combined_df['net_PL'].diff()
        combined_df['monthly_pnl'] = combined_df['net_PL'].resample('M').last().diff()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        fig.suptitle(f'Strategy {i+1}: Profit and Loss Distribution')

        sns.histplot(combined_df['daily_pnl'].dropna(), bins=50, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Daily P&L Distribution')
        axes[0, 0].set_xlabel('Daily P&L')
        axes[0, 0].set_ylabel('Frequency')

        sns.kdeplot(combined_df['daily_pnl'].dropna(), ax=axes[0, 1], fill=True)
        axes[0, 1].set_title('Daily P&L Density')
        axes[0, 1].set_xlabel('Daily P&L')
        axes[0, 1].set_ylabel('Density')

        sns.histplot(combined_df['monthly_pnl'].dropna(), bins=50, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Monthly P&L Distribution')
        axes[1, 0].set_xlabel('Monthly P&L')
        axes[1, 0].set_ylabel('Frequency')

        sns.kdeplot(combined_df['monthly_pnl'].dropna(), ax=axes[1, 1], fill=True)
        axes[1, 1].set_title('Monthly P&L Density')
        axes[1, 1].set_xlabel('Monthly P&L')
        axes[1, 1].set_ylabel('Density')

        plt.show()

def plot_trade_activity_comparison(*combined_dfs, frequency='D'):
    """
    Creates bar charts to compare 'gross_option_trades' and 'gross_stock_trades' based on the specified frequency
    for multiple trading strategies, without displaying x-axis labels.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include 'gross_option_trades' and 'gross_stock_trades' columns.
        frequency (str): Frequency for resampling data. 'D' for daily, 'W' for weekly.
    """
    for i, combined_df in enumerate(combined_dfs):
        resampled_data = combined_df[['gross_option_trades', 'gross_stock_trades']].resample(frequency).sum()

        # Plotting
        plt.figure(figsize=(14, 7))
        resampled_data.plot(kind='bar', width=0.8)
        plt.title(f'Strategy {i+1}: Comparison of Trading Activity (Frequency: {frequency})')
        plt.xlabel('Date')
        plt.ylabel('Volume of Trades')
        plt.legend(['Gross Option Trades', 'Gross Stock Trades'])
        plt.grid(axis='y', linestyle='--')
        
        # Remove x-axis labels
        plt.xticks(ticks=[], labels=[])  # Remove x-axis tick labels

        plt.show()

def plot_pl_boxplots(combined_df):
    """
    Creates box plots to visualize the distribution, median, and outliers of 'option_PL', 
    'stock_PL', and 'net_PL' from the combined_df DataFrame.

    Parameters:
        combined_df (DataFrame): A DataFrame that includes 'option_PL', 'stock_PL', and 'net_PL' columns.
    """
    plt.figure(figsize=(10, 6))

    pl_data = combined_df[['option_PL', 'stock_PL', 'net_PL']]
    pl_data_melted = pl_data.melt(var_name='Type', value_name='P&L')

    sns.boxplot(x='Type', y='P&L', data=pl_data_melted)
    
    plt.title('Distribution of P&L for Options, Stocks, and Net')
    plt.xlabel('P&L Type')
    plt.ylabel('Profit and Loss')
    plt.grid(axis='y', linestyle='--')

    plt.show()

def plot_pl_boxplots(*combined_dfs):
    """
    Creates box plots to visualize the distribution, median, and outliers of 'option_PL', 
    'stock_PL', and 'net_PL' from multiple combined_df DataFrames.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include 'option_PL', 'stock_PL', and 'net_PL' columns.
    """
    for i, combined_df in enumerate(combined_dfs):
        plt.figure(figsize=(10, 6))

        pl_data = combined_df[['option_PL', 'stock_PL', 'net_PL']]
        pl_data_melted = pl_data.melt(var_name='Type', value_name='P&L')

        sns.boxplot(x='Type', y='P&L', data=pl_data_melted)
        
        plt.title(f'Strategy {i+1}: Distribution of P&L for Options, Stocks, and Net')
        plt.xlabel('P&L Type')
        plt.ylabel('Profit and Loss')
        plt.grid(axis='y', linestyle='--')

        plt.show()

def calculate_profit_factor_analysis(*combined_dfs):
    """
    Calculates and analyzes the profit factor (total gains / total losses) for both options and stocks
    from multiple combined_df DataFrames.

    Parameters:
        combined_dfs (tuple of DataFrames): DataFrames that each include 'daily_option_PL' and 'daily_stock_PL' columns.
    """
    for i, combined_df in enumerate(combined_dfs):
        if 'daily_option_PL' not in combined_df.columns or 'daily_stock_PL' not in combined_df.columns:
            raise ValueError(f"DataFrame for Strategy {i+1} must contain both 'daily_option_PL' and 'daily_stock_PL' columns")

        # Calculate total gains and total losses for options
        option_gains = combined_df[combined_df['daily_option_PL'] > 0]['daily_option_PL'].sum()
        option_losses = combined_df[combined_df['daily_option_PL'] < 0]['daily_option_PL'].sum()

        # Avoid division by zero and negative values for losses
        if option_losses == 0:
            option_profit_factor = 'Infinity'  # No losses
        else:
            option_profit_factor = option_gains / abs(option_losses)

        # Calculate total gains and total losses for stocks
        stock_gains = combined_df[combined_df['daily_stock_PL'] > 0]['daily_stock_PL'].sum()
        stock_losses = combined_df[combined_df['daily_stock_PL'] < 0]['daily_stock_PL'].sum()

        # Avoid division by zero and negative values for losses
        if stock_losses == 0:
            stock_profit_factor = 'Infinity'  # No losses
        else:
            stock_profit_factor = stock_gains / abs(stock_losses)

        print(f"Strategy {i+1}: Option Profit Factor: {option_profit_factor}")
        print(f"Strategy {i+1}: Stock Profit Factor: {stock_profit_factor}\n")

def plot_cum_net_pl_and_spy_returns(combined_1, combined_2, combined_3, spydata):

    # Ensure that the index is datetime
    combined_1.index = pd.to_datetime(combined_1.index)
    combined_2.index = pd.to_datetime(combined_2.index)
    combined_3.index = pd.to_datetime(combined_3.index)
    spydata.index = pd.to_datetime(spydata.index)
    
    # Create a new figure and set its size
    plt.figure(figsize=(14, 8))
    
    # Plotting
    plt.plot(combined_1.index, combined_1['net_PL'], label='Strategy 1 Net P&L', alpha=0.7)
    plt.plot(combined_2.index, combined_2['net_PL'], label='Strategy 2 Net P&L', alpha=0.7)
    plt.plot(combined_3.index, combined_3['net_PL'], label='Strategy 3 Net P&L', alpha=0.7)
    plt.plot(spydata.index, spydata['spy_return'], label='SPY Returns', alpha=0.7)
    
    # Adding legend
    plt.legend()
    
    # Adding title and labels
    plt.title('Net P&L and SPY Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Net P&L / Returns')
    
    # Show plot
    plt.show()

def plot_daily_net_pl_and_spy_returns(combined_1, combined_2, combined_3, spydata):
    # Ensure that the index is datetime
    combined_1.index = pd.to_datetime(combined_1.index)
    combined_2.index = pd.to_datetime(combined_2.index)
    combined_3.index = pd.to_datetime(combined_3.index)
    spydata.index = pd.to_datetime(spydata.index)
    
    # Create a new figure and set its size
    plt.figure(figsize=(14, 8))
    
    # Plotting
    plt.plot(combined_1.index, combined_1['daily_net_PL'], label='Strategy 1 Daily Net P&L', alpha=0.7)
    plt.plot(combined_2.index, combined_2['daily_net_PL'], label='Strategy 2 Daily Net P&L', alpha=0.7)
    plt.plot(combined_3.index, combined_3['daily_net_PL'], label='Strategy 3 Daily Net P&L', alpha=0.7)
    plt.plot(spydata.index, spydata['returns'], label='SPY Daily Returns', alpha=0.7)
    
    # Adding legend
    plt.legend()
    
    # Adding title and labels
    plt.title('Daily Net P&L and SPY Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Net P&L / Returns')
    
    # Show plot
    plt.show()

def plot_net_pos_val(combined_1, combined_2, combined_3):
    """
    Plots net position value over time for three trading strategies.

    Parameters:
        combined_1, combined_2, combined_3 (DataFrame): DataFrames each with Date index and 'net_pos_val' column.
    """

    # Ensure that the index is datetime
    combined_1.index = pd.to_datetime(combined_1.index)
    combined_2.index = pd.to_datetime(combined_2.index)
    combined_3.index = pd.to_datetime(combined_3.index)
    
    # Create a new figure and set its size
    plt.figure(figsize=(14, 8))
    
    # Plotting
    plt.plot(combined_1.index, combined_1['net_pos_value'], label='Strategy 1 Net Position Value', alpha=0.7)
    plt.plot(combined_2.index, combined_2['net_pos_value'], label='Strategy 2 Net Position Value', alpha=0.7)
    plt.plot(combined_3.index, combined_3['net_pos_value'], label='Strategy 3 Net Position Value', alpha=0.7)
    
    # Adding legend
    plt.legend()
    
    # Adding title and labels
    plt.title('Net Position Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Net Position Value ($)')
    
    # Show plot
    plt.show()

# ============================================

def max_drawdown(returns):
    local_max = [n for n in range(len(returns)-1) if ((n==0) and (returns.iloc[0] > returns.iloc[1])) or 
       ((n > 0) and (returns.iloc[n-1] < returns.iloc[n]) and (returns.iloc[n+1] < returns.iloc[n]))] 
    
    local_min = [n for n in range(1, len(returns)) if ((n == len(returns)-1) and (returns.iloc[-1] < returns.iloc[-2])) or
            (returns.iloc[n-1] > returns.iloc[n]) and (returns.iloc[n+1] > returns.iloc[n])]
    
    def next_local_min(n):
        mins_after_n = [m for m in local_min if m > n]
        return mins_after_n[0] if mins_after_n else None
    
    drawdowns = [(n, next_local_min(n)) for n in local_max]
    drawdown_values = [returns.iloc[n] - returns.iloc[m] for (n, m) in drawdowns if m is not None]
    
    return np.max(drawdown_values) if drawdown_values else 0.0

def max_dd(returns):
    # Use .values to work with the underlying numpy array for compatibility
    cum_max = np.maximum.accumulate(returns.values)
    drawdown = cum_max - returns.values
    i = np.argmax(drawdown)  # Position of the max drawdown
    j = np.argmax(returns.values[:i])  # Position of the peak before the max drawdown
    
    plt.plot(returns.index, returns.values)  # Plot using the index for x-axis
    plt.plot([returns.index[i], returns.index[j]], [returns.values[i], returns.values[j]], 'o', color='Red', markersize=10)
    
    max_drawdown_value = np.abs(returns.values[j] - returns.values[i]) / returns.values[j]
    print(max_drawdown_value)
    return max_drawdown_value
    
def combined_performance_metrics(spydata, *dfs, rfr=0.01):
    """
    Calculate traditional and regression-based performance metrics for multiple trading strategies against a given benchmark.

    Args:
        spydata (DataFrame): A DataFrame representing benchmark data, must contain 'returns' column for daily returns.
        *dfs (DataFrame): Variable length DataFrame list, each representing a different trading strategy.
                          Each DataFrame should contain at least 'daily_net_PL' and 'initial_kapital' columns.
        rfr (float): Annual risk-free rate of return, default is 0.01 (1%).

    Returns:
        DataFrame: A DataFrame containing calculated performance metrics for each trading strategy,
                   with the strategy as the index and metrics rounded to six decimal places, except for Beta which is rounded to eight decimal places.
    """
    if 'returns' not in spydata.columns:
        raise ValueError("Benchmark data must contain 'returns' column")
    benchmark_returns_monthly = spydata['spy_return1'].resample('Y').last().pct_change()
    benchmark_returns_monthly[0] = spydata['spy_return1'].resample('Y').last()[0] - 1

    metrics_dict = {}

    for idx, df in enumerate(dfs, start=1):
        if not {'daily_net_PL', 'initial_kapital'}.issubset(df.columns):
            raise ValueError(f"DataFrame {idx} is missing required columns.")

        df['daily_returns'] = df['net_pos_value'].pct_change()
        annualized_return = (df['net_pos_value'].iloc[-1] / df['net_pos_value'].iloc[0]) ** (1/((df['net_pos_value'].index[-1] - df['net_pos_value'].index[0]).total_seconds() / (365.25 * 24 * 60 * 60))) - 1
        annualized_volatility = df['daily_returns'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - rfr) / annualized_volatility

        fund_ret_monthly = df['net_pos_value'].resample('Y').last().pct_change()
        fund_ret_monthly[0] = df['net_pos_value'].resample('Y').last()[0]/1e6 - 1
        X = sm.add_constant(benchmark_returns_monthly)  
        y = fund_ret_monthly
        model = sm.OLS(y, X, missing='drop').fit()
        beta = round(float(model.params[1]), 8)  
        alpha = round(float(model.params[0]), 6)
        tracking_error = round(model.resid.std(), 6)
        information_ratio = round(alpha / tracking_error, 6) if tracking_error != 0 else np.nan
        r_squared = round(model.rsquared, 6)

        metrics_dict[f'Strategy {idx}'] = {
            'Annualized Return': round(annualized_return, 6),
            'Annualized Volatility': round(annualized_volatility, 6),
            'Sharpe Ratio': round(sharpe_ratio, 6),
            'Beta': beta,  
            'Information Ratio': information_ratio,
            'Alpha': alpha,
            'R-squared': r_squared,
            'Tracking Error': tracking_error
        }

    metrics_df = pd.DataFrame(metrics_dict).T
    return metrics_df