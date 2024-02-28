# Stat-222-Project

Textual Analysis of Financial Statements

Zhengxing Cheng, Owen Lin, Isaac Liu, Sean Zhou

### Data
1. Credit Rating
    1. corporate_rating.csv
        * Columns: Rating, Name, Symbol, Rating Agency, Date, Sector, other ratios
        * Size: 2029 x 31
    2. corporateCreditRatingWithFinancialRatios.csv
        * Columns: rating, corporation, ticker, rating agency, date, sector, other variables.
        * Size: 7805 x 25
    3. Concatenate the data, remove duplicates, and add some columns (combined_credit_rating_data.csv)
        * Columns: rating, symbol, rating agency, date, previous/next rating/date, ordinal rating ...
        * Size: 8732 x 13
2. Earnings Call
    1. Nested folders with sector-company-earnings_call. Merge everything in calls_short.csv
        * Columns: company, sector, year, quarter, date, transcript
        * Size: 24580 x 6
3. Tabular Financial Variables
    1. balance_sheet_df.csv
        * Columns: date, symbol, year, period, filing_date, total_current_asset...
        * Size: x 52
    2. income_statement_df.csv
        * Columns: date, symbol, year, period, filing_date, revenue...
        * Size:  x 36
    3. cash_flow_statement_df.csv
        * Columns: date, symbol, year, period, filing_date, netIncome, deferredIncomeTax, ...
        * Size:  x 37
4. All Data
    * Merge everything into all_data by company ticker and date
    * Take the intersection within the period 2010-2016
    * Columns: rating, ticker, sector, year, quarter, call_date, transcript, financial variables, ... 
    * Size: 4532 × 161
    * 319 unique companies

### Data Pipeline Steps

1. `Code/Data Loading and Cleaning/Combine Credit Rating Data.ipynb`, `Code/Earning Calls/calls2sec.ipynb`, `Code/Data Loading and Cleaning/tabular_findata_retrival&loading.ipynb`
2. `Code/Data Loading and Cleaning/Credit Ratings on Earnings Call Dates.ipynb`
3. `Code/Data Loading and Cleaning/Create Combined All Data.ipynb`

### Project Updates

Slides [here](https://docs.google.com/presentation/d/1JJEnThJ8J-kww_SiqMceNVPTG_3i5U472d_8RIgSb-o/edit#slide=id.p).

### Box

All data for this project is kept on Box.

For filepaths, access it using `"~/Box/STAT 222 Capstone"` to ensure code is usable across all machines.

### Conda Environment

The environment `capstone` can be found in [`environment.yml`](https://github.com/current12/Stat-222-Project/blob/main/environment.yml).

If you have the environment activated, you can run `conda env export > environment.yml` while in this directory to update the yaml file.
