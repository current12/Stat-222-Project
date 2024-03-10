# Stat-222-Project

Textual Analysis of Financial Statements

Zhengxing Cheng, Owen Lin, Isaac Liu, Sean Zhou

### Data (2010-2016)
1. Credit Rating
    1. corporate_rating.csv (raw data)
        * Columns: Rating, Name, Symbol, Rating Agency, Date, Sector, other ratios
        * Size: 2029 x 31
    2. corporateCreditRatingWithFinancialRatios.csv (raw data)
        * Columns: rating, corporation, ticker, rating agency, date, sector, other variables.
        * Size: 7805 x 25
    3. Select common columns, concatenate the data, remove duplicates, and add some columns (full data)
        * File: combined_credit_rating_data.csv
        * Columns: rating, symbol, rating agency, date, previous/next rating/date, type, change in rating ...
        * Size: 8732 x 13
    4. Restrict to 2010-2016, create the column "fixed_quarter_date" for later merging
        * File: credit_ratings_on_fixed_quarter_dates.csv
        * Columns: rating, symbol, rating agency, date, previous/next rating/date, type, change in rating ...
        * Size: 9117 × 16
        * 638 unique companies
2. Earnings Call
    1. Nested folders with sector-company-earnings_call. (raw data)
        * File: calls.csv
        * Columns: company, sector, year, quarter, date, transcript
        * Size: 62074 x 6
    2. Restrict to 2010-2016
        * File: calls_short.csv
        * Columns: company, sector, year, quarter, date, transcript
        * Size: 24580 x 6
        * 1322 unique companies
3. Tabular Financial Variables
    1. tabuler_fin_data(balance_sheet).csv (raw data)
        * Columns: date, symbol, year, period, filing_date, total_current_asset...
        * Size: 55377 x 55
    2. tabuler_fin_data(income_statement).csv (raw data)
        * Columns: date, symbol, year, period, filing_date, revenue...
        * Size: 55805 x 39
    3. tabuler_fin_data(cash_flow_statement).csv (raw data)
        * Columns: date, symbol, year, period, filing_date, netIncome, deferredIncomeTax, ...
        * Size: 54808 x 41
    4. Merge the dataset by [symbol, year, period]
        * File: combined_corrected_tabular_financial_statements_data.parquet
        * Columns: date, symbol, year, period, filing_date, financial variables, ...
        * Size: 54218 x 126
    5. Restrict to 2010-2016
        * File: combined_financial_data_short.csv
        * Columns: date, symbol, year, period, filing_date, financial variables, ...
        * Size: 22488 × 126
        * 862 unique companies
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

### March 5 Presentation

Slides [here](https://docs.google.com/presentation/d/1g28qdef5ddqo8jX7AW_3p60fzBnzMxD4_EPGpjcrWkU/edit#slide=id.p).

### Repo Filepaths

Try to use relative paths (`..`, etc.) when referencing other folders in this repository. It's also recommended to clone the repository in `~/repo` (create a folder `repo` in whatever directory `~` references on your machine).

### Box

All data for this project is kept on Box.

For filepaths, access it using `"~/Box/STAT 222 Capstone"` to ensure code is usable across all machines.

### Conda Environment

The environment `capstone` can be found in [`environment.yml`](https://github.com/current12/Stat-222-Project/blob/main/environment.yml).

To make yourself a copy of the environment, run `conda env create -f environment.yml`. To update the environment if the yaml changes, run `conda env update --name capstone --file environment.yml --prune`.

If you have the environment activated, you can run `conda env export > environment.yml` while in this directory to update the yaml file.

### Acknowledgements

Special thanks to the Berkeley Statistical Computing Facility (SCF) for resources.
