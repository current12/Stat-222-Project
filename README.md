# Stat-222-Project

Textual Analysis of Financial Statements

Zhengxing Cheng, Owen Lin, Isaac Liu, Sean Zhou

### Data Loading and Cleaning (2010-2016)
1. Credit Rating
    1. corporate_rating.csv
        * Columns: Rating, Name, Symbol, Rating Agency Name, Date, Sector, other ratios ...
        * Size: 2029 x 31
    2. corporateCreditRatingWithFinancialRatios.csv
        * Columns: Rating, Name, Symbol, Rating Agency Name, Date, Sector, other ratios ...
        * Size: 7805 x 25
    3. combined_credit_rating_data.csv
        * Columns: Rating, Symbol, Rating Agency Name, Date, Source, Rating Rank AAA is 10, previous/next rating/date, type, change in rating ...
        * Size: 8732 x 13
    4. credit_ratings_on_fixed_quarter_dates_with_earnings_call_date.csv
        * Columns: rating, symbol, rating agency, rating_date, fixed_quarter_date, ...
        * Size: 7981 × 16
        * 587 unique companies
2. Earnings Call
    1. earning_call_web.csv
        * Columns: symbol, quarter, year, date, content, source
        * Size: 18346 x 6
    2. calls_short.csv
        * Columns: symbol, quarter', year, date, content, source
        * Size: 24580 x 6
    3. combined_calls.csv
        * Columns: symbol, quarter', year, earnings_call_datetime, content, source, web, earnings_call_date
        * Size: 31067 x 8
        * 1646 unique companies
3. Tabular Financial Variables
    1. tabuler_fin_data(balance_sheet).csv
        * Columns: date, symbol, year, period, filing_date, total_current_asset...
        * Size: 55377 x 55
    2. tabuler_fin_data(income_statement).csv
        * Columns: date, symbol, year, period, filing_date, revenue...
        * Size: 55805 x 39
    3. tabuler_fin_data(cash_flow_statement).csv
        * Columns: date, symbol, year, period, filing_date, netIncome, deferredIncomeTax, ...
        * Size: 54808 x 41
    4. daily_market_cap.parquet
        * Columns: symbol, date, marketcap
        * Size: 1859825 x 3
    5. combined_corrected_tabular_financial_statements_data.parquet
        * Columns: date, symbol, reportedCurrency, period, filing_date, financial variables, Altman_Z ...
        * Size: 20825 x 134
        * 796 unique companies
4. combined_sector_data.csv  
    * Columns: Ticker, Description, Company Name, Sector, Industry Group, Industry, Sub-Industry, Comment
    * Size: 3389 x 8
5. all_data_fixed_quarter_dates.parquet
    * Columns: rating, ticker, sector, year, quarter, call_date, transcript, financial variables, ... 
    * Size: 7334 x 166
    * 536 unique companies
 
<img src="Output\Mind Maps.jpg" style="max-width: 100%; height: auto; display: block;" />

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
