================================================================================
                               Pricing Engines
              Futures and options pricing calculators in your browser
================================================================================

WHAT THIS IS
------------
Two calculators that run on your own computer, in Docker, and open in your
browser:

- Futures Calculator - http://localhost:8050
  Prices a futures contract from the spot price, interest rate, storage cost,
  convenience yield and time to maturity, with five models, a model comparison
  and a portfolio simulation.

- Options Calculator - http://localhost:8051
  Prices a European or American call or put with five models, shows its
  Greeks and how its price moves with the underlying price and with
  volatility, and compares the models.

Both are reachable from this computer only. Nothing is fetched from the
internet while they run: the page theme and its font are bundled in assets/.


WHAT YOU NEED
-------------
- Docker Desktop: https://www.docker.com/products/docker-desktop
- Windows 10/11, macOS or Linux, and a current web browser.


STARTING AND STOPPING
---------------------
1. Start Docker Desktop and wait until it is running.
2. Start both calculators:
   - Mac: double-click start.command (if it will not open: right-click > Open)
   - Windows: double-click start.bat
   - Linux: run ./start.command in a terminal, then open the two addresses
     above yourself (the script's browser step works on Mac only)
   The first start builds the calculators and downloads their Python packages,
   which takes a few minutes; later starts take seconds. Both pages then open
   in your browser.
3. To stop both: stop.command (Mac/Linux) or stop.bat (Windows).

The start script checks that Docker is installed and running first. If the
calculators cannot start, it says FAILED and does not open the browser.

From a terminal, in this folder:
     docker-compose up -d --build       start both
     docker-compose down                stop both
     docker logs futures_calculator     messages from the futures calculator
     docker logs options_calculator     messages from the options calculator


THE FUTURES CALCULATOR
----------------------
Inputs, in the top card: Spot Price ($), Risk-Free Rate (%), Time to Maturity
(years), Storage Cost (% per year), Convenience Yield (% per year). The result
updates as soon as an input changes.

Models (Select Pricing Model):

- Cost of Carry: F = S x e^((r + storage cost - convenience yield) x T).
  Shows the price, the basis (F - S) and the carry rate, the futures curve
  from 3 months to 3 years, and a grid of prices for rates and yields up to
  2 percentage points either side of yours.

- Gibson-Schwartz: the spot price and a convenience yield that reverts to its
  long-run level both move randomly, with the correlation you set. The price
  is the average of a Monte Carlo simulation, shown with its standard error
  and a band of 1.96 standard errors either side.

- Merton Jump-Diffusion: the spot price moves smoothly with occasional jumps.
  The jumps are compensated, so the futures price is S x e^((r - yield) x T)
  whatever the jump settings; the simulated price is shown beside it, with
  sample paths and the spread of final prices.

- Multi-Factor: three random factors - spot price, convenience yield and
  interest rate - with the correlations you set. Monte Carlo price and its
  standard error.

- Term Structure: a curve model with three factors (level, slope, curvature).
  The price shown for your maturity is S x e^((r - yield) x T); the factor
  volatilities and the simulation horizon drive the simulated futures curves
  in its charts.

- Model Comparison: the five models side by side for your inputs, each at its
  default settings, with their futures curves. Storage cost is included in
  the Cost of Carry price only - the other four price without it, and the
  page says so.

- Portfolio Simulation: a simplified Monte Carlo of a portfolio of up to five
  assets (S&P 500, Crude Oil, Gold, EUR/USD, Natural Gas). Every asset moves
  as a correlated random walk (geometric Brownian motion), not with its own
  pricing model: drift 3% a year for the S&P 500, 2% for EUR/USD and 5% for
  the others; volatility 25% for Gold and 20% for the others. Tick the assets,
  set weights adding up to 100%, choose the horizon, number of paths, time
  steps and base correlation, then Run Portfolio Simulation. Results:
  expected return, annualised volatility, Sharpe ratio (against a 5%
  risk-free rate), maximum drawdown, value at risk at 95% and 99%,
  conditional value at risk at 95%, and charts.

Advanced Model Parameters (Toggle Advanced) holds the selected model's own
settings: volatilities, mean-reversion speeds, correlations, jump intensity,
mean and spread, and the number of simulation paths. Changing one updates
the result.

Market Analysis (Toggle Analysis): enter the contract's market price; it is
compared with the Cost of Carry fair value, and a difference of more than
$0.50 is flagged as over- or under-priced.

Quick Stats shows "Basic FV": the spot price grown at the risk-free rate as
simple interest, S x (1 + r x T). It is a quick reference, NOT a futures
price: it leaves out storage cost and convenience yield and comes from no
model. At the default inputs it shows $105.00 (100 x (1 + 5% x 1)), while the
Cost of Carry futures price below it is $104.08. For the futures price, read
the model's result, not Basic FV.

The futures simulations draw new random numbers each time, so their prices
move a little from one calculation to the next.

Upload CSV Data, then Use Uploaded: fills the inputs from the last row of a
file with these columns (futures.csv is an example):
     Spot_Price,Risk_Free_Rate,Storage_Cost,Convenience_Yield
Rates are decimals in the file (0.05 = 5%). Time to maturity is not read from
the file. Reset Default puts back 100, 5%, 2% and 3%.

Download Results saves a CSV of the figures the selected model last
computed: its inputs (rates as decimals) and its price; one row per model
for Model Comparison; the risk figures after a Portfolio Simulation run.


THE OPTIONS CALCULATOR
----------------------
Data Input Method: Manual Input, or Upload CSV File (below).

Manual inputs: Underlying Price ($), Strike Price ($), Time to Expiry
(years), Option Type (Call or Put), Exercise Style (European or American),
Risk-Free Rate (%), Dividend Yield (%), Volatility (%). Then click Calculate
Option Price.

Models (Select Pricing Model):

- Black-Scholes-Merton: the closed-form price of a European option with a
  continuous dividend yield.

- Binomial Tree: a Cox-Ross-Rubinstein tree of 10 to 100 steps. The only
  model that prices American options (early exercise is checked at every
  step).

- Monte Carlo Simulation: 100 to 10,000 simulated price paths. European only.

- Heston Stochastic Volatility: volatility itself moves randomly - mean
  reversion speed (kappa), long-term variance (theta; 0.04 is 20%
  volatility), volatility of volatility (xi) and its correlation with the
  price (rho). European only.

- Jump-Diffusion (Merton): the price can jump - jump intensity (lambda,
  jumps per year), jump mean (mu) and jump size spread (delta). European
  only.

- Model Comparison: the five prices side by side, using the settings in the
  advanced panel. With Exercise Style set to American, the Binomial Tree
  prices the American option and the other four the European one, and the
  page says so.

Choosing American with any model other than Binomial Tree shows a message
pointing to the Binomial Tree instead of a price.

Advanced Model Parameters (Toggle Advanced) holds the settings of the
selected model: the binomial steps, the number of simulation paths, and the
Heston and jump settings above.

Results: option price, time value, moneyness (S/K) and intrinsic value; the
Greeks - delta, gamma, theta (the change in price as one day passes), vega
(per 1 percentage point of volatility) and rho (per 1 percentage point of
interest rate); and two charts drawn with the same model and settings, the
price against the underlying (70% to 130% of it) and against volatility
(10% to 80%). Black-Scholes Greeks come from its formulas; the other models'
from small changes to the inputs.

The simulated models (Monte Carlo, Heston, Jump-Diffusion) use a fixed random
seed, so the same inputs always give the same answer.

Export Results saves a CSV of the last calculation: the inputs (rates and
volatility as decimals), price, time value, intrinsic value, moneyness and
Greeks; for Model Comparison, the five prices with their average, range and
standard deviation.

Upload CSV File: drop in a file, pick an option from the list, choose a
model, then Calculate. options.csv is a template with two examples. Columns:
     option_id,underlying_price,strike,time_to_expiry,risk_free_rate,
     dividend_yield,volatility,option_type,exercise_style
- risk_free_rate, dividend_yield and volatility as decimals (0.05 = 5%),
  unlike the page, which takes percentages;
- time_to_expiry in years; option_type call or put; exercise_style european
  or american.
A file missing any of these columns is refused, with the missing ones named.


USING THE PAGES
---------------
- A value typed into the box beside a slider takes effect when you leave the
  box (press Tab or click elsewhere).
- The top card stays in place as you scroll. Toggle Controls folds it away to
  see more of the page.


CHANGING A PORT
---------------
docker-compose.yml publishes the calculators as
     "127.0.0.1:8050:8050"     futures
     "127.0.0.1:8051:8051"     options
To use another port on your computer, change the middle number only - for
example "127.0.0.1:9050:8050" - and keep 127.0.0.1, so the calculator stays
reachable from this computer only. Change the address the start scripts open
(start.command, start.bat) to match, then stop and start again.


TROUBLESHOOTING
---------------
"Docker is not installed" / "Docker Desktop is not running"
   Install or start Docker Desktop, wait until it is running, try again.

"FAILED: the calculators did not start"
   Read the messages above it. If port 8050 or 8051 is already in use by
   another program, stop that program or change the port (above).

The page does not load
   Wait a few seconds after starting, then refresh; the first start takes a
   few minutes. Check in Docker Desktop that futures_calculator and
   options_calculator are running.


FILES
-----
     futures.py            the futures calculator
     options.py            the options calculator
     assets/               the bundled theme (bootstrap.css) and Lato font
     futures.csv           example input for the futures calculator
     options.csv           template for the options calculator
     requirements.txt      Python packages, exact versions
     Dockerfile, docker-compose.yml, .dockerignore
     start.* / stop.*      double-click start and stop scripts


TECHNICAL DETAILS
-----------------
- Python 3.11 in the container; Dash, Dash Bootstrap Components, Plotly,
  NumPy, pandas and SciPy.
- Containers: futures_calculator (port 8050) and options_calculator (port
  8051), both published on this computer only (127.0.0.1).


LICENCE
-------
MIT - see LICENSE. The bundled Bootswatch "Flatly" theme and Bootstrap are
MIT licensed and the Lato font is under the SIL Open Font License - see
THIRD_PARTY_LICENSES. The Python packages are installed by pip from
requirements.txt and keep their own licences.


================================================================================
                    From ChorBazaar - https://chorbazaar.co.in
================================================================================
