"""Options pricing calculator (Dash app, port 8051): Black-Scholes-Merton,
binomial tree, Monte Carlo, Heston and Merton jump-diffusion, with Greeks,
sensitivity charts, a model comparison, CSV import and CSV export."""
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import base64
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app. The Bootstrap "Flatly" theme and its Lato font are
# bundled in assets/ (Dash loads every .css there automatically), so the page
# fetches nothing from outside.
app = dash.Dash(__name__)
app.title = "Options Pricing Calculator"
app.config.suppress_callback_exceptions = True

# Add custom CSS for animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes heartbeat {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
            a[href*="chorbazaar.co.in"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def validate_option_inputs(S, K, T, r, q=0.0, sigma=0.0):
    """Validate option pricing inputs"""
    if S <= 0:
        raise ValueError("Underlying price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T <= 0:
        raise ValueError("Time to expiry must be positive")
    if sigma < 0:
        raise ValueError("Volatility cannot be negative")
    return True

class BaseOptionModel:
    """Base class for all option pricing models"""
    
    def __init__(self, S, K, T, r, q, sigma, option_type):
        validate_option_inputs(S, K, T, r, q, sigma)
        self.S = S  # Underlying price
        self.K = K  # Strike price
        self.T = T  # Time to expiry
        self.r = r  # Risk-free rate
        self.q = q  # Dividend yield
        self.sigma = sigma  # Volatility
        self.option_type = option_type.lower()  # 'call' or 'put'

    def calculate_greeks(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

class BlackScholesMertonModel(BaseOptionModel):
    """Black-Scholes-Merton model for European options"""
    
    def calculate_price(self):
        """Calculate option price using Black-Scholes formula"""
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.option_type == 'call':
            price = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - 
                    self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:  # put
            price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
                    self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
        
        return price
    
    def calculate_greeks(self):
        """Calculate analytical Greeks for Black-Scholes"""
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        # Delta
        if self.option_type == 'call':
            delta = np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            delta = -np.exp(-self.q * self.T) * norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = (np.exp(-self.q * self.T) * norm.pdf(d1)) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Theta
        term1 = (-self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            term2 = self.q * self.S * norm.cdf(d1) * np.exp(-self.q * self.T)
            term3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            term2 = -self.q * self.S * norm.cdf(-d1) * np.exp(-self.q * self.T)
            term3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        theta = (term1 + term2 + term3) / 365  # Per day
        
        # Vega (same for calls and puts)
        vega = self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T) / 100  # Per 1% vol change
        
        # Rho
        if self.option_type == 'call':
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100  # Per 1% rate change
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class BinomialTreeModel(BaseOptionModel):
    """Cox-Ross-Rubinstein binomial tree for European or American options"""

    def __init__(self, S, K, T, r, q, sigma, option_type, n_steps=100, exercise_style='european'):
        super().__init__(S, K, T, r, q, sigma, option_type)
        self.n_steps = min(n_steps, 100)  # Limit to 100 steps
        self.exercise_style = exercise_style.lower()  # 'european' or 'american'
    
    def calculate_price(self):
        """Calculate option price using binomial tree"""
        dt = self.T / self.n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        prices = np.zeros(self.n_steps + 1)
        for i in range(self.n_steps + 1):
            prices[i] = self.S * (u ** (self.n_steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(self.n_steps + 1)
        for i in range(self.n_steps + 1):
            if self.option_type == 'call':
                option_values[i] = max(0, prices[i] - self.K)
            else:
                option_values[i] = max(0, self.K - prices[i])
        
        # Backward induction
        for step in range(self.n_steps - 1, -1, -1):
            for i in range(step + 1):
                # Calculate hold value
                hold_value = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])

                if self.exercise_style == 'american':
                    # American: exercise early when that is worth more than holding
                    asset_price = self.S * (u ** (step - i)) * (d ** i)
                    if self.option_type == 'call':
                        intrinsic_value = max(0, asset_price - self.K)
                    else:
                        intrinsic_value = max(0, self.K - asset_price)
                    option_values[i] = max(hold_value, intrinsic_value)
                else:
                    # European: exercise only at expiry
                    option_values[i] = hold_value
        
        return option_values[0]
    
    def calculate_greeks(self):
        """Calculate Greeks using finite differences"""
        h_S = 0.01 * self.S  # 1% change in underlying
        h_sigma = 0.01  # 1% change in volatility
        h_r = 0.0001  # 1 basis point change in rate
        h_T = 1/365  # 1 day change in time
        
        base_price = self.calculate_price()
        
        # Delta
        self.S += h_S
        price_up = self.calculate_price()
        self.S -= 2 * h_S
        price_down = self.calculate_price()
        self.S += h_S  # Reset
        delta = (price_up - price_down) / (2 * h_S)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (h_S ** 2)
        
        # Theta
        self.T -= h_T
        if self.T > 0:
            price_theta = self.calculate_price()
            theta = price_theta - base_price  # Per day: the change as one day passes
        else:
            theta = 0
        self.T += h_T  # Reset
        
        # Vega
        self.sigma += h_sigma
        price_vega = self.calculate_price()
        self.sigma -= h_sigma  # Reset
        vega = (price_vega - base_price) / h_sigma
        
        # Rho
        self.r += h_r
        price_rho = self.calculate_price()
        self.r -= h_r  # Reset
        rho = (price_rho - base_price) / h_r
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega / 100,  # Per 1% vol change
            'rho': rho / 100   # Per 1% rate change
        }

class MonteCarloModel(BaseOptionModel):
    """Monte Carlo simulation of geometric Brownian motion, for European options
    only (American options are priced by the binomial tree)"""

    def __init__(self, S, K, T, r, q, sigma, option_type, n_paths=1000):
        super().__init__(S, K, T, r, q, sigma, option_type)
        self.n_paths = min(n_paths, 10000)  # Limit to 10,000 paths
        # Daily steps (252 trading days a year), at least one. Fixed here, so
        # that re-pricing with one day less (theta) changes only the step
        # length and reuses the same random numbers.
        self.n_steps = max(1, int(round(T * 252)))
    
    def calculate_price(self):
        """Calculate option price using Monte Carlo simulation"""
        np.random.seed(42)  # For reproducible results
        
        # Each step is T / n_steps years, so the simulation covers exactly T
        # years (n_steps is set when the model is created)
        n_steps = self.n_steps
        dt = self.T / n_steps
        
        # Generate asset price paths
        paths = np.zeros((self.n_paths, n_steps + 1))
        paths[:, 0] = self.S
        
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(self.n_paths)
            paths[:, t] = paths[:, t-1] * np.exp((self.r - self.q - 0.5 * self.sigma**2) * dt + 
                                                self.sigma * np.sqrt(dt) * Z)
        
        # Calculate payoffs at expiry
        if self.option_type == 'call':
            payoffs = np.maximum(paths[:, -1] - self.K, 0)
        else:
            payoffs = np.maximum(self.K - paths[:, -1], 0)

        # Discount and average
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return option_price
    
    def calculate_greeks(self):
        """Calculate Greeks using finite differences"""
        h_S = 0.01 * self.S
        h_sigma = 0.01
        h_r = 0.0001
        h_T = 1/365
        
        base_price = self.calculate_price()
        
        # Delta
        self.S += h_S
        price_up = self.calculate_price()
        self.S -= 2 * h_S
        price_down = self.calculate_price()
        self.S += h_S
        delta = (price_up - price_down) / (2 * h_S)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (h_S ** 2)
        
        # Theta
        self.T -= h_T
        if self.T > 0:
            price_theta = self.calculate_price()
            theta = price_theta - base_price  # Per day: the change as one day passes
        else:
            theta = 0
        self.T += h_T
        
        # Vega
        self.sigma += h_sigma
        price_vega = self.calculate_price()
        self.sigma -= h_sigma
        vega = (price_vega - base_price) / h_sigma
        
        # Rho
        self.r += h_r
        price_rho = self.calculate_price()
        self.r -= h_r
        rho = (price_rho - base_price) / h_r
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega / 100,
            'rho': rho / 100
        }

class HestonModel(BaseOptionModel):
    """Heston stochastic volatility model"""
    
    def __init__(self, S, K, T, r, q, sigma, option_type, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, n_paths=1000):
        super().__init__(S, K, T, r, q, sigma, option_type)
        self.kappa = kappa  # Mean reversion speed
        self.theta = theta  # Long-term variance (0.04 = 20% volatility)
        self.xi = xi       # Volatility of volatility
        self.rho = rho     # Correlation between asset and volatility
        self.n_paths = min(n_paths, 10000)
        # Daily steps, fixed here so that theta reuses the same random numbers
        self.n_steps = max(1, int(round(T * 252)))
        self.v0 = sigma ** 2  # Initial variance
    
    def calculate_price(self):
        """Calculate option price using Heston Monte Carlo"""
        np.random.seed(42)
        
        # Each step is T / n_steps years, so the simulation covers exactly T
        # years (n_steps is set when the model is created)
        n_steps = self.n_steps
        dt = self.T / n_steps
        
        # Initialize paths
        S_paths = np.zeros((self.n_paths, n_steps + 1))
        v_paths = np.zeros((self.n_paths, n_steps + 1))
        
        S_paths[:, 0] = self.S
        v_paths[:, 0] = self.v0
        
        # Generate correlated random numbers
        for t in range(1, n_steps + 1):
            Z1 = np.random.standard_normal(self.n_paths)
            Z2 = np.random.standard_normal(self.n_paths)
            
            # Correlated random numbers
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # Update variance (Euler scheme with reflection)
            v_paths[:, t] = np.maximum(
                v_paths[:, t-1] + self.kappa * (self.theta - v_paths[:, t-1]) * dt + 
                self.xi * np.sqrt(v_paths[:, t-1]) * np.sqrt(dt) * W2, 0
            )
            
            # Update asset price
            S_paths[:, t] = S_paths[:, t-1] * np.exp(
                (self.r - self.q - 0.5 * v_paths[:, t-1]) * dt + 
                np.sqrt(v_paths[:, t-1]) * np.sqrt(dt) * W1
            )
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(S_paths[:, -1] - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_paths[:, -1], 0)
        
        return np.exp(-self.r * self.T) * np.mean(payoffs)
    
    def calculate_greeks(self):
        """Calculate Greeks using finite differences"""
        return self._finite_difference_greeks()
    
    def _finite_difference_greeks(self):
        """Helper method for finite difference Greeks calculation"""
        h_S = 0.01 * self.S
        h_sigma = 0.01
        h_r = 0.0001
        h_T = 1/365
        
        base_price = self.calculate_price()
        
        # Delta
        self.S += h_S
        price_up = self.calculate_price()
        self.S -= 2 * h_S
        price_down = self.calculate_price()
        self.S += h_S
        delta = (price_up - price_down) / (2 * h_S)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (h_S ** 2)
        
        # Theta
        self.T -= h_T
        if self.T > 0:
            price_theta = self.calculate_price()
            theta = price_theta - base_price  # Per day: the change as one day passes
        else:
            theta = 0
        self.T += h_T
        
        # Vega (approximate using initial volatility)
        original_v0 = self.v0
        self.v0 = (self.sigma + h_sigma) ** 2
        price_vega = self.calculate_price()
        self.v0 = original_v0
        vega = (price_vega - base_price) / h_sigma
        
        # Rho
        self.r += h_r
        price_rho = self.calculate_price()
        self.r -= h_r
        rho = (price_rho - base_price) / h_r
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega / 100,
            'rho': rho / 100
        }

class JumpDiffusionModel(BaseOptionModel):
    """Merton Jump-Diffusion model for options"""
    
    def __init__(self, S, K, T, r, q, sigma, option_type, jump_intensity=0.1, jump_mean=0.0, jump_std=0.1, n_paths=1000):
        super().__init__(S, K, T, r, q, sigma, option_type)
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.n_paths = min(n_paths, 10000)
        # Daily steps, fixed here so that theta reuses the same random numbers
        self.n_steps = max(1, int(round(T * 252)))
    
    def calculate_price(self):
        """Calculate option price using jump-diffusion Monte Carlo"""
        np.random.seed(42)
        
        # Each step is T / n_steps years, so the simulation covers exactly T
        # years (n_steps is set when the model is created)
        n_steps = self.n_steps
        dt = self.T / n_steps
        
        # Adjust drift for jump risk
        jump_adjustment = self.jump_intensity * (np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1)
        drift = self.r - self.q - 0.5 * self.sigma**2 - jump_adjustment
        
        # Initialize paths
        paths = np.zeros((self.n_paths, n_steps + 1))
        paths[:, 0] = self.S
        
        for t in range(1, n_steps + 1):
            # Continuous part
            Z = np.random.standard_normal(self.n_paths)
            continuous_return = drift * dt + self.sigma * np.sqrt(dt) * Z
            
            # Jump part
            n_jumps = np.random.poisson(self.jump_intensity * dt, self.n_paths)
            jump_return = np.zeros(self.n_paths)
            
            for i in range(self.n_paths):
                if n_jumps[i] > 0:
                    jumps = np.random.normal(self.jump_mean, self.jump_std, n_jumps[i])
                    jump_return[i] = np.sum(jumps)
            
            # Update paths
            total_return = continuous_return + jump_return
            paths[:, t] = paths[:, t-1] * np.exp(total_return)
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(paths[:, -1] - self.K, 0)
        else:
            payoffs = np.maximum(self.K - paths[:, -1], 0)
        
        return np.exp(-self.r * self.T) * np.mean(payoffs)
    
    def calculate_greeks(self):
        """Calculate Greeks using finite differences"""
        return self._finite_difference_greeks()
    
    def _finite_difference_greeks(self):
        """Helper method for finite difference Greeks calculation"""
        h_S = 0.01 * self.S
        h_sigma = 0.01
        h_r = 0.0001
        h_T = 1/365
        
        base_price = self.calculate_price()
        
        # Delta
        self.S += h_S
        price_up = self.calculate_price()
        self.S -= 2 * h_S
        price_down = self.calculate_price()
        self.S += h_S
        delta = (price_up - price_down) / (2 * h_S)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (h_S ** 2)
        
        # Theta
        self.T -= h_T
        if self.T > 0:
            price_theta = self.calculate_price()
            theta = price_theta - base_price  # Per day: the change as one day passes
        else:
            theta = 0
        self.T += h_T
        
        # Vega
        self.sigma += h_sigma
        price_vega = self.calculate_price()
        self.sigma -= h_sigma
        vega = (price_vega - base_price) / h_sigma
        
        # Rho
        self.r += h_r
        price_rho = self.calculate_price()
        self.r -= h_r
        rho = (price_rho - base_price) / h_r
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega / 100,
            'rho': rho / 100
        }

# =====================================================
# DASH LAYOUT
# =====================================================

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id='uploaded-options-data', data=None),
        dcc.Store(id='selected-option-params', data=None),
        dcc.Store(id='greeks-data', data=None),
        dcc.Store(id='calculation-results', data=None),
        
        # Header
        dbc.Row([
            dbc.Col([
                html.H1(
                    "📊 Options Pricing Calculator",
                    className="text-center my-3",
                    style={'color': '#2c3e50', 'fontWeight': 'bold'}
                )
            ])
        ]),
        
        # Top Controls Section (Sticky and Collapsible)
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.H5("⚙️ Option Configuration & Parameters", className="mb-0")
                    ], width=10),
                    dbc.Col([
                        dbc.Button(
                            "Toggle Controls",
                            id="controls-collapse-button",
                            className="btn-outline-secondary btn-sm",
                            n_clicks=0,
                        )
                    ], width=2, className="text-end")
                ])
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    # Data Input Method Selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("Data Input Method:", className="fw-bold"),
                            dcc.RadioItems(
                                id='input-method',
                                options=[
                                    {'label': '📁 Upload CSV File', 'value': 'csv'},
                                    {'label': '📝 Manual Input', 'value': 'manual'}
                                ],
                                value='manual',
                                inline=True,
                                className="mb-3"
                            )
                        ], width=12)
                    ]),
                    
                    # Model Selection Row (used with both input methods)
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Pricing Model:", className="fw-bold"),
                            dcc.Dropdown(
                                id='model-selector',
                                options=[
                                    {'label': 'Black-Scholes-Merton', 'value': 'black_scholes'},
                                    {'label': 'Binomial Tree', 'value': 'binomial'},
                                    {'label': 'Monte Carlo Simulation', 'value': 'monte_carlo'},
                                    {'label': 'Heston Stochastic Volatility', 'value': 'heston'},
                                    {'label': 'Jump-Diffusion (Merton)', 'value': 'jump_diffusion'},
                                    {'label': 'Model Comparison', 'value': 'model_comparison'},
                                ],
                                value='black_scholes',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=12)
                    ]),

                    # CSV Upload Section
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="csv-upload-section", children=[]),
                            # CSV option selector (always in the page; shown once a file is loaded)
                            dcc.Dropdown(id='csv-option-selector', value=0, style={'display': 'none'})
                        ], width=12)
                    ]),

                    # Manual Input Section
                    html.Div(id="manual-input-section", children=[
                        # Basic Option Parameters Row
                        dbc.Row([
                            dbc.Col([
                                html.H6("📊 Basic Option Parameters", className="text-primary mb-2"),
                                html.Label("Underlying Price ($)"),
                                dbc.Input(id='underlying-price', type='number', value=100.0, min=0.01, step=0.01, size="sm")
                            ], width=3),
                            dbc.Col([
                                html.Label("Strike Price ($)", className="mt-4"),
                                dbc.Input(id='strike-price', type='number', value=100.0, min=0.01, step=0.01, size="sm")
                            ], width=3),
                            dbc.Col([
                                html.Label("Time to Expiry (Years)", className="mt-4"),
                                dcc.Slider(
                                    id='time-to-expiry', 
                                    min=0.01, max=5.0, step=0.01, value=0.25,
                                    marks={0.25: '3M', 0.5: '6M', 1.0: '1Y', 2.0: '2Y', 5.0: '5Y'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=3),
                            dbc.Col([
                                html.H6("📈 Option Type", className="text-success mb-2"),
                                html.Label("Option Type", className="small"),
                                dcc.Dropdown(
                                    id='option-type',
                                    options=[
                                        {'label': 'Call', 'value': 'call'},
                                        {'label': 'Put', 'value': 'put'}
                                    ],
                                    value='call',
                                    clearable=False,
                                    className="mb-2"
                                ),
                                html.Label("Exercise Style", className="small"),
                                dcc.Dropdown(
                                    id='exercise-style',
                                    options=[
                                        {'label': 'European', 'value': 'european'},
                                        {'label': 'American', 'value': 'american'}
                                    ],
                                    value='european',
                                    clearable=False
                                )
                            ], width=3)
                        ], className="mb-4"),
                        
                        # Market Parameters Row
                        dbc.Row([
                            dbc.Col([
                                html.H6("🏦 Market Parameters", className="text-info mb-2"),
                                html.Label("Risk-Free Rate (%)", className="small"),
                                dcc.Slider(
                                    id='risk-free-rate', 
                                    min=0.0, max=10.0, step=0.1, value=5.0,
                                    marks={i: f'{i}%' for i in range(0, 11, 2)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Dividend Yield (%)", className="small mt-4"),
                                dcc.Slider(
                                    id='dividend-yield', 
                                    min=0.0, max=10.0, step=0.1, value=2.0,
                                    marks={0: '0%', 5: '5%', 10: '10%'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Volatility (%)", className="small mt-4"),
                                dcc.Slider(
                                    id='volatility', 
                                    min=1.0, max=100.0, step=0.5, value=25.0,
                                    marks={10: '10%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=4)
                        ], className="mb-4")
                    ])
                ])
            ], id="controls-collapse", is_open=True),
        ], className="mb-4", style={'position': 'sticky', 'top': '10px', 'z-index': '1000'}),
        
        # Advanced Parameters Section (Model-specific)
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.H5("🔧 Advanced Model Parameters", className="mb-0")
                    ], width=10),
                    dbc.Col([
                        dbc.Button(
                            "Toggle Advanced",
                            id="advanced-collapse-button",
                            className="btn-outline-info btn-sm",
                            n_clicks=0,
                        )
                    ], width=2, className="text-end")
                ])
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    # Every slider stays in the page (only the groups for the
                    # selected model are shown), so the calculation can always
                    # read its value.
                    html.Div(id='advanced-parameters', children=[
                        html.P("No advanced parameters for this model.", id='adv-none',
                               className="text-muted"),
                        html.Div(id='adv-binomial', style={'display': 'none'}, children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Binomial Tree: Number of Steps", className="fw-bold"),
                                    dcc.Slider(
                                        id='binomial-steps',
                                        min=10, max=100, step=1, value=50,
                                        marks={10: '10', 50: '50', 100: '100'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=12)
                            ], className="mb-3")
                        ]),
                        html.Div(id='adv-paths', style={'display': 'none'}, children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Simulation: Number of Paths", className="fw-bold"),
                                    dcc.Slider(
                                        id='monte-carlo-paths',
                                        min=100, max=10000, step=100, value=1000,
                                        marks={100: '100', 1000: '1K', 5000: '5K', 10000: '10K'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=12)
                            ], className="mb-3")
                        ]),
                        html.Div(id='adv-heston', style={'display': 'none'}, children=[
                            html.Hr(),
                            html.H6("Heston Model Parameters", className="text-info"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Mean Reversion Speed (κ)", className="small"),
                                    dcc.Slider(
                                        id='heston-kappa',
                                        min=0.1, max=5.0, step=0.1, value=2.0,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Long-term Variance (θ)", className="small"),
                                    dcc.Slider(
                                        id='heston-theta',
                                        min=0.01, max=0.1, step=0.01, value=0.04,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Volatility of Volatility (ξ)", className="small"),
                                    dcc.Slider(
                                        id='heston-xi',
                                        min=0.1, max=1.0, step=0.1, value=0.3,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Correlation (ρ)", className="small"),
                                    dcc.Slider(
                                        id='heston-rho',
                                        min=-0.9, max=0.9, step=0.1, value=-0.5,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=3)
                            ], className="mb-3")
                        ]),
                        html.Div(id='adv-jump', style={'display': 'none'}, children=[
                            html.Hr(),
                            html.H6("Jump-Diffusion Parameters", className="text-warning"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Jump Intensity (λ)", className="small"),
                                    dcc.Slider(
                                        id='jump-intensity',
                                        min=0.01, max=1.0, step=0.01, value=0.1,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Jump Mean (μ)", className="small"),
                                    dcc.Slider(
                                        id='jump-mean',
                                        min=-0.2, max=0.2, step=0.01, value=0.0,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Jump Std Dev (δ)", className="small"),
                                    dcc.Slider(
                                        id='jump-std',
                                        min=0.01, max=0.5, step=0.01, value=0.1,
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], width=4)
                            ])
                        ]),
                    ])
                ])
            ], id="advanced-collapse", is_open=False),
        ], className="mb-4"),
        
        # Pricing Results Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "💰 Calculate Option Price",
                            id="calculate-button",
                            color="primary",
                            size="lg",
                            className="me-2"
                        ),
                        dbc.Button(
                            "📊 Export Results",
                            id="export-button",
                            color="success",
                            size="lg",
                            outline=True,
                            disabled=True
                        )
                    ], className="mb-3 w-100"),
                    dcc.Download(id="download-dataframe-csv"),
                    dcc.Loading(
                        id="loading-calculation",
                        type="default",
                        children=[
                            html.Div(id='pricing-results', children=[])
                        ]
                    )
                ])
            ], width=12)
        ]),
        
        # Visualization Section
        dbc.Row([
            dbc.Col([
                html.Div(id='visualizations', children=[])
            ], width=12)
        ], className="mt-4"),
        
        # Footer
        html.Footer([
            html.Hr(style={'margin': '40px 0 0 0', 'border': 'none'}),
            dbc.Row([
                dbc.Col([
                    html.P([
                        "Made with ", 
                        html.Span("♥", style={'color': '#ff6b6b', 'fontSize': '1.2em', 'animation': 'heartbeat 1.5s ease-in-out infinite'}),
                        " by ",
                        html.A("ChorBazaar",
                               href="https://chorbazaar.co.in",
                               target="_blank",
                               style={
                                   'textDecoration': 'none', 
                                   'color': '#ffffff', 
                                   'fontWeight': '600',
                                   'padding': '4px 12px',
                                   'borderRadius': '15px',
                                   'background': 'linear-gradient(45deg, #667eea 0%, #764ba2 100%)',
                                   'transition': 'all 0.3s ease',
                                   'boxShadow': '0 2px 8px rgba(102, 126, 234, 0.3)'
                               })
                    ], className="text-center mb-0", 
                       style={'color': '#2d3748', 'fontSize': '1rem', 'fontWeight': '600'})
                ], width=12)
            ])
        ], style={
            'marginTop': '12px', 
            'paddingTop': '6px',
            'paddingBottom': '18px',
            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'borderRadius': '15px 15px 0 0',
            'boxShadow': '0 -4px 20px rgba(102, 126, 234, 0.15)'
        })
    ]
)

# =====================================================
# DASH CALLBACKS
# =====================================================

# Toggle controls collapse
@app.callback(
    Output("controls-collapse", "is_open"),
    Input("controls-collapse-button", "n_clicks"),
    State("controls-collapse", "is_open"),
)
def toggle_controls(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Toggle advanced parameters collapse
@app.callback(
    Output("advanced-collapse", "is_open"),
    Input("advanced-collapse-button", "n_clicks"),
    State("advanced-collapse", "is_open"),
)
def toggle_advanced(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Show/hide CSV upload section based on input method
@app.callback(
    Output('csv-upload-section', 'children'),
    Input('input-method', 'value')
)
def update_csv_section(input_method):
    if input_method == 'csv':
        return [
            html.H6("📁 CSV File Upload", className="text-warning mb-2"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='upload-status'),
            html.Div(id='option-selector-div', style={'margin-top': '20px'})
        ]
    return []

# Show/hide manual input section
@app.callback(
    Output('manual-input-section', 'style'),
    Input('input-method', 'value')
)
def toggle_manual_section(input_method):
    if input_method == 'manual':
        return {'display': 'block'}
    return {'display': 'none'}

# Handle CSV file upload
@app.callback(
    [Output('upload-status', 'children'),
     Output('option-selector-div', 'children'),
     Output('uploaded-options-data', 'data'),
     Output('csv-option-selector', 'options'),
     Output('csv-option-selector', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_csv_upload(contents, filename):
    if contents is None:
        return "", "", None, [], {'display': 'none'}
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['option_id', 'underlying_price', 'strike', 'time_to_expiry', 
                        'risk_free_rate', 'dividend_yield', 'volatility', 'option_type', 'exercise_style']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return dbc.Alert(f"❌ Missing required columns: {missing_cols}", color="danger"), "", None, [], {'display': 'none'}
        
        # Create options for the dropdown
        dropdown_options = [{'label': f"Option {row['option_id']} - {row['option_type'].title()} Strike ${row['strike']}", 
                           'value': idx} for idx, row in df.iterrows()]
        
        # Create visible option selector display
        option_selector = [
            html.Label("Select Option to Price:", className="fw-bold"),
            html.P(f"Use the dropdown below to select from {len(df)} loaded options:")
        ]
        
        return (
            dbc.Alert(f"✅ Successfully loaded {len(df)} options from {filename}", color="success"),
            option_selector,
            df.to_dict('records'),
            dropdown_options,
            {'display': 'block'}
        )
        
    except Exception as e:
        return dbc.Alert(f"❌ Error processing file: {str(e)}", color="danger"), "", None, [], {'display': 'none'}

# Which advanced groups each model uses (Model Comparison uses them all)
ADVANCED_GROUPS = {
    'adv-none': ['black_scholes'],
    'adv-binomial': ['binomial', 'model_comparison'],
    'adv-paths': ['monte_carlo', 'heston', 'jump_diffusion', 'model_comparison'],
    'adv-heston': ['heston', 'model_comparison'],
    'adv-jump': ['jump_diffusion', 'model_comparison'],
}

# Show the selected model's advanced groups
@app.callback(
    [Output(group, 'style') for group in ADVANCED_GROUPS],
    Input('model-selector', 'value')
)
def update_advanced_parameters(model_name):
    return [{'display': 'block'} if model_name in models else {'display': 'none'}
            for models in ADVANCED_GROUPS.values()]


# Main pricing calculation callback
@app.callback(
    [Output('pricing-results', 'children'),
     Output('visualizations', 'children'),
     Output('export-button', 'disabled'),
     Output('calculation-results', 'data')],
    Input('calculate-button', 'n_clicks'),
    [State('input-method', 'value'),
     State('model-selector', 'value'),
     State('underlying-price', 'value'),
     State('strike-price', 'value'),
     State('time-to-expiry', 'value'),
     State('risk-free-rate', 'value'),
     State('dividend-yield', 'value'),
     State('volatility', 'value'),
     State('option-type', 'value'),
     State('exercise-style', 'value'),
     State('uploaded-options-data', 'data'),
     State('csv-option-selector', 'value'),
     State('binomial-steps', 'value'),
     State('monte-carlo-paths', 'value'),
     State('heston-kappa', 'value'),
     State('heston-theta', 'value'),
     State('heston-xi', 'value'),
     State('heston-rho', 'value'),
     State('jump-intensity', 'value'),
     State('jump-mean', 'value'),
     State('jump-std', 'value')],
    prevent_initial_call=True
)
def calculate_option_price(n_clicks, input_method, model_name, S, K, T, r, q, sigma, 
                          option_type, exercise_style, uploaded_data, csv_option_idx,
                          binomial_steps, mc_paths, heston_kappa, heston_theta, 
                          heston_xi, heston_rho, jump_intensity, jump_mean, jump_std):
    
    if n_clicks is None or n_clicks == 0:
        return "", "", True, None
    
    try:
        # Get parameters from CSV if using CSV input
        if input_method == 'csv' and uploaded_data:
            # Use selected option index or default to first option
            csv_idx = csv_option_idx if csv_option_idx is not None else 0
            selected_option = uploaded_data[csv_idx]
            S = selected_option['underlying_price']
            K = selected_option['strike']
            T = selected_option['time_to_expiry']
            r = selected_option['risk_free_rate']
            q = selected_option['dividend_yield']
            sigma = selected_option['volatility']
            option_type = selected_option['option_type'].lower()
            exercise_style = selected_option['exercise_style'].lower()
        else:
            # Convert percentages to decimals for manual input
            r = r / 100.0 if r is not None else 0.05
            q = q / 100.0 if q is not None else 0.02
            sigma = sigma / 100.0 if sigma is not None else 0.25
        
        # Validate inputs
        if not all([S, K, T, r is not None, q is not None, sigma]):
            return dbc.Alert("❌ Please fill in all required parameters", color="danger"), "", True, None
        
        # Store calculation parameters for export
        calc_data = {
            'model': model_name,
            'underlying_price': S,
            'strike_price': K,
            'time_to_expiry': T,
            'risk_free_rate': r,
            'dividend_yield': q,
            'volatility': sigma,
            'option_type': option_type,
            'exercise_style': exercise_style,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # The advanced sliders' values, with defaults when a value is missing
        params = {
            'binomial_steps': binomial_steps if binomial_steps is not None else 50,
            'mc_paths': mc_paths if mc_paths is not None else 1000,
            'heston_kappa': heston_kappa if heston_kappa is not None else 2.0,
            'heston_theta': heston_theta if heston_theta is not None else 0.04,
            'heston_xi': heston_xi if heston_xi is not None else 0.3,
            'heston_rho': heston_rho if heston_rho is not None else -0.5,
            'jump_intensity': jump_intensity if jump_intensity is not None else 0.1,
            'jump_mean': jump_mean if jump_mean is not None else 0.0,
            'jump_std': jump_std if jump_std is not None else 0.1,
        }

        if model_name == 'model_comparison':
            results, viz, calc_data = calculate_model_comparison(S, K, T, r, q, sigma, option_type, exercise_style,
                                                                 params, calc_data)
            return results, viz, False, calc_data

        # Only the binomial tree prices American options
        if exercise_style == 'american' and model_name in EUROPEAN_ONLY_MODELS:
            return dbc.Alert(
                f"{EUROPEAN_ONLY_MODELS[model_name]} prices European options only. "
                "Use Binomial Tree to price an American option.",
                color="warning"
            ), "", True, None

        results, viz, calc_data = calculate_single_model(model_name, S, K, T, r, q, sigma, option_type,
                                                         exercise_style, params, calc_data)
        return results, viz, False, calc_data

    except Exception as e:
        return dbc.Alert(f"❌ Error in calculation: {str(e)}", color="danger"), "", True, None

# Models that price European options only, with their display names
EUROPEAN_ONLY_MODELS = {
    'black_scholes': 'Black-Scholes-Merton',
    'monte_carlo': 'Monte Carlo',
    'heston': 'Heston',
    'jump_diffusion': 'Jump-Diffusion',
}

def build_model(model_name, S, K, T, r, q, sigma, option_type, exercise_style, params):
    """Create the selected model with the user's settings. The price, the
    Greeks and the sensitivity charts are all built here, so all three use
    the same settings."""
    if model_name == 'black_scholes':
        return BlackScholesMertonModel(S, K, T, r, q, sigma, option_type)
    if model_name == 'binomial':
        return BinomialTreeModel(S, K, T, r, q, sigma, option_type,
                                 params['binomial_steps'], exercise_style)
    if model_name == 'monte_carlo':
        return MonteCarloModel(S, K, T, r, q, sigma, option_type, params['mc_paths'])
    if model_name == 'heston':
        return HestonModel(S, K, T, r, q, sigma, option_type,
                           params['heston_kappa'], params['heston_theta'],
                           params['heston_xi'], params['heston_rho'], params['mc_paths'])
    if model_name == 'jump_diffusion':
        return JumpDiffusionModel(S, K, T, r, q, sigma, option_type,
                                  params['jump_intensity'], params['jump_mean'],
                                  params['jump_std'], params['mc_paths'])
    raise ValueError(f"Unknown model: {model_name}")

def calculate_single_model(model_name, S, K, T, r, q, sigma, option_type, exercise_style, params, calc_data):

    model = build_model(model_name, S, K, T, r, q, sigma, option_type, exercise_style, params)

    # Calculate price and Greeks
    option_price = model.calculate_price()
    greeks = model.calculate_greeks()
    
    # Calculate additional metrics
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    time_value = option_price - intrinsic
    moneyness = S / K
    
    # Create results display
    pricing_results = [
        html.H4(f"💰 {model_name.replace('_', ' ').title()} Results", className="text-primary mb-3"),
        
        # Main metrics cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Option Price", className="card-title"),
                        html.H3(f"${option_price:.4f}", className="text-success")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Time Value", className="card-title"),
                        html.H3(f"${time_value:.4f}", className="text-info")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Moneyness (S/K)", className="card-title"),
                        html.H3(f"{moneyness:.4f}", className="text-primary")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Intrinsic Value", className="card-title"),
                        html.H3(f"${intrinsic:.4f}", className="text-secondary")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Greeks display
        html.H5("📊 Greeks Analysis", className="text-info mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Delta", className="card-title"),
                        html.H4(f"{greeks['delta']:.4f}", className="text-primary")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Gamma", className="card-title"),
                        html.H4(f"{greeks['gamma']:.4f}", className="text-success")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Theta", className="card-title"),
                        html.H4(f"{greeks['theta']:.4f}", className="text-danger")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Vega", className="card-title"),
                        html.H4(f"{greeks['vega']:.4f}", className="text-warning")
                    ])
                ])
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Rho", className="card-title"),
                        html.H4(f"{greeks['rho']:.4f}", className="text-info")
                    ])
                ])
            ], width=2)
        ], className="mb-4")
    ]
    
    # Store calculation results for export
    calc_data.update({
        'option_price': option_price,
        'time_value': time_value,
        'intrinsic_value': intrinsic,
        'moneyness': moneyness,
        'delta': greeks['delta'],
        'gamma': greeks['gamma'],
        'theta': greeks['theta'],
        'vega': greeks['vega'],
        'rho': greeks['rho']
    })
    
    # Create visualizations
    visualizations = create_sensitivity_analysis(model_name, S, K, T, r, q, sigma, option_type,
                                                 exercise_style, params)
    
    return pricing_results, visualizations, calc_data

def calculate_model_comparison(S, K, T, r, q, sigma, option_type, exercise_style, params, calc_data):
    """Compare results across the five pricing models, each with the user's
    settings. The binomial tree follows the chosen exercise style; the other
    four price the European option."""

    try:
        # Calculate prices using different models
        binomial_label = 'Binomial Tree (American)' if exercise_style == 'american' else 'Binomial Tree'
        models_data = {}
        for label, name in [('Black-Scholes', 'black_scholes'), (binomial_label, 'binomial'),
                            ('Monte Carlo', 'monte_carlo'), ('Heston', 'heston'),
                            ('Jump-Diffusion', 'jump_diffusion')]:
            model = build_model(name, S, K, T, r, q, sigma, option_type, exercise_style, params)
            models_data[label] = model.calculate_price()

        # Create comparison table
        comparison_df = pd.DataFrame([
            {
                'Model': model_name,
                'Price': f"${price:.4f}",
                'Difference from BS': f"${price - models_data['Black-Scholes']:.4f}",
                'Difference %': f"{((price - models_data['Black-Scholes']) / models_data['Black-Scholes'] * 100):.2f}%"
            }
            for model_name, price in models_data.items()
        ])
        
        # Create price comparison chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(models_data.keys()), 
                y=list(models_data.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
        ])
        
        fig.update_layout(
            title="Option Price Comparison Across Models",
            xaxis_title="Pricing Model",
            yaxis_title="Option Price ($)",
            template='plotly_white',
            height=400
        )
        
        # Statistics
        prices = list(models_data.values())
        
        pricing_results = [
            html.H4("🔄 Model Comparison Results", className="text-primary mb-3"),
            dbc.Alert(
                "Exercise Style is American: the Binomial Tree prices the American option, while "
                "Black-Scholes, Monte Carlo, Heston and Jump-Diffusion price the European one. "
                "Part of the Binomial Tree's difference is the value of early exercise.",
                color="secondary"
            ) if exercise_style == 'american' else html.Div(),

            # Comparison table
            html.Div([
                dash_table.DataTable(
                    data=comparison_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in comparison_df.columns],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': '#ffffff'}
                )
            ], style={'marginBottom': '1.5rem'}),
            
            # Statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Average Price", className="card-title"),
                            html.H3(f"${np.mean(prices):.4f}", className="text-primary")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Price Range", className="card-title"),
                            html.H3(f"${np.max(prices) - np.min(prices):.4f}", className="text-success")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Standard Deviation", className="card-title"),
                            html.H3(f"${np.std(prices):.4f}", className="text-info")
                        ])
                    ])
                ], width=4)
            ], className="mb-4")
        ]
        
        # Store comparison results for export
        calc_data.update({
            'model_prices': models_data,
            'average_price': np.mean(prices),
            'price_range': np.max(prices) - np.min(prices),
            'price_std': np.std(prices)
        })
        
        visualizations = [
            dcc.Graph(figure=fig)
        ]
        
        return pricing_results, visualizations, calc_data
        
    except Exception as e:
        return dbc.Alert(f"Error in model comparison: {str(e)}", color="danger"), "", calc_data

def create_sensitivity_analysis(model_name, S, K, T, r, q, sigma, option_type, exercise_style, params):
    """Create sensitivity analysis charts, using the same model settings as the price"""

    # Price vs Underlying
    spot_range = np.linspace(0.7 * S, 1.3 * S, 50)
    prices_spot = []

    for spot in spot_range:
        temp_model = build_model(model_name, spot, K, T, r, q, sigma, option_type, exercise_style, params)
        prices_spot.append(temp_model.calculate_price())
    
    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(x=spot_range, y=prices_spot, mode='lines', name='Option Price'))
    fig_spot.add_vline(x=S, line_dash="dash", annotation_text="Current Price")
    fig_spot.add_vline(x=K, line_dash="dash", annotation_text="Strike", line_color="red")
    
    fig_spot.update_layout(
        title="Option Price vs Underlying Price",
        xaxis_title="Underlying Price ($)",
        yaxis_title="Option Price ($)",
        template='plotly_white'
    )
    
    # Price vs Volatility
    vol_range = np.linspace(0.1, 0.8, 50)
    prices_vol = []
    
    for vol in vol_range:
        temp_model = build_model(model_name, S, K, T, r, q, vol, option_type, exercise_style, params)
        prices_vol.append(temp_model.calculate_price())
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=vol_range * 100, y=prices_vol, mode='lines', name='Option Price'))
    fig_vol.add_vline(x=sigma * 100, line_dash="dash", annotation_text="Current Vol")
    
    fig_vol.update_layout(
        title="Option Price vs Volatility",
        xaxis_title="Volatility (%)",
        yaxis_title="Option Price ($)",
        template='plotly_white'
    )
    
    return [
        html.H4("📈 Sensitivity Analysis", className="text-info mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_spot)
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=fig_vol)
            ], width=6)
        ])
    ]

# Export functionality callback
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-button", "n_clicks"),
    State("calculation-results", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, calc_data):
    if n_clicks and calc_data:
        # Convert to DataFrame for export
        export_df = pd.DataFrame([calc_data])
        
        # Generate filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"options_pricing_results_{timestamp}.csv"
        
        return dcc.send_data_frame(export_df.to_csv, filename, index=False)
    
    return dash.no_update

if __name__ == '__main__':
    port = int(os.getenv('PORT_OPTIONS', 8051))
    app.run(debug=False, host='0.0.0.0', port=port)