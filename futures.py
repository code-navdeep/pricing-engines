"""Futures pricing calculator (Dash app, port 8050): cost of carry,
Gibson-Schwartz, Merton jump-diffusion, three-factor and term-structure
models, a model comparison and a portfolio simulation."""
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import base64
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app. The Bootstrap "Flatly" theme and its Lato font are
# bundled in assets/ (Dash loads every .css there automatically), so the page
# fetches nothing from outside.
app = dash.Dash(__name__)
app.title = "Futures Pricing Calculator"
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

def validate_inputs(spot_price, risk_free_rate, time_to_maturity, storage_cost=0.0, convenience_yield=0.0):
    """Validate model inputs for mathematical constraints"""
    if spot_price <= 0:
        raise ValueError("Spot price must be positive")
    if time_to_maturity <= 0:
        raise ValueError("Time to maturity must be positive")
    if risk_free_rate < -0.1:  # Allow some negative rates but not extreme values
        raise ValueError("Risk-free rate appears unrealistic")
    return True

class CostOfCarryModel:
    """Cost of Carry Model - F = S × e^((r + s - c) × T)"""
    
    def __init__(self, spot_price, risk_free_rate, storage_cost=0.0, convenience_yield=0.0):
        validate_inputs(spot_price, risk_free_rate, 1.0, storage_cost, convenience_yield)
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.storage_cost = storage_cost
        self.convenience_yield = convenience_yield
    
    def futures_price(self, time_to_maturity):
        """Calculate futures price using cost of carry formula"""
        carry_rate = self.risk_free_rate + self.storage_cost - self.convenience_yield
        return self.spot_price * np.exp(carry_rate * time_to_maturity)
    
    def sensitivity_analysis(self, time_to_maturity, param_range=0.02):
        """Perform sensitivity analysis on model parameters"""
        base_price = self.futures_price(time_to_maturity)
        
        # Interest rate sensitivity
        rate_up = CostOfCarryModel(self.spot_price, self.risk_free_rate + param_range, 
                                  self.storage_cost, self.convenience_yield)
        rate_down = CostOfCarryModel(self.spot_price, self.risk_free_rate - param_range, 
                                    self.storage_cost, self.convenience_yield)
        
        rate_sensitivity = (rate_up.futures_price(time_to_maturity) - 
                           rate_down.futures_price(time_to_maturity)) / (2 * param_range)
        
        # Convenience yield sensitivity
        cy_up = CostOfCarryModel(self.spot_price, self.risk_free_rate, 
                                self.storage_cost, self.convenience_yield + param_range)
        cy_down = CostOfCarryModel(self.spot_price, self.risk_free_rate, 
                                  self.storage_cost, self.convenience_yield - param_range)
        
        cy_sensitivity = (cy_up.futures_price(time_to_maturity) - 
                         cy_down.futures_price(time_to_maturity)) / (2 * param_range)
        
        return {
            'base_price': base_price,
            'rate_sensitivity': rate_sensitivity,
            'convenience_yield_sensitivity': cy_sensitivity
        }

class GibsonSchwartzModel:
    """Gibson-Schwartz stochastic convenience yield model"""
    
    def __init__(self, spot_price, risk_free_rate, kappa, alpha, sigma1, sigma2, rho):
        validate_inputs(spot_price, risk_free_rate, 1.0)  # Use 1.0 as default maturity for validation
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.kappa = kappa  # Mean reversion speed
        self.alpha = alpha  # Long-term mean of convenience yield
        self.sigma1 = sigma1  # Spot price volatility
        self.sigma2 = sigma2  # Convenience yield volatility
        self.rho = rho  # Correlation
    
    def simulate_paths(self, initial_convenience_yield, time_to_maturity, n_steps, n_paths, random_seed=None):
        """Simulate price paths using Monte Carlo"""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        dt = time_to_maturity / n_steps
        
        # Initialize arrays
        spot_paths = np.zeros((n_paths, n_steps + 1))
        cy_paths = np.zeros((n_paths, n_steps + 1))
        
        spot_paths[:, 0] = self.spot_price
        cy_paths[:, 0] = initial_convenience_yield
        
        # Generate correlated random numbers
        for i in range(n_steps):
            z1 = np.random.normal(0, 1, n_paths)
            z2 = np.random.normal(0, 1, n_paths)
            
            # Correlated random numbers
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            
            # Update convenience yield (Ornstein-Uhlenbeck process)
            cy_paths[:, i + 1] = (cy_paths[:, i] + 
                                  self.kappa * (self.alpha - cy_paths[:, i]) * dt + 
                                  self.sigma2 * np.sqrt(dt) * w2)
            
            # Update spot price
            drift = (self.risk_free_rate - cy_paths[:, i]) * dt
            diffusion = self.sigma1 * np.sqrt(dt) * w1
            spot_paths[:, i + 1] = spot_paths[:, i] * np.exp(drift - 0.5 * self.sigma1**2 * dt + diffusion)
        
        return spot_paths, cy_paths
    
    def monte_carlo_futures_price(self, initial_convenience_yield, time_to_maturity, 
                                  n_steps=100, n_paths=1000):
        """Calculate futures price using Monte Carlo simulation"""
        spot_paths, _ = self.simulate_paths(initial_convenience_yield, time_to_maturity, 
                                           n_steps, n_paths)
        
        terminal_prices = spot_paths[:, -1]
        futures_price = np.mean(terminal_prices)
        
        return futures_price, np.std(terminal_prices) / np.sqrt(n_paths)

class MertonJumpDiffusionModel:
    """Merton Jump-Diffusion model"""
    
    def __init__(self, spot_price, risk_free_rate, convenience_yield, sigma, 
                 jump_intensity, jump_mean, jump_std):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.convenience_yield = convenience_yield
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
    
    def analytical_futures_price(self, time_to_maturity):
        """Futures price in closed form.

        Under the pricing measure the jumps are compensated (as in
        simulate_paths), so they change the distribution of the terminal price
        but not its mean - and the futures price is that mean: S·e^((r−δ)T).
        """
        return self.spot_price * np.exp(
            (self.risk_free_rate - self.convenience_yield) * time_to_maturity
        )
    
    def simulate_paths(self, time_to_maturity, n_steps, n_paths):
        """Simulate price paths with jumps"""
        dt = time_to_maturity / n_steps
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.spot_price
        
        jump_compensation = self.jump_intensity * (np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1)
        drift = self.risk_free_rate - self.convenience_yield - jump_compensation
        
        for i in range(n_steps):
            # Continuous component
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            continuous_return = (drift - 0.5 * self.sigma**2) * dt + self.sigma * dW
            
            # Jump component
            n_jumps = np.random.poisson(self.jump_intensity * dt, n_paths)
            jump_return = np.zeros(n_paths)
            
            for j in range(n_paths):
                if n_jumps[j] > 0:
                    jumps = np.random.normal(self.jump_mean, self.jump_std, n_jumps[j])
                    jump_return[j] = np.sum(jumps)
            
            # Update prices
            total_return = continuous_return + jump_return
            paths[:, i + 1] = paths[:, i] * np.exp(total_return)
        
        return paths

class MultiFactorModel:
    """Three-factor model: spot price, convenience yield, interest rate"""
    
    def __init__(self, spot_price, initial_convenience_yield, initial_interest_rate,
                 sigma_s, kappa_delta, alpha_delta, sigma_delta,
                 kappa_r, alpha_r, sigma_r, correlation_matrix):
        self.spot_price = spot_price
        self.initial_convenience_yield = initial_convenience_yield
        self.initial_interest_rate = initial_interest_rate
        self.sigma_s = sigma_s
        self.kappa_delta = kappa_delta
        self.alpha_delta = alpha_delta
        self.sigma_delta = sigma_delta
        self.kappa_r = kappa_r
        self.alpha_r = alpha_r
        self.sigma_r = sigma_r
        self.correlation_matrix = correlation_matrix
    
    def simulate_paths(self, time_to_maturity, n_steps, n_paths):
        """Simulate paths for all three factors"""
        dt = time_to_maturity / n_steps
        
        # Initialize paths
        spot_paths = np.zeros((n_paths, n_steps + 1))
        delta_paths = np.zeros((n_paths, n_steps + 1))
        rate_paths = np.zeros((n_paths, n_steps + 1))
        
        spot_paths[:, 0] = self.spot_price
        delta_paths[:, 0] = self.initial_convenience_yield
        rate_paths[:, 0] = self.initial_interest_rate
        
        # Cholesky decomposition for correlated random numbers
        try:
            L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if matrix not positive definite
            eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        for i in range(n_steps):
            # Generate correlated random numbers
            z = np.random.normal(0, 1, (3, n_paths))
            w = L @ z
            
            # Update convenience yield (mean-reverting)
            delta_paths[:, i + 1] = (delta_paths[:, i] + 
                                    self.kappa_delta * (self.alpha_delta - delta_paths[:, i]) * dt +
                                    self.sigma_delta * np.sqrt(dt) * w[1, :])
            
            # Update interest rate (mean-reverting)
            rate_paths[:, i + 1] = (rate_paths[:, i] + 
                                   self.kappa_r * (self.alpha_r - rate_paths[:, i]) * dt +
                                   self.sigma_r * np.sqrt(dt) * w[2, :])
            
            # Update spot price
            drift = (rate_paths[:, i] - delta_paths[:, i]) * dt
            diffusion = self.sigma_s * np.sqrt(dt) * w[0, :]
            spot_paths[:, i + 1] = spot_paths[:, i] * np.exp(drift - 0.5 * self.sigma_s**2 * dt + diffusion)
        
        return spot_paths, delta_paths, rate_paths
    
    def monte_carlo_futures_price(self, time_to_maturity, n_steps=100, n_paths=1000):
        """Calculate futures price using Monte Carlo simulation"""
        spot_paths, _, _ = self.simulate_paths(time_to_maturity, n_steps, n_paths)
        terminal_prices = spot_paths[:, -1]
        
        futures_price = np.mean(terminal_prices)
        standard_error = np.std(terminal_prices) / np.sqrt(n_paths)
        
        return futures_price, standard_error

class TermStructureModel:
    """Simplified term structure model using PCA factors"""
    
    def __init__(self, spot_price, risk_free_rate, convenience_yield):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.convenience_yield = convenience_yield
        
        # PCA factor loadings (level, slope, curvature)
        self.factor_loadings = {
            'level': lambda T: np.ones_like(T),
            'slope': lambda T: T,
            'curvature': lambda T: T * (1 - T)
        }
        
        # Factor volatilities
        self.factor_volatilities = [0.15, 0.05, 0.03]  # Level, slope, curvature
        
        # Factor mean reversion speeds
        self.factor_kappa = [0.5, 1.0, 2.0]
    
    def simulate_term_structure(self, maturities, time_horizon, n_steps, n_paths):
        """Simulate entire futures curve evolution"""
        dt = time_horizon / n_steps
        
        # Initialize factor processes
        n_factors = len(self.factor_volatilities)
        factor_paths = np.zeros((n_factors, n_paths, n_steps + 1))
        
        # Simulate factor evolution
        for i in range(n_steps):
            for f in range(n_factors):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                factor_paths[f, :, i + 1] = (factor_paths[f, :, i] + 
                                           -self.factor_kappa[f] * factor_paths[f, :, i] * dt +
                                           self.factor_volatilities[f] * dW)
        
        # Convert factor paths to futures curves
        futures_curves = np.zeros((len(maturities), n_paths, n_steps + 1))
        
        for t in range(n_steps + 1):
            for i, T in enumerate(maturities):
                base_price = self.spot_price * np.exp((self.risk_free_rate - self.convenience_yield) * T)
                
                # Apply factor loadings
                factor_adjustment = 0
                for f, (name, loading_func) in enumerate(self.factor_loadings.items()):
                    factor_adjustment += loading_func(np.array([T]))[0] * factor_paths[f, :, t]
                
                futures_curves[i, :, t] = base_price * np.exp(factor_adjustment)
        
        return futures_curves
    
    def calculate_futures_price(self, time_to_maturity):
        """Calculate futures price for a single maturity"""
        base_price = self.spot_price * np.exp((self.risk_free_rate - self.convenience_yield) * time_to_maturity)
        return base_price

class PortfolioSimulator:
    """Portfolio Monte Carlo: one correlated geometric Brownian motion per asset,
    with drift and volatility read from that asset's model inputs. The named
    pricing models are not simulated individually."""
    
    def __init__(self, assets_config, correlation_matrix=None):
        """
        assets_config: dict with asset info
        {
            'S&P 500': {'weight': 0.6, 'model': 'cost_of_carry', 'params': {...}},
            'Crude Oil': {'weight': 0.3, 'model': 'gibson_schwartz', 'params': {...}},
            'Gold': {'weight': 0.1, 'model': 'jump_diffusion', 'params': {...}}
        }
        """
        self.assets_config = assets_config
        self.asset_names = list(assets_config.keys())
        self.weights = np.array([config['weight'] for config in assets_config.values()])
        
        # Default correlation matrix if not provided
        n_assets = len(self.asset_names)
        if correlation_matrix is None:
            self.correlation_matrix = np.eye(n_assets) + 0.1 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
        else:
            self.correlation_matrix = correlation_matrix
        
        # Initialize models for each asset
        self.models = {}
        for asset_name, config in assets_config.items():
            self.models[asset_name] = self._create_model(config['model'], config['params'])
    
    def _create_model(self, model_type, params):
        """Create appropriate pricing model based on type"""
        if model_type == 'cost_of_carry':
            return CostOfCarryModel(**params)
        elif model_type == 'gibson_schwartz':
            return GibsonSchwartzModel(**params)
        elif model_type == 'jump_diffusion':
            return MertonJumpDiffusionModel(**params)
        elif model_type == 'multi_factor':
            return MultiFactorModel(**params)
        else:
            # Default to cost of carry
            return CostOfCarryModel(**params)
    
    def simulate_portfolio(self, time_horizon, n_steps, n_paths):
        """Simulate portfolio evolution using correlated asset returns"""
        dt = time_horizon / n_steps
        
        # Initialize portfolio value paths
        portfolio_values = np.zeros((n_paths, n_steps + 1))
        portfolio_values[:, 0] = 100.0  # Start with $100 portfolio
        
        # Initialize individual asset paths
        asset_paths = {}
        for asset_name in self.asset_names:
            asset_paths[asset_name] = np.zeros((n_paths, n_steps + 1))
            asset_paths[asset_name][:, 0] = self.assets_config[asset_name]['params']['spot_price']
        
        # Cholesky decomposition for correlated returns
        try:
            L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if matrix not positive definite
            eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Simulate each time step
        for t in range(n_steps):
            # Generate correlated random shocks
            z = np.random.normal(0, 1, (len(self.asset_names), n_paths))
            correlated_shocks = L @ z
            
            # Update each asset using its specific model
            for i, asset_name in enumerate(self.asset_names):
                model = self.models[asset_name]
                current_price = asset_paths[asset_name][:, t]
                
                # Get model-specific parameters
                params = self.assets_config[asset_name]['params']
                
                # Geometric Brownian motion: drift from the model's rate and
                # yield inputs, volatility from its sigma (20% when it has none)
                if hasattr(model, 'risk_free_rate'):
                    drift = model.risk_free_rate - getattr(model, 'convenience_yield', 0)
                    volatility = getattr(model, 'sigma', 0.2)
                else:
                    drift = 0.05  # Default
                    volatility = 0.2  # Default
                
                # Apply correlated shock
                returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * correlated_shocks[i, :]
                asset_paths[asset_name][:, t + 1] = current_price * np.exp(returns)
            
            # Calculate portfolio value
            for path in range(n_paths):
                portfolio_value = 0
                
                for i, asset_name in enumerate(self.asset_names):
                    # Portfolio weight * asset performance
                    asset_performance = asset_paths[asset_name][path, t + 1] / asset_paths[asset_name][path, 0]
                    portfolio_value += self.weights[i] * asset_performance
                
                portfolio_values[path, t + 1] = portfolio_values[path, 0] * portfolio_value
        
        return portfolio_values, asset_paths
    
    def calculate_risk_metrics(self, portfolio_values, dt, risk_free_rate=0.05):
        """Calculate risk metrics. dt is the length of one simulation step in
        years; risk_free_rate is the rate every asset in the simulation uses."""
        returns = np.diff(np.log(portfolio_values), axis=1)   # log return per step

        # Portfolio statistics
        final_values = portfolio_values[:, -1]
        total_return = (final_values / portfolio_values[:, 0] - 1) * 100
        horizon = dt * returns.shape[1]

        # Annualised volatility: the per-step spread scaled by 1/sqrt(dt)
        volatility = np.std(returns) / np.sqrt(dt) * 100
        # Sharpe ratio: annualised mean log return in excess of the risk-free
        # rate, divided by annualised volatility
        annual_return = np.mean(np.log(final_values / portfolio_values[:, 0])) / horizon
        sharpe_ratio = (annual_return - risk_free_rate) / (volatility / 100) if volatility > 0 else 0
        
        # Drawdown calculation
        cummax = np.maximum.accumulate(portfolio_values, axis=1)
        drawdowns = (portfolio_values - cummax) / cummax * 100
        max_drawdown = np.min(drawdowns)
        
        # VaR calculations
        var_95 = np.percentile(total_return, 5)
        var_99 = np.percentile(total_return, 1)
        cvar_95 = np.mean(total_return[total_return <= var_95])
        
        return {
            'total_return_mean': np.mean(total_return),
            'total_return_std': np.std(total_return),
            'volatility_annualized': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'final_values': final_values
        }

# =====================================================
# DASH APP LAYOUT
# =====================================================

def serve_layout():
    """The page, built on each load (after every component builder below is
    defined). Attached as app.layout at the end of this file."""
    return dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id='uploaded-data', data=None),
        dcc.Store(id='use-uploaded-params', data=False),
        dcc.Store(id='portfolio-config', data={}),
        # Rows of the figures the current model last computed — what
        # "Download Results" exports
        dcc.Store(id='futures-results', data=None),
        
        # Header
        dbc.Row([
            dbc.Col([
                html.H1(
                    "📈 Futures Pricing Calculator",
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
                        html.H5("⚙️ Model Configuration & Parameters", className="mb-0")
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
                    # Model Selection Row
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Pricing Model:", className="fw-bold"),
                            dcc.Dropdown(
                                id='model-selector',
                                options=[
                                    {'label': 'Cost of Carry', 'value': 'cost_of_carry'},
                                    {'label': 'Gibson-Schwartz', 'value': 'gibson_schwartz'},
                                    {'label': 'Merton Jump-Diffusion', 'value': 'jump_diffusion'},
                                    {'label': 'Multi-Factor', 'value': 'multi_factor'},
                                    {'label': 'Term Structure', 'value': 'term_structure'},
                                    {'label': 'Model Comparison', 'value': 'model_comparison'},
                                    {'label': 'Portfolio Simulation', 'value': 'portfolio_simulation'},
                                ],
                                value='cost_of_carry',
                                clearable=False,
                                className="mb-3"
                            )
                        ], width=12)
                    ]),
                    
                    # Universal Parameters Row
                    dbc.Row([
                        dbc.Col([
                            html.H6("📊 Universal Parameters", className="text-primary mb-2"),
                            html.Label("Spot Price ($)"),
                            dbc.Input(id='spot-price', type='number', value=100.0, min=0.01, step=0.01, size="sm")
                        ], width=3),
                        dbc.Col([
                            html.Label("Risk-Free Rate (%)", className="mt-4"),
                            dcc.Slider(
                                id='risk-free-rate', 
                                min=0.0, max=10.0, step=0.1, value=5.0,
                                marks={i: f'{i}%' for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Time to Maturity (Years)", className="mt-4"),
                            dcc.Slider(
                                id='time-to-maturity', 
                                min=0.1, max=5.0, step=0.1, value=1.0,
                                marks={0.5: '0.5Y', 1.0: '1Y', 2.0: '2Y', 3.0: '3Y', 5.0: '5Y'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=3),
                        dbc.Col([
                            html.H6("🏭 Asset Parameters", className="text-success mb-2"),
                            html.Label("Storage Cost (% per year)", className="small"),
                            dcc.Slider(
                                id='storage-cost', 
                                min=0.0, max=10.0, step=0.1, value=2.0,
                                marks={0: '0%', 5: '5%', 10: '10%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Label("Convenience Yield (% per year)", className="small mt-2"),
                            dcc.Slider(
                                id='convenience-yield', 
                                min=0.0, max=10.0, step=0.1, value=3.0,
                                marks={0: '0%', 5: '5%', 10: '10%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=3)
                    ], className="mb-3"),
                    
                    html.Hr(),
                    
                    # Data & Export Row
                    dbc.Row([
                        dbc.Col([
                            html.H6("📁 Data Management", className="text-info mb-2"),
                            dcc.Upload(
                                id='upload-data',
                                children=dbc.Button("Upload CSV Data",
                                                    color="outline-info", size="sm"),
                                multiple=False
                            ),
                            html.Div(id='upload-status', className="mt-1 small"),
                        ], width=3),
                        dbc.Col([
                            html.H6("⚙️ Actions", className="text-warning mb-2"),
                            dbc.ButtonGroup([
                                dbc.Button("Use Uploaded", id='use-uploaded-btn', color="primary", size="sm", disabled=True),
                                dbc.Button("Reset Default", id='reset-params-btn', color="secondary", size="sm"),
                            ]),
                        ], width=3),
                        dbc.Col([
                            html.H6("💾 Export", className="text-success mb-2"),
                            dbc.Button("Download Results", id='download-btn',
                                       color="success", size="sm"),
                            dcc.Download(id="download-results"),
                        ], width=3),
                        dbc.Col([
                            html.H6("📊 Quick Stats", className="text-muted mb-2"),
                            html.Small(id="quick-stats", className="text-muted")
                        ], width=3)
                    ])
                ])
            ], id="controls-collapse", is_open=True)
        ], className="mb-4", style={"position": "sticky", "top": "10px", "zIndex": "1000"}),
        
        # Dynamic Model-Specific Parameters (Collapsible)
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.H6("🔧 Advanced Model Parameters", className="mb-0")
                    ], width=10),
                    dbc.Col([
                        dbc.Button(
                            "Toggle Advanced",
                            id="advanced-collapse-button",
                            className="btn-outline-secondary btn-sm",
                            n_clicks=0,
                        )
                    ], width=2, className="text-end")
                ])
            ]),
            dbc.Collapse([
                # Every model's sliders stay in the page (only the selected
                # model's group is shown), so their current values can always
                # be read by the main calculation.
                html.Div([
                    html.Div(create_gibson_schwartz_parameters(), id='adv-gibson_schwartz', style={'display': 'none'}),
                    html.Div(create_jump_diffusion_parameters(), id='adv-jump_diffusion', style={'display': 'none'}),
                    html.Div(create_multi_factor_parameters(), id='adv-multi_factor', style={'display': 'none'}),
                    html.Div(create_term_structure_parameters(), id='adv-term_structure', style={'display': 'none'}),
                ], id='advanced-parameters', className="p-3")
            ], id="advanced-collapse", is_open=False)
        ], className="mb-4", id="advanced-card", style={'display': 'none'}),
        
        # Market Analysis Section (Always available)
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.H6("📊 Market Analysis", className="mb-0")
                    ], width=10),
                    dbc.Col([
                        dbc.Button(
                            "Toggle Analysis",
                            id="analysis-collapse-button",
                            className="btn-outline-info btn-sm",
                            n_clicks=0,
                        )
                    ], width=2, className="text-end")
                ])
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Current Market Price ($):", className="fw-bold"),
                            dbc.Input(id='market-price', type='number', value=102.0, min=0.01, step=0.01, size="sm")
                        ], width=3),
                        dbc.Col([
                            html.Label("Market Analysis:", className="fw-bold"),
                            html.Div(id='market-analysis-output', className="small")
                        ], width=9),
                    ])
                ])
            ], id="analysis-collapse", is_open=False)
        ], className="mb-4"),

        # Main Content Area (Full Width)
        dbc.Row([
            dbc.Col([
                html.Div(id='model-output', className="p-3")
            ], width=12)
        ]),
        
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
    ],
    style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'paddingBottom': '20px'}
)

# =====================================================
# CALLBACKS
# =====================================================

# Callback for controls collapse toggle
@app.callback(
    Output("controls-collapse", "is_open"),
    Input("controls-collapse-button", "n_clicks"),
    State("controls-collapse", "is_open"),
)
def toggle_controls_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for quick stats
@app.callback(
    Output('quick-stats', 'children'),
    [Input('spot-price', 'value'),
     Input('risk-free-rate', 'value'),
     Input('time-to-maturity', 'value'),
     Input('model-selector', 'value')]
)
def update_quick_stats(spot_price, risk_free_rate, time_to_maturity, model_name):
    if spot_price is None or risk_free_rate is None or time_to_maturity is None:
        return "Enter parameters to see quick stats"
    
    try:
        # Calculate basic future value for quick reference
        basic_future = spot_price * (1 + risk_free_rate/100 * time_to_maturity)
        model_display = model_name.replace('_', ' ').title()
        
        return f"Model: {model_display} | Basic FV: ${basic_future:.2f} | T={time_to_maturity}Y"
    except:
        return "Invalid parameters"

# Callback for advanced parameters collapse toggle
@app.callback(
    Output("advanced-collapse", "is_open"),
    Input("advanced-collapse-button", "n_clicks"),
    State("advanced-collapse", "is_open"),
)
def toggle_advanced_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for analysis section collapse toggle
@app.callback(
    Output("analysis-collapse", "is_open"),
    Input("analysis-collapse-button", "n_clicks"),
    State("analysis-collapse", "is_open"),
)
def toggle_analysis_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Models with advanced sliders (each has a group 'adv-<model>' in the page)
ADVANCED_MODELS = ['gibson_schwartz', 'jump_diffusion', 'multi_factor', 'term_structure']

# Callback showing the selected model's advanced sliders
@app.callback(
    [Output(f'adv-{m}', 'style') for m in ADVANCED_MODELS] +
    [Output('advanced-card', 'style')],
    Input('model-selector', 'value')
)
def update_advanced_parameters(model_name):
    group_styles = [{'display': 'block'} if m == model_name else {'display': 'none'}
                    for m in ADVANCED_MODELS]
    card_style = ({'display': 'block', 'marginBottom': '1rem'}
                  if model_name in ADVANCED_MODELS else {'display': 'none'})
    return group_styles + [card_style]

def create_gibson_schwartz_parameters():
    return dbc.Row([
        dbc.Col([
            html.H6("🌊 Gibson-Schwartz Parameters", className="text-primary mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Mean Reversion Speed (κ)"),
                    dcc.Slider(id='gs-kappa', min=0.1, max=5.0, step=0.1, value=2.0,
                              marks={0.1: '0.1', 2.0: '2.0', 5.0: '5.0'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6),
                dbc.Col([
                    html.Label("Spot Price Volatility (σ₁)"),
                    dcc.Slider(id='gs-sigma1', min=0.1, max=0.5, step=0.01, value=0.25,
                              marks={0.1: '0.1', 0.25: '0.25', 0.5: '0.5'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Convenience Yield Volatility (σ₂)"),
                    dcc.Slider(id='gs-sigma2', min=0.05, max=0.3, step=0.01, value=0.15,
                              marks={0.05: '0.05', 0.15: '0.15', 0.3: '0.3'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6),
                dbc.Col([
                    html.Label("Correlation (ρ)"),
                    dcc.Slider(id='gs-rho', min=-0.9, max=0.9, step=0.1, value=-0.3,
                              marks={-0.9: '-0.9', 0: '0', 0.9: '0.9'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Number of Simulation Paths"),
                    dcc.Dropdown(
                        id='gs-n-paths',
                        options=[
                            {'label': '100', 'value': 100},
                            {'label': '500', 'value': 500},
                            {'label': '1,000', 'value': 1000},
                            {'label': '2,000', 'value': 2000}
                        ],
                        value=1000,
                        clearable=False
                    )
                ], width=6)
            ], className="mt-2")
        ], width=12)
    ])

def create_jump_diffusion_parameters():
    return dbc.Row([
        dbc.Col([
            html.H6("⚡ Jump-Diffusion Parameters", className="text-danger mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Continuous Volatility (σ)"),
                    dcc.Slider(id='jd-sigma', min=0.1, max=0.5, step=0.01, value=0.25,
                              marks={0.1: '0.1', 0.25: '0.25', 0.5: '0.5'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6),
                dbc.Col([
                    html.Label("Jump Intensity (λ)"),
                    dcc.Slider(id='jd-intensity', min=0.1, max=3.0, step=0.1, value=0.5,
                              marks={0.1: '0.1', 1.5: '1.5', 3.0: '3.0'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Jump Mean (μⱼ)"),
                    dcc.Slider(id='jd-mean', min=-0.1, max=0.1, step=0.01, value=-0.02,
                              marks={-0.1: '-0.1', 0: '0', 0.1: '0.1'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6),
                dbc.Col([
                    html.Label("Jump Volatility (σⱼ)"),
                    dcc.Slider(id='jd-std', min=0.05, max=0.3, step=0.01, value=0.10,
                              marks={0.05: '0.05', 0.15: '0.15', 0.3: '0.3'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Number of Simulation Paths"),
                    dcc.Dropdown(
                        id='jd-n-paths',
                        options=[
                            {'label': '500', 'value': 500},
                            {'label': '1,000', 'value': 1000},
                            {'label': '2,000', 'value': 2000}
                        ],
                        value=1000,
                        clearable=False
                    )
                ], width=6)
            ], className="mt-2")
        ], width=12)
    ])

def create_multi_factor_parameters():
    return dbc.Row([
        dbc.Col([
            html.H6("🔀 Multi-Factor Parameters", className="text-success mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Spot Price Volatility (σₛ)"),
                    dcc.Slider(id='mf-sigma-s', min=0.1, max=0.5, step=0.01, value=0.25,
                              marks={0.1: '0.1', 0.25: '0.25', 0.5: '0.5'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("CY Mean Reversion (κδ)"),
                    dcc.Slider(id='mf-kappa-delta', min=0.1, max=5.0, step=0.1, value=2.0,
                              marks={0.1: '0.1', 2.5: '2.5', 5.0: '5.0'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("Rate Mean Reversion (κᵣ)"),
                    dcc.Slider(id='mf-kappa-r', min=0.1, max=3.0, step=0.1, value=0.5,
                              marks={0.1: '0.1', 1.5: '1.5', 3.0: '3.0'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("CY Volatility (σδ)"),
                    dcc.Slider(id='mf-sigma-delta', min=0.05, max=0.3, step=0.01, value=0.15,
                              marks={0.05: '0.05', 0.15: '0.15', 0.3: '0.3'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("Rate Volatility (σᵣ)"),
                    dcc.Slider(id='mf-sigma-r', min=0.005, max=0.05, step=0.005, value=0.02,
                              marks={0.005: '0.005', 0.025: '0.025', 0.05: '0.05'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("Number of Simulation Paths"),
                    dcc.Dropdown(
                        id='mf-n-paths',
                        options=[
                            {'label': '500', 'value': 500},
                            {'label': '1,000', 'value': 1000},
                            {'label': '2,000', 'value': 2000}
                        ],
                        value=1000,
                        clearable=False
                    )
                ], width=4)
            ], className="mt-2"),
            html.Hr(),
            html.H6("Correlation Matrix", className="text-info mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Label("Spot-CY Correlation"),
                    dcc.Slider(id='mf-rho-s-delta', min=-0.9, max=0.9, step=0.1, value=-0.3,
                              marks={-0.9: '-0.9', 0: '0', 0.9: '0.9'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("Spot-Rate Correlation"),
                    dcc.Slider(id='mf-rho-s-r', min=-0.5, max=0.5, step=0.1, value=0.1,
                              marks={-0.5: '-0.5', 0: '0', 0.5: '0.5'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("CY-Rate Correlation"),
                    dcc.Slider(id='mf-rho-delta-r', min=-0.5, max=0.5, step=0.1, value=-0.2,
                              marks={-0.5: '-0.5', 0: '0', 0.5: '0.5'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4)
            ])
        ], width=12)
    ])

def create_term_structure_parameters():
    return dbc.Row([
        dbc.Col([
            html.H6("📈 Term Structure PCA Parameters", className="text-warning mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Level Factor Volatility"),
                    dcc.Slider(id='ts-level-vol', min=0.05, max=0.3, step=0.01, value=0.15,
                              marks={0.05: '0.05', 0.15: '0.15', 0.3: '0.3'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("Slope Factor Volatility"),
                    dcc.Slider(id='ts-slope-vol', min=0.01, max=0.1, step=0.01, value=0.05,
                              marks={0.01: '0.01', 0.05: '0.05', 0.1: '0.1'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4),
                dbc.Col([
                    html.Label("Curvature Factor Volatility"),
                    dcc.Slider(id='ts-curve-vol', min=0.005, max=0.05, step=0.005, value=0.03,
                              marks={0.005: '0.005', 0.025: '0.025', 0.05: '0.05'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Simulation Horizon (Years)"),
                    dcc.Slider(id='ts-time-horizon', min=0.5, max=3.0, step=0.1, value=1.0,
                              marks={0.5: '0.5', 1.5: '1.5', 3.0: '3.0'},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=6),
                dbc.Col([
                    html.Label("Number of Simulation Paths"),
                    dcc.Dropdown(
                        id='ts-n-paths',
                        options=[
                            {'label': '100', 'value': 100},
                            {'label': '500', 'value': 500},
                            {'label': '1,000', 'value': 1000}
                        ],
                        value=500,
                        clearable=False
                    )
                ], width=6)
            ], className="mt-2")
        ], width=12)
    ])

# Callback for data upload
@app.callback(
    [Output('uploaded-data', 'data'),
     Output('upload-status', 'children'),
     Output('use-uploaded-btn', 'disabled')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_uploaded_data(contents, filename):
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df.to_dict('records'), html.Div(f"✅ Successfully uploaded {filename}", style={'color': 'green'}), False
        except Exception as e:
            return dash.no_update, html.Div(f"❌ Error: {str(e)}", style={'color': 'red'}), True
    return dash.no_update, html.Div("No file uploaded"), True

# Callback for using uploaded parameters
@app.callback(
    [Output('spot-price', 'value'),
     Output('risk-free-rate', 'value'),
     Output('storage-cost', 'value'),
     Output('convenience-yield', 'value'),
     Output('use-uploaded-params', 'data')],
    Input('use-uploaded-btn', 'n_clicks'),
    State('uploaded-data', 'data'),
    prevent_initial_call=True
)
def use_uploaded_params(n_clicks, uploaded_data):
    if n_clicks is None or uploaded_data is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, False
    
    df = pd.DataFrame(uploaded_data)
    latest = df.iloc[-1]
    
    # Extract parameters with fallbacks. The file's decimals become the page's
    # percentages, rounded so that 0.018 shows as 1.8 (not 1.7999999999999998)
    spot_price = latest.get('Spot_Price', latest.get('spot_price', 100.0))
    risk_free_rate = round(float(latest.get('Risk_Free_Rate', latest.get('risk_free_rate', 0.05))) * 100, 6)
    storage_cost = round(float(latest.get('Storage_Cost', latest.get('storage_cost', 0.02))) * 100, 6)
    convenience_yield = round(float(latest.get('Convenience_Yield', latest.get('convenience_yield', 0.03))) * 100, 6)
    
    return spot_price, risk_free_rate, storage_cost, convenience_yield, True

# Callback for resetting parameters
@app.callback(
    [Output('spot-price', 'value', allow_duplicate=True),
     Output('risk-free-rate', 'value', allow_duplicate=True),
     Output('storage-cost', 'value', allow_duplicate=True),
     Output('convenience-yield', 'value', allow_duplicate=True),
     Output('use-uploaded-params', 'data', allow_duplicate=True)],
    Input('reset-params-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_params(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, False
    return 100.0, 5.0, 2.0, 3.0, False

# Callback for downloading results: the figures the current model last computed
@app.callback(
    Output("download-results", "data"),
    Input("download-btn", "n_clicks"),
    State('futures-results', 'data'),
    prevent_initial_call=True,
)
def download_results(n_clicks, results_rows):
    if n_clicks is None or not results_rows:
        return dash.no_update

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame(results_rows)

    return dcc.send_data_frame(
        results_df.to_csv, 
        f"futures_results_{timestamp}.csv",
        index=False
    )

# Market Analysis callback
@app.callback(
    Output('market-analysis-output', 'children'),
    [Input('market-price', 'value'),
     Input('spot-price', 'value'),
     Input('risk-free-rate', 'value'),
     Input('time-to-maturity', 'value'),
     Input('storage-cost', 'value'),
     Input('convenience-yield', 'value')]
)
def update_market_analysis(market_price, spot_price, risk_free_rate, time_to_maturity, storage_cost, convenience_yield):
    if market_price is None or spot_price is None or risk_free_rate is None or time_to_maturity is None:
        return "Enter parameters for market analysis"

    try:
        # Calculate theoretical fair value using Cost of Carry model (0% is a valid input)
        model = CostOfCarryModel(spot_price, risk_free_rate/100,
                                 (storage_cost or 0)/100, (convenience_yield or 0)/100)
        theoretical_price = model.futures_price(time_to_maturity)
        
        # Analysis
        arbitrage = market_price - theoretical_price
        arbitrage_pct = arbitrage / spot_price * 100
        
        if abs(arbitrage) > 0.5:
            if arbitrage > 0:
                color = "warning"
                status = "⚠️ Market OVERPRICED"
                suggestion = "💡 Consider: Sell futures, buy underlying"
            else:
                color = "warning" 
                status = "⚠️ Market UNDERPRICED"
                suggestion = "💡 Consider: Buy futures, sell underlying"
        else:
            color = "success"
            status = "✅ Market fairly priced"
            suggestion = "💡 No arbitrage opportunity"
        
        return dbc.Alert([
            html.P(f"Fair Value: ${theoretical_price:.2f}", className="mb-1"),
            html.P(f"Market vs Fair: ${arbitrage:.2f} ({arbitrage_pct:.2f}%)", className="mb-1"),
            html.P(status, className="mb-1 fw-bold"),
            html.P(suggestion, className="mb-0 small")
        ], color=color, className="p-2 mb-0")
        
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger", className="p-2 mb-0")


# Main callback for model output - simplified to only handle basic parameters
@app.callback(
    [Output('futures-results', 'data'),
     Output('model-output', 'children')],
    [Input('model-selector', 'value'),
     Input('spot-price', 'value'),
     Input('risk-free-rate', 'value'),
     Input('time-to-maturity', 'value'),
     Input('storage-cost', 'value'),
     Input('convenience-yield', 'value')],
    # The advanced sliders' current values (always in the page)
    [State('gs-kappa', 'value'), State('gs-sigma1', 'value'), State('gs-sigma2', 'value'),
     State('gs-rho', 'value'), State('gs-n-paths', 'value'),
     State('jd-sigma', 'value'), State('jd-intensity', 'value'), State('jd-mean', 'value'),
     State('jd-std', 'value'), State('jd-n-paths', 'value'),
     State('mf-sigma-s', 'value'), State('mf-kappa-delta', 'value'), State('mf-sigma-delta', 'value'),
     State('mf-kappa-r', 'value'), State('mf-sigma-r', 'value'), State('mf-rho-s-delta', 'value'),
     State('mf-rho-s-r', 'value'), State('mf-rho-delta-r', 'value'), State('mf-n-paths', 'value'),
     State('ts-level-vol', 'value'), State('ts-slope-vol', 'value'), State('ts-curve-vol', 'value'),
     State('ts-time-horizon', 'value'), State('ts-n-paths', 'value')]
)
def update_model_output(model_name, spot_price, risk_free_rate, time_to_maturity, storage_cost,
                        convenience_yield, *advanced):
    gs_params, jd_params = advanced[0:5], advanced[5:10]
    mf_params, ts_params = advanced[10:19], advanced[19:24]
    
    # Validate required inputs
    if spot_price is None or spot_price <= 0:
        return None, dbc.Alert("Please enter a valid spot price", color="warning", className="p-2")
    if time_to_maturity is None or time_to_maturity <= 0:
        return None, dbc.Alert("Please enter a valid time to maturity", color="warning", className="p-2")
    
    # Convert percentages to decimals (0% is a valid input)
    risk_free_rate = risk_free_rate / 100 if risk_free_rate is not None else 0.05
    storage_cost = storage_cost / 100 if storage_cost is not None else 0.02
    convenience_yield = convenience_yield / 100 if convenience_yield is not None else 0.03
    
    if model_name == 'cost_of_carry':
        return render_cost_of_carry(spot_price, risk_free_rate, time_to_maturity, storage_cost, convenience_yield)
    elif model_name == 'gibson_schwartz':
        return render_gibson_schwartz(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                                      *gs_params)
    elif model_name == 'jump_diffusion':
        return render_jump_diffusion(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                                     *jd_params)
    elif model_name == 'multi_factor':
        return render_multi_factor(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                                   *mf_params)
    elif model_name == 'term_structure':
        return render_term_structure(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                                     *ts_params)
    elif model_name == 'model_comparison':
        return render_model_comparison(spot_price, risk_free_rate, time_to_maturity, storage_cost, convenience_yield)
    elif model_name == 'portfolio_simulation':
        # Results arrive when the simulation is run
        return None, render_portfolio_simulation()
    else:
        return None, html.Div("Select a model from the dropdown")

# Additional callbacks for advanced parameter updates
@app.callback(
    [Output('futures-results', 'data', allow_duplicate=True),
     Output('model-output', 'children', allow_duplicate=True)],
    [Input('gs-kappa', 'value'),
     Input('gs-sigma1', 'value'),
     Input('gs-sigma2', 'value'),
     Input('gs-rho', 'value'),
     Input('gs-n-paths', 'value')],
    [State('model-selector', 'value'),
     State('spot-price', 'value'),
     State('risk-free-rate', 'value'),
     State('time-to-maturity', 'value'),
     State('convenience-yield', 'value')],
    prevent_initial_call=True
)
def update_gibson_schwartz_output(gs_kappa, gs_sigma1, gs_sigma2, gs_rho, gs_n_paths, 
                                 model_name, spot_price, risk_free_rate, time_to_maturity, convenience_yield):
    if model_name != 'gibson_schwartz':
        return dash.no_update, dash.no_update
    
    risk_free_rate = risk_free_rate / 100 if risk_free_rate is not None else 0.05
    convenience_yield = convenience_yield / 100 if convenience_yield is not None else 0.03
    
    return render_gibson_schwartz(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                                  gs_kappa, gs_sigma1, gs_sigma2, gs_rho, gs_n_paths)

@app.callback(
    [Output('futures-results', 'data', allow_duplicate=True),
     Output('model-output', 'children', allow_duplicate=True)],
    [Input('jd-sigma', 'value'),
     Input('jd-intensity', 'value'),
     Input('jd-mean', 'value'),
     Input('jd-std', 'value'),
     Input('jd-n-paths', 'value')],
    [State('model-selector', 'value'),
     State('spot-price', 'value'),
     State('risk-free-rate', 'value'),
     State('time-to-maturity', 'value'),
     State('convenience-yield', 'value')],
    prevent_initial_call=True
)
def update_jump_diffusion_output(jd_sigma, jd_intensity, jd_mean, jd_std, jd_n_paths,
                                model_name, spot_price, risk_free_rate, time_to_maturity, convenience_yield):
    if model_name != 'jump_diffusion':
        return dash.no_update, dash.no_update
    
    risk_free_rate = risk_free_rate / 100 if risk_free_rate is not None else 0.05
    convenience_yield = convenience_yield / 100 if convenience_yield is not None else 0.03
    
    return render_jump_diffusion(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                                 jd_sigma, jd_intensity, jd_mean, jd_std, jd_n_paths)

@app.callback(
    [Output('futures-results', 'data', allow_duplicate=True),
     Output('model-output', 'children', allow_duplicate=True)],
    [Input('mf-sigma-s', 'value'),
     Input('mf-kappa-delta', 'value'),
     Input('mf-sigma-delta', 'value'),
     Input('mf-kappa-r', 'value'),
     Input('mf-sigma-r', 'value'),
     Input('mf-rho-s-delta', 'value'),
     Input('mf-rho-s-r', 'value'),
     Input('mf-rho-delta-r', 'value'),
     Input('mf-n-paths', 'value')],
    [State('model-selector', 'value'),
     State('spot-price', 'value'),
     State('risk-free-rate', 'value'),
     State('time-to-maturity', 'value'),
     State('convenience-yield', 'value')],
    prevent_initial_call=True
)
def update_multi_factor_output(mf_sigma_s, mf_kappa_delta, mf_sigma_delta, mf_kappa_r, mf_sigma_r,
                              mf_rho_s_delta, mf_rho_s_r, mf_rho_delta_r, mf_n_paths,
                              model_name, spot_price, risk_free_rate, time_to_maturity, convenience_yield):
    if model_name != 'multi_factor':
        return dash.no_update, dash.no_update
    
    risk_free_rate = risk_free_rate / 100 if risk_free_rate is not None else 0.05
    convenience_yield = convenience_yield / 100 if convenience_yield is not None else 0.03
    
    return render_multi_factor(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                               mf_sigma_s, mf_kappa_delta, mf_sigma_delta, mf_kappa_r, mf_sigma_r,
                               mf_rho_s_delta, mf_rho_s_r, mf_rho_delta_r, mf_n_paths)

@app.callback(
    [Output('futures-results', 'data', allow_duplicate=True),
     Output('model-output', 'children', allow_duplicate=True)],
    [Input('ts-level-vol', 'value'),
     Input('ts-slope-vol', 'value'),
     Input('ts-curve-vol', 'value'),
     Input('ts-time-horizon', 'value'),
     Input('ts-n-paths', 'value')],
    [State('model-selector', 'value'),
     State('spot-price', 'value'),
     State('risk-free-rate', 'value'),
     State('time-to-maturity', 'value'),
     State('convenience-yield', 'value')],
    prevent_initial_call=True
)
def update_term_structure_output(ts_level_vol, ts_slope_vol, ts_curve_vol, ts_time_horizon, ts_n_paths,
                                model_name, spot_price, risk_free_rate, time_to_maturity, convenience_yield):
    if model_name != 'term_structure':
        return dash.no_update, dash.no_update
    
    risk_free_rate = risk_free_rate / 100 if risk_free_rate is not None else 0.05
    convenience_yield = convenience_yield / 100 if convenience_yield is not None else 0.03
    
    return render_term_structure(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                                 ts_level_vol, ts_slope_vol, ts_curve_vol, ts_time_horizon, ts_n_paths)

# =====================================================
# RENDER FUNCTIONS FOR EACH MODEL
# =====================================================

def render_cost_of_carry(spot_price, risk_free_rate, time_to_maturity, storage_cost, convenience_yield):
    try:
        model = CostOfCarryModel(spot_price, risk_free_rate, storage_cost, convenience_yield)
        futures_price = model.futures_price(time_to_maturity)
        sensitivity = model.sensitivity_analysis(time_to_maturity)
        
        # Term structure
        maturities = np.linspace(0.25, 3.0, 12)
        term_structure = [model.futures_price(t) for t in maturities]
        
        fig_term = go.Figure()
        fig_term.add_trace(go.Scatter(
            x=maturities, y=term_structure,
            mode='lines+markers', name='Futures Prices',
            line=dict(color='blue', width=3)
        ))
        fig_term.add_hline(y=spot_price, line_dash="dash", line_color="red", annotation_text="Spot Price")
        fig_term.update_layout(
            title="Futures Term Structure",
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Futures Price ($)",
            template='plotly_white',
            height=400
        )
        
        # Sensitivity heatmap
        rates = np.linspace(risk_free_rate - 0.02, risk_free_rate + 0.02, 10)
        yields = np.linspace(max(0, convenience_yield - 0.02), convenience_yield + 0.02, 10)
        prices_matrix = np.zeros((len(rates), len(yields)))
        
        for i, r in enumerate(rates):
            for j, y in enumerate(yields):
                temp_model = CostOfCarryModel(spot_price, r, storage_cost, y)
                prices_matrix[i, j] = temp_model.futures_price(time_to_maturity)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=prices_matrix,
            x=[f"{y:.1%}" for y in yields],
            y=[f"{r:.1%}" for r in rates],
            colorscale='Viridis',
            hoverongaps=False
        ))
        fig_heatmap.update_layout(
            title=f"Futures Price Sensitivity (Maturity: {time_to_maturity:.1f}Y)",
            xaxis_title="Convenience Yield",
            yaxis_title="Risk-Free Rate",
            height=400
        )
        
        rows = [{
            'Model': 'Cost of Carry', 'Spot_Price': spot_price,
            'Risk_Free_Rate': risk_free_rate, 'Storage_Cost': storage_cost,
            'Convenience_Yield': convenience_yield, 'Time_to_Maturity': time_to_maturity,
            'Futures_Price': futures_price, 'Basis': futures_price - spot_price,
            'Carry_Rate': risk_free_rate + storage_cost - convenience_yield,
        }]
        return rows, html.Div([
            html.H3("💰 Cost of Carry Model"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Futures Price"),
                        dbc.CardBody([
                            html.H4(f"${futures_price:.2f}", className="card-title")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Basis (F-S)"),
                        dbc.CardBody([
                            html.H4(f"${futures_price - spot_price:.2f}", className="card-title")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Carry Rate"),
                        dbc.CardBody([
                            html.H4(f"{(risk_free_rate + storage_cost - convenience_yield):.2%}", className="card-title")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dcc.Graph(figure=fig_term),
            html.H4("📈 Sensitivity Analysis", className="mt-4"),
            dcc.Graph(figure=fig_heatmap)
        ])
    except Exception as e:
        return None, html.Div([f"Error: {str(e)}"])

def render_gibson_schwartz(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                           kappa=None, sigma1=None, sigma2=None, rho=None, n_paths=None):
    try:
        # Use advanced parameters or defaults
        kappa = kappa if kappa is not None else 2.0
        sigma1 = sigma1 if sigma1 is not None else 0.25
        sigma2 = sigma2 if sigma2 is not None else 0.15
        rho = rho if rho is not None else -0.3
        n_paths = n_paths if n_paths is not None else 1000
        
        # Create model
        model = GibsonSchwartzModel(spot_price, risk_free_rate, kappa, convenience_yield, sigma1, sigma2, rho)
        
        # Monte Carlo simulation using advanced n_paths
        futures_price, std_error = model.monte_carlo_futures_price(
            convenience_yield, time_to_maturity, n_paths=n_paths
        )
        
        # Generate paths for visualization
        spot_paths, cy_paths = model.simulate_paths(
            convenience_yield, time_to_maturity, 100, min(n_paths, 100)
        )
        
        # Create visualizations
        time_grid = np.linspace(0, time_to_maturity, spot_paths.shape[1])
        
        # Spot price paths
        fig_spot = go.Figure()
        n_display = min(50, spot_paths.shape[0])
        for i in range(n_display):
            fig_spot.add_trace(go.Scatter(
                x=time_grid, y=spot_paths[i, :],
                mode='lines', line=dict(color='lightblue', width=1),
                opacity=0.3, showlegend=False
            ))
        
        mean_spot = np.mean(spot_paths, axis=0)
        fig_spot.add_trace(go.Scatter(
            x=time_grid, y=mean_spot,
            mode='lines', name='Mean Path',
            line=dict(color='red', width=3)
        ))
        fig_spot.update_layout(
            title="Spot Price Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Price ($)",
            template='plotly_white',
            height=400
        )
        
        # Convenience yield paths
        fig_cy = go.Figure()
        for i in range(n_display):
            fig_cy.add_trace(go.Scatter(
                x=time_grid, y=cy_paths[i, :],
                mode='lines', line=dict(color='lightgreen', width=1),
                opacity=0.3, showlegend=False
            ))
        
        mean_cy = np.mean(cy_paths, axis=0)
        fig_cy.add_trace(go.Scatter(
            x=time_grid, y=mean_cy,
            mode='lines', name='Mean Path',
            line=dict(color='darkgreen', width=3)
        ))
        fig_cy.update_layout(
            title="Convenience Yield Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Convenience Yield",
            template='plotly_white',
            height=400
        )
        
        rows = [{
            'Model': 'Gibson-Schwartz', 'Spot_Price': spot_price,
            'Risk_Free_Rate': risk_free_rate, 'Convenience_Yield': convenience_yield,
            'Time_to_Maturity': time_to_maturity, 'Kappa': kappa, 'Sigma1': sigma1,
            'Sigma2': sigma2, 'Rho': rho, 'Paths': n_paths,
            'Futures_Price': futures_price, 'Standard_Error': std_error,
        }]
        return rows, html.Div([
            html.H3("🌊 Gibson-Schwartz Stochastic Model"),
            
            # Key metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Futures Price (MC)"),
                        dbc.CardBody([html.H4(f"${futures_price:.2f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Standard Error"),
                        dbc.CardBody([html.H4(f"${std_error:.3f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Confidence Interval"),
                        dbc.CardBody([html.H4(f"±${1.96*std_error:.2f}", className="card-title")])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Parameter info
            dbc.Alert([
                html.H5("Model Parameters:"),
                html.P(f"Mean Reversion Speed (κ): {kappa}"),
                html.P(f"Spot Volatility (σ₁): {sigma1}"),
                html.P(f"CY Volatility (σ₂): {sigma2}"),
                html.P(f"Correlation (ρ): {rho}")
            ], color="info", className="mb-3"),
            
            # Visualizations
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_spot)], width=6),
                dbc.Col([dcc.Graph(figure=fig_cy)], width=6)
            ])
        ])
    except Exception as e:
        return None, html.Div([f"Error: {str(e)}"])

def render_jump_diffusion(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                          sigma=None, jump_intensity=None, jump_mean=None, jump_std=None, n_paths=None):
    try:
        # The sliders' values, or the defaults
        sigma = sigma if sigma is not None else 0.25  # Continuous volatility
        jump_intensity = jump_intensity if jump_intensity is not None else 0.5
        jump_mean = jump_mean if jump_mean is not None else -0.02
        jump_std = jump_std if jump_std is not None else 0.10
        n_paths = n_paths if n_paths is not None else 1000
        
        # Create model
        model = MertonJumpDiffusionModel(
            spot_price, risk_free_rate, convenience_yield, 
            sigma, jump_intensity, jump_mean, jump_std
        )
        
        # Calculate results
        analytical_price = model.analytical_futures_price(time_to_maturity)
        
        # Monte Carlo simulation
        paths = model.simulate_paths(time_to_maturity, 252, n_paths)
        mc_price = np.mean(paths[:, -1])
        terminal_prices = paths[:, -1]
        
        # Create visualizations
        time_grid = np.linspace(0, time_to_maturity, paths.shape[1])
        
        # Price paths
        fig_paths = go.Figure()
        n_display = min(50, paths.shape[0])
        for i in range(n_display):
            fig_paths.add_trace(go.Scatter(
                x=time_grid, y=paths[i, :],
                mode='lines', line=dict(color='lightblue', width=1),
                opacity=0.3, showlegend=False
            ))
        
        mean_path = np.mean(paths, axis=0)
        fig_paths.add_trace(go.Scatter(
            x=time_grid, y=mean_path,
            mode='lines', name='Mean Path',
            line=dict(color='red', width=3)
        ))
        fig_paths.update_layout(
            title="Jump-Diffusion Price Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Price ($)",
            template='plotly_white',
            height=400
        )
        
        # Terminal price distribution
        fig_dist = go.Figure(data=[go.Histogram(x=terminal_prices, nbinsx=30)])
        fig_dist.update_layout(
            title="Terminal Price Distribution",
            xaxis_title="Terminal Price ($)",
            yaxis_title="Frequency",
            template='plotly_white',
            height=400
        )
        
        rows = [{
            'Model': 'Merton Jump-Diffusion', 'Spot_Price': spot_price,
            'Risk_Free_Rate': risk_free_rate, 'Convenience_Yield': convenience_yield,
            'Time_to_Maturity': time_to_maturity, 'Sigma': sigma,
            'Jump_Intensity': jump_intensity, 'Jump_Mean': jump_mean, 'Jump_Std': jump_std,
            'Paths': n_paths, 'Futures_Price_Analytical': analytical_price,
            'Futures_Price_MC': mc_price,
        }]
        return rows, html.Div([
            html.H3("⚡ Merton Jump-Diffusion Model"),
            
            # Key metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Analytical Price"),
                        dbc.CardBody([html.H4(f"${analytical_price:.2f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Monte Carlo Price"),
                        dbc.CardBody([html.H4(f"${mc_price:.2f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Difference"),
                        dbc.CardBody([html.H4(f"${abs(analytical_price - mc_price):.3f}", className="card-title")])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Parameter info
            dbc.Alert([
                html.H5("Jump Parameters:"),
                html.P(f"Continuous Volatility (σ): {sigma}"),
                html.P(f"Jump Intensity (λ): {jump_intensity}"),
                html.P(f"Jump Mean (μⱼ): {jump_mean}"),
                html.P(f"Jump Volatility (σⱼ): {jump_std}")
            ], color="info", className="mb-3"),
            
            # Visualizations
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_paths)], width=6),
                dbc.Col([dcc.Graph(figure=fig_dist)], width=6)
            ])
        ])
    except Exception as e:
        return None, html.Div([f"Error: {str(e)}"])

def render_multi_factor(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                        sigma_s=None, kappa_delta=None, sigma_delta=None, kappa_r=None, sigma_r=None,
                        rho_s_delta=None, rho_s_r=None, rho_delta_r=None, n_paths=None):
    try:
        # The sliders' values, or the defaults
        sigma_s = sigma_s if sigma_s is not None else 0.25  # Spot price volatility
        kappa_delta = kappa_delta if kappa_delta is not None else 2.0  # CY mean reversion
        alpha_delta = convenience_yield  # CY long-term mean
        sigma_delta = sigma_delta if sigma_delta is not None else 0.15  # CY volatility
        kappa_r = kappa_r if kappa_r is not None else 0.5  # Rate mean reversion
        alpha_r = risk_free_rate  # Rate long-term mean
        sigma_r = sigma_r if sigma_r is not None else 0.02  # Rate volatility
        rho_s_delta = rho_s_delta if rho_s_delta is not None else -0.3
        rho_s_r = rho_s_r if rho_s_r is not None else 0.1
        rho_delta_r = rho_delta_r if rho_delta_r is not None else -0.2
        n_paths = n_paths if n_paths is not None else 1000
        
        # Correlation matrix (spot, convenience yield, rate)
        correlation_matrix = np.array([
            [1.0, rho_s_delta, rho_s_r],
            [rho_s_delta, 1.0, rho_delta_r],
            [rho_s_r, rho_delta_r, 1.0]
        ])
        
        # Create model
        model = MultiFactorModel(
            spot_price, convenience_yield, risk_free_rate,
            sigma_s, kappa_delta, alpha_delta, sigma_delta,
            kappa_r, alpha_r, sigma_r, correlation_matrix
        )
        
        # Monte Carlo simulation
        futures_price, std_error = model.monte_carlo_futures_price(time_to_maturity, n_paths=n_paths)
        
        # Generate paths for visualization
        spot_paths, delta_paths, rate_paths = model.simulate_paths(time_to_maturity, 100, min(n_paths, 100))
        
        # Create visualizations
        time_grid = np.linspace(0, time_to_maturity, spot_paths.shape[1])
        
        def create_factor_chart(paths, title, ylabel):
            fig = go.Figure()
            n_display = min(50, paths.shape[0])
            for i in range(n_display):
                fig.add_trace(go.Scatter(
                    x=time_grid, y=paths[i, :],
                    mode='lines', line=dict(color='lightblue', width=1),
                    opacity=0.3, showlegend=False
                ))
            
            mean_path = np.mean(paths, axis=0)
            fig.add_trace(go.Scatter(
                x=time_grid, y=mean_path,
                mode='lines', name='Mean Path',
                line=dict(color='red', width=3)
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Time (Years)",
                yaxis_title=ylabel,
                template='plotly_white',
                height=300
            )
            return fig
        
        fig_spot = create_factor_chart(spot_paths, "Spot Price Evolution", "Price ($)")
        fig_delta = create_factor_chart(delta_paths, "Convenience Yield Evolution", "Convenience Yield")
        fig_rate = create_factor_chart(rate_paths, "Interest Rate Evolution", "Interest Rate")
        
        # Correlation matrix heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=['Spot', 'Conv. Yield', 'Interest Rate'],
            y=['Spot', 'Conv. Yield', 'Interest Rate'],
            colorscale='RdBu',
            zmid=0
        ))
        fig_corr.update_layout(title="Factor Correlation Matrix", height=300)
        
        rows = [{
            'Model': 'Multi-Factor', 'Spot_Price': spot_price,
            'Risk_Free_Rate': risk_free_rate, 'Convenience_Yield': convenience_yield,
            'Time_to_Maturity': time_to_maturity, 'Sigma_S': sigma_s,
            'Kappa_Delta': kappa_delta, 'Sigma_Delta': sigma_delta, 'Kappa_R': kappa_r,
            'Sigma_R': sigma_r, 'Rho_S_Delta': rho_s_delta, 'Rho_S_R': rho_s_r,
            'Rho_Delta_R': rho_delta_r, 'Paths': n_paths,
            'Futures_Price': futures_price, 'Standard_Error': std_error,
        }]
        return rows, html.Div([
            html.H3("🔀 Multi-Factor Model (3-Factor)"),
            
            # Key metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Futures Price (MC)"),
                        dbc.CardBody([html.H4(f"${futures_price:.2f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Standard Error"),
                        dbc.CardBody([html.H4(f"${std_error:.3f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Confidence Interval"),
                        dbc.CardBody([html.H4(f"±${1.96*std_error:.2f}", className="card-title")])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Parameter info
            dbc.Alert([
                html.H5("Multi-Factor Parameters:"),
                html.P(f"Spot Volatility (σs): {sigma_s}"),
                html.P(f"CY Mean Reversion (κδ): {kappa_delta}"),
                html.P(f"Rate Mean Reversion (κr): {kappa_r}"),
                html.P(f"CY Volatility (σδ): {sigma_delta}"),
                html.P(f"Rate Volatility (σr): {sigma_r}")
            ], color="info", className="mb-3"),
            
            # Factor evolution charts
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_spot)], width=4),
                dbc.Col([dcc.Graph(figure=fig_delta)], width=4),
                dbc.Col([dcc.Graph(figure=fig_rate)], width=4)
            ]),
            
            # Correlation matrix
            html.H4("📊 Correlation Matrix", className="mt-4"),
            dcc.Graph(figure=fig_corr)
        ])
    except Exception as e:
        return None, html.Div([f"Error: {str(e)}"])

def render_term_structure(spot_price, risk_free_rate, time_to_maturity, convenience_yield,
                          level_vol=None, slope_vol=None, curve_vol=None, time_horizon=None, n_paths=None):
    try:
        # The sliders' values, or the defaults
        level_vol = level_vol if level_vol is not None else 0.15
        slope_vol = slope_vol if slope_vol is not None else 0.05
        curve_vol = curve_vol if curve_vol is not None else 0.03
        time_horizon = time_horizon if time_horizon is not None else 1.0
        n_paths = n_paths if n_paths is not None else 500

        # Create model
        model = TermStructureModel(spot_price, risk_free_rate, convenience_yield)
        model.factor_volatilities = [level_vol, slope_vol, curve_vol]
        
        # Calculate single futures price
        single_price = model.calculate_futures_price(time_to_maturity)
        
        # Term structure simulation maturities
        maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
        
        # Initial term structure
        initial_prices = [model.calculate_futures_price(T) for T in maturities]
        
        # Simulate term structure evolution
        futures_curves = model.simulate_term_structure(maturities, time_horizon, 100, n_paths)
        
        # Create visualizations
        # Initial term structure
        fig_initial = go.Figure()
        fig_initial.add_trace(go.Scatter(
            x=maturities, y=initial_prices,
            mode='lines+markers', name='Initial Term Structure',
            line=dict(color='blue', width=3)
        ))
        fig_initial.add_hline(y=spot_price, line_dash="dash", line_color="red", annotation_text="Spot Price")
        fig_initial.update_layout(
            title="Initial Futures Term Structure",
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Futures Price ($)",
            template='plotly_white',
            height=400
        )
        
        # Final term structure distribution
        final_curves = futures_curves[:, :, -1]  # Final time step
        
        fig_final = go.Figure()
        
        # Show sample paths
        for i in range(min(20, n_paths)):
            fig_final.add_trace(go.Scatter(
                x=maturities, y=final_curves[:, i],
                mode='lines', opacity=0.3, showlegend=False,
                line=dict(color='lightblue', width=1)
            ))
        
        # Show mean curve
        mean_final = np.mean(final_curves, axis=1)
        fig_final.add_trace(go.Scatter(
            x=maturities, y=mean_final,
            mode='lines+markers', name=f'Mean Final Curve (T={time_horizon}Y)',
            line=dict(color='red', width=3)
        ))
        
        fig_final.update_layout(
            title=f"Term Structure After {time_horizon} Years",
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Futures Price ($)",
            template='plotly_white',
            height=400
        )
        
        # Factor information table
        factor_data = {
            'Factor': ['Level', 'Slope', 'Curvature'],
            'Volatility': [f"{vol:.1%}" for vol in model.factor_volatilities],
            'Mean Reversion': [f"{kappa:.1f}" for kappa in model.factor_kappa],
            'Description': [
                'Parallel shifts in entire curve',
                'Steepening/flattening of curve', 
                'Changes in curve curvature'
            ]
        }
        factor_df = pd.DataFrame(factor_data)
        
        rows = [{
            'Model': 'Term Structure', 'Spot_Price': spot_price,
            'Risk_Free_Rate': risk_free_rate, 'Convenience_Yield': convenience_yield,
            'Time_to_Maturity': time_to_maturity, 'Level_Vol': level_vol,
            'Slope_Vol': slope_vol, 'Curvature_Vol': curve_vol,
            'Simulation_Horizon': time_horizon, 'Paths': n_paths,
            'Futures_Price': single_price, 'Basis': single_price - spot_price,
        }]
        return rows, html.Div([
            html.H3("📈 Term Structure Model (PCA-Based)"),
            
            # Key metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Futures Price"),
                        dbc.CardBody([html.H4(f"${single_price:.2f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Basis (F-S)"),
                        dbc.CardBody([html.H4(f"${single_price - spot_price:.2f}", className="card-title")])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Net Carry Rate"),
                        dbc.CardBody([html.H4(f"{(risk_free_rate - convenience_yield):.2%}", className="card-title")])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Visualizations
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_initial)], width=6),
                dbc.Col([dcc.Graph(figure=fig_final)], width=6)
            ]),
            
            # Factor analysis
            html.H4("📊 Factor Analysis", className="mt-4"),
            dbc.Alert(
                "💡 This model uses Principal Component Analysis (PCA) with Level, Slope, and Curvature factors to model the entire futures curve evolution.",
                color="info",
                className="mb-3"
            ),
            
            # Factor table
            dash_table.DataTable(
                data=factor_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in factor_df.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ])
    except Exception as e:
        return None, html.Div([f"Error: {str(e)}"])

def render_model_comparison(spot_price, risk_free_rate, time_to_maturity, storage_cost, convenience_yield):
    try:
        # Calculate prices from all 5 models
        coc_model = CostOfCarryModel(spot_price, risk_free_rate, storage_cost, convenience_yield)
        coc_price = coc_model.futures_price(time_to_maturity)
        
        gs_model = GibsonSchwartzModel(spot_price, risk_free_rate, 2.0, convenience_yield, 0.25, 0.15, -0.3)
        gs_price, _ = gs_model.monte_carlo_futures_price(convenience_yield, time_to_maturity, n_paths=1000)
        
        mjd_model = MertonJumpDiffusionModel(spot_price, risk_free_rate, convenience_yield, 0.25, 0.5, -0.02, 0.10)
        mjd_price = mjd_model.analytical_futures_price(time_to_maturity)
        
        # Multi-factor model
        correlation_matrix = np.array([[1.0, -0.3, 0.1], [-0.3, 1.0, -0.2], [0.1, -0.2, 1.0]])
        mf_model = MultiFactorModel(
            spot_price, convenience_yield, risk_free_rate,
            0.25, 2.0, convenience_yield, 0.15, 0.5, risk_free_rate, 0.02, correlation_matrix
        )
        mf_price, _ = mf_model.monte_carlo_futures_price(time_to_maturity, n_paths=1000)
        
        # Term structure model
        ts_model = TermStructureModel(spot_price, risk_free_rate, convenience_yield)
        ts_price = ts_model.calculate_futures_price(time_to_maturity)
        
        # Display comparison table
        comparison_data = {
            'Model': ['Cost of Carry', 'Gibson-Schwartz', 'Jump-Diffusion', 'Multi-Factor', 'Term Structure'],
            'Futures Price': [f"${coc_price:.2f}", f"${gs_price:.2f}", f"${mjd_price:.2f}", f"${mf_price:.2f}", f"${ts_price:.2f}"],
            'Basis (F-S)': [f"${coc_price-spot_price:.2f}", f"${gs_price-spot_price:.2f}", f"${mjd_price-spot_price:.2f}", f"${mf_price-spot_price:.2f}", f"${ts_price-spot_price:.2f}"],
            'Complexity': ['Low', 'Medium', 'Medium', 'High', 'High'],
            'Best For': ['Financial Assets', 'Commodities', 'Crisis Events', 'Complex Portfolios', 'Full Curve Analysis']
        }
        comparison_df = pd.DataFrame(comparison_data)
        
        # Price comparison chart
        models_data = {
            'Cost of Carry': coc_price,
            'Gibson-Schwartz': gs_price,
            'Jump-Diffusion': mjd_price,
            'Multi-Factor': mf_price,
            'Term Structure': ts_price
        }
        
        fig_comparison = go.Figure(data=[
            go.Bar(x=list(models_data.keys()), y=list(models_data.values()),
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ])
        fig_comparison.add_hline(y=spot_price, line_dash="dash", line_color="red", annotation_text="Spot Price")
        fig_comparison.update_layout(
            title="Futures Price Comparison Across All Models",
            xaxis_title="Pricing Model",
            yaxis_title="Futures Price ($)",
            template='plotly_white',
            height=400
        )
        
        # Term structure comparison
        maturities = np.linspace(0.25, 3.0, 12)
        
        models_term_data = {
            'Cost of Carry': [coc_model.futures_price(t) for t in maturities],
            'Gibson-Schwartz': [gs_model.monte_carlo_futures_price(convenience_yield, t, n_paths=500)[0] for t in maturities],
            'Jump-Diffusion': [mjd_model.analytical_futures_price(t) for t in maturities],
            'Multi-Factor': [mf_model.monte_carlo_futures_price(t, n_paths=500)[0] for t in maturities],
            'Term Structure': [ts_model.calculate_futures_price(t) for t in maturities]
        }
        
        fig_term_comparison = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model_name, prices) in enumerate(models_term_data.items()):
            fig_term_comparison.add_trace(go.Scatter(
                x=maturities, y=prices,
                mode='lines+markers', name=model_name,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig_term_comparison.update_layout(
            title="Futures Term Structure Comparison",
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Futures Price ($)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        # Model statistics
        prices = [coc_price, gs_price, mjd_price, mf_price, ts_price]
        price_std = np.std(prices)
        price_range = max(prices) - min(prices)
        complexity_scores = {'Low': 1, 'Medium': 2, 'High': 3}
        avg_complexity = np.mean([complexity_scores[c] for c in comparison_df['Complexity']])
        
        rows = [{
            'Model': name, 'Spot_Price': spot_price, 'Risk_Free_Rate': risk_free_rate,
            'Storage_Cost': storage_cost if name == 'Cost of Carry' else 0.0,
            'Convenience_Yield': convenience_yield, 'Time_to_Maturity': time_to_maturity,
            'Futures_Price': price, 'Basis': price - spot_price,
        } for name, price in models_data.items()]
        return rows, html.Div([
            html.H3("🔄 Model Comparison (All 5 Models)"),
            dbc.Alert(
                "Storage cost is included in the Cost of Carry price only; the other four "
                "models price without it. With a storage cost above 0%, part of the "
                "difference between Cost of Carry and the others is that storage cost.",
                color="secondary", className="mt-2"
            ),
            
            # Comparison table
            html.H4("📊 Model Comparison Table", className="mt-4 mb-3"),
            dash_table.DataTable(
                data=comparison_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in comparison_df.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Futures Price'},
                        'backgroundColor': 'rgb(248, 248, 255)',
                        'fontWeight': 'bold'
                    }
                ]
            ),
            
            # Price comparison chart
            html.H4("📈 Price Comparison", className="mt-4 mb-3"),
            dcc.Graph(figure=fig_comparison),
            
            # Term structure comparison
            html.H4("📈 Term Structure Comparison", className="mt-4 mb-3"),
            dcc.Graph(figure=fig_term_comparison),
            
            # Model statistics
            html.H4("📊 Model Statistics", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price Standard Deviation"),
                        dbc.CardBody([html.H5(f"${price_std:.3f}")])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price Range"),
                        dbc.CardBody([html.H5(f"${price_range:.3f}")])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Average Complexity"),
                        dbc.CardBody([html.H5(f"{avg_complexity:.1f}/3.0")])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Models Available"),
                        dbc.CardBody([html.H5(f"{len(comparison_df)}/5")])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Recommendations
            html.H4("💡 Model Selection Recommendations", className="mt-4 mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5("🏦 Financial Assets"),
                        html.P("Use: Cost of Carry"),
                        html.P("Reason: Simple, accurate, fast calculations")
                    ], color="primary")
                ], width=4),
                dbc.Col([
                    dbc.Alert([
                        html.H5("🛢️ Commodities"),
                        html.P("Use: Gibson-Schwartz"),
                        html.P("Reason: Captures storage dynamics")
                    ], color="warning")
                ], width=4),
                dbc.Col([
                    dbc.Alert([
                        html.H5("⚡ Crisis Periods"),
                        html.P("Use: Jump-Diffusion"),
                        html.P("Reason: Models extreme events")
                    ], color="danger")
                ], width=4)
            ])
        ])
    except Exception as e:
        return None, html.Div([f"Error: {str(e)}"])

def render_portfolio_simulation():
    """Render Portfolio Simulation interface"""
    return html.Div([
        html.H3("📊 Portfolio Monte Carlo Simulation"),
        html.P("A simplified Monte Carlo of a multi-asset portfolio: every asset is simulated as a "
               "correlated geometric Brownian motion, not with its own pricing model. Drift: S&P 500 3%, "
               "EUR/USD 2%, the others 5% a year; volatility: Gold 25%, the others 20%.",
               className="lead"),
        
        # Portfolio Configuration Section
        dbc.Card([
            dbc.CardHeader(html.H5("🎯 Portfolio Configuration")),
            dbc.CardBody([
                html.H6("Asset Selection and Allocation:", className="mb-3"),
                
                # Asset checkboxes and sliders
                html.Div(id='asset-selection-container', children=[
                    # S&P 500
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='sp500-checkbox',
                                options=[{'label': 'S&P 500', 'value': 'include'}],
                                value=['include'],
                                inline=True
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Weight (%):"),
                            dcc.Slider(id='sp500-weight', min=0, max=100, step=5, value=40, 
                                      marks={0: '0%', 50: '50%', 100: '100%'})
                        ], width=9)
                    ], className="mb-2"),
                    
                    # Crude Oil
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='oil-checkbox',
                                options=[{'label': 'Crude Oil', 'value': 'include'}],
                                value=['include'],
                                inline=True
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Weight (%):"),
                            dcc.Slider(id='oil-weight', min=0, max=100, step=5, value=25,
                                      marks={0: '0%', 50: '50%', 100: '100%'})
                        ], width=9)
                    ], className="mb-2"),
                    
                    # Gold
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='gold-checkbox',
                                options=[{'label': 'Gold', 'value': 'include'}],
                                value=['include'],
                                inline=True
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Weight (%):"),
                            dcc.Slider(id='gold-weight', min=0, max=100, step=5, value=20,
                                      marks={0: '0%', 50: '50%', 100: '100%'})
                        ], width=9)
                    ], className="mb-2"),
                    
                    # EUR/USD
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='eurusd-checkbox',
                                options=[{'label': 'EUR/USD', 'value': 'include'}],
                                value=['include'],
                                inline=True
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Weight (%):"),
                            dcc.Slider(id='eurusd-weight', min=0, max=100, step=5, value=10,
                                      marks={0: '0%', 50: '50%', 100: '100%'})
                        ], width=9)
                    ], className="mb-2"),
                    
                    # Natural Gas
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='natgas-checkbox',
                                options=[{'label': 'Natural Gas', 'value': 'include'}],
                                value=['include'],
                                inline=True
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Weight (%):"),
                            dcc.Slider(id='natgas-weight', min=0, max=100, step=5, value=5,
                                      marks={0: '0%', 50: '50%', 100: '100%'})
                        ], width=9)
                    ], className="mb-3")
                ]),
                
                # Weight validation
                html.Div(id='weight-validation', className="mb-3"),
                
                # Simulation Parameters
                html.H6("⚙️ Simulation Parameters:", className="mb-3 mt-4"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Time Horizon (Years):"),
                        dcc.Slider(id='portfolio-time-horizon', min=0.5, max=5.0, step=0.5, value=2.0,
                                  marks={0.5: '0.5', 2.0: '2', 5.0: '5'})
                    ], width=4),
                    dbc.Col([
                        html.Label("Number of Paths:"),
                        dcc.Dropdown(
                            id='portfolio-paths',
                            options=[
                                {'label': '1,000', 'value': 1000},
                                {'label': '2,500', 'value': 2500},
                                {'label': '5,000', 'value': 5000},
                                {'label': '10,000', 'value': 10000}
                            ],
                            value=2500
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Time Steps:"),
                        dcc.Slider(id='portfolio-steps', min=50, max=500, step=50, value=252,
                                  marks={50: '50', 252: '252', 500: '500'})
                    ], width=4)
                ], className="mb-3"),
                
                # Correlation Settings
                dbc.Collapse([
                    dbc.Card([
                        dbc.CardHeader("📈 Correlation Settings"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Base Correlation:"),
                                    dcc.Slider(id='correlation-strength', min=0.0, max=0.8, step=0.1, value=0.3,
                                              marks={0: '0', 0.4: '0.4', 0.8: '0.8'})
                                ], width=12)
                            ])
                        ])
                    ])
                ], id='correlation-collapse', is_open=False),
                
                dbc.Button("🔧 Advanced Correlation Settings", id='correlation-toggle', 
                          color="outline-secondary", size="sm", className="mb-3"),
                
                # Run Simulation Button
                dbc.Row([
                    dbc.Col([
                        dbc.Button("🚀 Run Portfolio Simulation", id='run-portfolio-sim', 
                                  color="primary", size="lg", className="w-100")
                    ], width=12)
                ], className="mt-4")
            ])
        ], className="mb-4"),
        
        # Results Section (initially hidden)
        html.Div(id='portfolio-results', children=[], style={'display': 'none'}),
        
        # Loading component
        dcc.Loading(
            id="portfolio-loading",
            children=[html.Div(id="portfolio-loading-output")],
            type="default"
        )
    ])

# =====================================================
# PORTFOLIO SIMULATION CALLBACKS
# =====================================================

# Callback for correlation settings toggle
@app.callback(
    Output('correlation-collapse', 'is_open'),
    Input('correlation-toggle', 'n_clicks'),
    State('correlation-collapse', 'is_open')
)
def toggle_correlation_settings(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for weight validation
@app.callback(
    Output('weight-validation', 'children'),
    [Input('sp500-checkbox', 'value'),
     Input('oil-checkbox', 'value'),
     Input('gold-checkbox', 'value'),
     Input('eurusd-checkbox', 'value'),
     Input('natgas-checkbox', 'value'),
     Input('sp500-weight', 'value'),
     Input('oil-weight', 'value'),
     Input('gold-weight', 'value'),
     Input('eurusd-weight', 'value'),
     Input('natgas-weight', 'value')]
)
def validate_weights(sp500_check, oil_check, gold_check, eurusd_check, natgas_check,
                    sp500_weight, oil_weight, gold_weight, eurusd_weight, natgas_weight):
    
    # Calculate total weight of selected assets
    total_weight = 0
    selected_assets = 0
    
    if sp500_check:
        total_weight += sp500_weight
        selected_assets += 1
    if oil_check:
        total_weight += oil_weight
        selected_assets += 1
    if gold_check:
        total_weight += gold_weight
        selected_assets += 1
    if eurusd_check:
        total_weight += eurusd_weight
        selected_assets += 1
    if natgas_check:
        total_weight += natgas_weight
        selected_assets += 1
    
    if selected_assets < 2:
        return dbc.Alert("❌ Select at least 2 assets for portfolio simulation.", color="danger")
    elif total_weight != 100:
        return dbc.Alert(f"⚠️ Weights sum to {total_weight}%. Adjust to 100%.", color="warning")
    else:
        return dbc.Alert("✅ Portfolio configuration is valid!", color="success")

# Main Portfolio Simulation callback
@app.callback(
    [Output('portfolio-results', 'children'),
     Output('portfolio-results', 'style'),
     Output('portfolio-loading-output', 'children'),
     Output('futures-results', 'data', allow_duplicate=True)],
    Input('run-portfolio-sim', 'n_clicks'),
    [State('sp500-checkbox', 'value'),
     State('oil-checkbox', 'value'),
     State('gold-checkbox', 'value'),
     State('eurusd-checkbox', 'value'),
     State('natgas-checkbox', 'value'),
     State('sp500-weight', 'value'),
     State('oil-weight', 'value'),
     State('gold-weight', 'value'),
     State('eurusd-weight', 'value'),
     State('natgas-weight', 'value'),
     State('portfolio-time-horizon', 'value'),
     State('portfolio-paths', 'value'),
     State('portfolio-steps', 'value'),
     State('correlation-strength', 'value')],
    prevent_initial_call=True
)
def run_portfolio_simulation(n_clicks, sp500_check, oil_check, gold_check, eurusd_check, natgas_check,
                           sp500_weight, oil_weight, gold_weight, eurusd_weight, natgas_weight,
                           time_horizon, n_paths, n_steps, correlation_strength):

    if n_clicks is None:
        return [], {'display': 'none'}, "", None
    
    try:
        # Asset configurations
        available_assets = {
            'S&P 500': {'weight': sp500_weight/100, 'model': 'cost_of_carry', 'spot': 4500, 'rate': 0.05, 'div': 0.02, 'selected': bool(sp500_check)},
            'Crude Oil': {'weight': oil_weight/100, 'model': 'gibson_schwartz', 'spot': 75, 'rate': 0.05, 'storage': 0.02, 'cy': 0.01, 'selected': bool(oil_check)},
            'Gold': {'weight': gold_weight/100, 'model': 'jump_diffusion', 'spot': 2000, 'rate': 0.05, 'storage': 0.005, 'cy': 0.0, 'selected': bool(gold_check)},
            'EUR/USD': {'weight': eurusd_weight/100, 'model': 'cost_of_carry', 'spot': 1.08, 'rate_dom': 0.05, 'rate_for': 0.03, 'selected': bool(eurusd_check)},
            'Natural Gas': {'weight': natgas_weight/100, 'model': 'gibson_schwartz', 'spot': 3.5, 'rate': 0.05, 'storage': 0.20, 'cy': 0.05, 'selected': bool(natgas_check)}
        }
        
        # Filter selected assets and validate
        selected_assets = {k: v for k, v in available_assets.items() if v['selected']}
        
        if len(selected_assets) < 2:
            return [dbc.Alert("❌ Select at least 2 assets", color="danger")], {'display': 'block'}, "", None

        total_weight = sum(config['weight'] for config in selected_assets.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            return [dbc.Alert(f"❌ Weights must sum to 100%, currently {total_weight*100:.1f}%", color="danger")], {'display': 'block'}, "", None
        
        # Create asset configurations for PortfolioSimulator
        assets_config = {}
        for asset_name, config in selected_assets.items():
            if config['model'] == 'cost_of_carry':
                if 'rate_dom' in config:  # Currency
                    params = {
                        'spot_price': config['spot'],
                        'risk_free_rate': config['rate_dom'] - config['rate_for'],
                        'storage_cost': 0.0,
                        'convenience_yield': 0.0
                    }
                else:  # Equity
                    params = {
                        'spot_price': config['spot'],
                        'risk_free_rate': config['rate'],
                        'storage_cost': 0.0,
                        'convenience_yield': config.get('div', 0.0)
                    }
            elif config['model'] == 'gibson_schwartz':
                params = {
                    'spot_price': config['spot'],
                    'risk_free_rate': config['rate'],
                    'kappa': 2.0,
                    'alpha': config.get('cy', 0.02),
                    'sigma1': 0.25,
                    'sigma2': 0.15,
                    'rho': -0.3
                }
            elif config['model'] == 'jump_diffusion':
                params = {
                    'spot_price': config['spot'],
                    'risk_free_rate': config['rate'],
                    'convenience_yield': config.get('cy', 0.0),
                    'sigma': 0.25,
                    'jump_intensity': 0.5,
                    'jump_mean': -0.02,
                    'jump_std': 0.10
                }
            else:  # Default to cost_of_carry
                params = {
                    'spot_price': config['spot'],
                    'risk_free_rate': config['rate'],
                    'storage_cost': config.get('storage', 0.0),
                    'convenience_yield': config.get('cy', 0.0)
                }
            
            assets_config[asset_name] = {
                'weight': config['weight'],
                'model': config['model'],
                'params': params
            }
        
        # Create correlation matrix
        n_assets = len(selected_assets)
        correlation_matrix = np.eye(n_assets) + correlation_strength * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
        
        # Run simulation
        portfolio_sim = PortfolioSimulator(assets_config, correlation_matrix)
        portfolio_values, asset_paths = portfolio_sim.simulate_portfolio(time_horizon, n_steps, n_paths)
        risk_metrics = portfolio_sim.calculate_risk_metrics(portfolio_values, time_horizon / n_steps)

        # Generate results display
        results = create_portfolio_results_display(portfolio_values, asset_paths, risk_metrics, selected_assets, time_horizon, n_steps, n_paths)

        rows = [{
            'Model': 'Portfolio Simulation',
            'Assets': ', '.join(f"{name} {config['weight']:.0%}" for name, config in selected_assets.items()),
            'Time_Horizon_Years': time_horizon, 'Paths': n_paths, 'Steps': n_steps,
            'Base_Correlation': correlation_strength,
            'Expected_Return_Pct': risk_metrics['total_return_mean'],
            'Return_Std_Pct': risk_metrics['total_return_std'],
            'Annualized_Volatility_Pct': risk_metrics['volatility_annualized'],
            'Sharpe_Ratio_vs_5pct': risk_metrics['sharpe_ratio'],
            'Max_Drawdown_Pct': risk_metrics['max_drawdown'],
            'VaR_95_Pct': risk_metrics['var_95'], 'VaR_99_Pct': risk_metrics['var_99'],
            'CVaR_95_Pct': risk_metrics['cvar_95'],
        }]
        return results, {'display': 'block'}, "", rows

    except Exception as e:
        error_msg = f"Simulation Error: {str(e)}"
        return [dbc.Alert(error_msg, color="danger")], {'display': 'block'}, "", None

def create_portfolio_results_display(portfolio_values, asset_paths, risk_metrics, selected_assets, time_horizon, n_steps, n_paths):
    """Create comprehensive portfolio results display"""
    
    # Key metrics cards
    metrics_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Expected Return"),
                dbc.CardBody([
                    html.H4(f"{risk_metrics['total_return_mean']:.1f}%", className="text-primary"),
                    html.P(f"±{risk_metrics['total_return_std']:.1f}%", className="text-muted small")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Annualized Volatility"),
                dbc.CardBody([
                    html.H4(f"{risk_metrics['volatility_annualized']:.1f}%", className="text-warning")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sharpe Ratio (vs 5%)"),
                dbc.CardBody([
                    html.H4(f"{risk_metrics['sharpe_ratio']:.2f}", className="text-success")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Max Drawdown"),
                dbc.CardBody([
                    html.H4(f"{risk_metrics['max_drawdown']:.1f}%", className="text-danger")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("VaR (95%)"),
                dbc.CardBody([
                    html.H4(f"{risk_metrics['var_95']:.1f}%", className="text-info")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Paths"),
                dbc.CardBody([
                    html.H4(f"{n_paths:,}", className="text-secondary")
                ])
            ])
        ], width=2)
    ], className="mb-4")
    
    # Portfolio evolution chart
    time_grid = np.linspace(0, time_horizon, n_steps + 1)
    
    fig_evolution = go.Figure()
    
    # Show sample paths
    n_display = min(100, portfolio_values.shape[0])
    for i in range(n_display):
        fig_evolution.add_trace(go.Scatter(
            x=time_grid, y=portfolio_values[i, :],
            mode='lines', line=dict(color='lightblue', width=0.5),
            opacity=0.3, showlegend=False
        ))
    
    # Mean path
    mean_path = np.mean(portfolio_values, axis=0)
    fig_evolution.add_trace(go.Scatter(
        x=time_grid, y=mean_path,
        mode='lines', name='Mean Portfolio Value',
        line=dict(color='red', width=3)
    ))
    
    # Confidence bands
    percentiles = np.percentile(portfolio_values, [5, 25, 75, 95], axis=0)
    
    fig_evolution.add_trace(go.Scatter(
        x=time_grid, y=percentiles[3],
        mode='lines', line=dict(color='rgba(0,100,80,0)'),
        showlegend=False
    ))
    
    fig_evolution.add_trace(go.Scatter(
        x=time_grid, y=percentiles[0],
        mode='lines', fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(0,100,80,0)'),
        name='90% Confidence Band'
    ))
    
    fig_evolution.update_layout(
        title=f"Portfolio Simulation ({n_paths:,} paths)",
        xaxis_title="Time (Years)",
        yaxis_title="Portfolio Value ($)",
        template='plotly_white',
        height=500
    )
    
    # Return distribution
    final_returns = (risk_metrics['final_values'] / portfolio_values[:, 0] - 1) * 100
    
    fig_distribution = go.Figure(data=[go.Histogram(x=final_returns, nbinsx=50)])
    fig_distribution.add_vline(x=np.mean(final_returns), line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: {np.mean(final_returns):.1f}%")
    fig_distribution.add_vline(x=risk_metrics['var_95'], line_dash="dash", line_color="orange",
                              annotation_text=f"VaR 95%: {risk_metrics['var_95']:.1f}%")
    
    fig_distribution.update_layout(
        title="Portfolio Return Distribution",
        xaxis_title="Total Return (%)",
        yaxis_title="Frequency",
        template='plotly_white',
        height=400
    )
    
    # Portfolio composition pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(selected_assets.keys()),
        values=[config['weight'] for config in selected_assets.values()],
        hole=0.3
    )])
    fig_pie.update_layout(
        title="Asset Allocation",
        height=400,
        showlegend=True
    )
    
    # Risk metrics table
    risk_data = {
        'Metric': ['Expected Return', 'Return Volatility', 'Sharpe Ratio (vs 5% risk-free)', 'Max Drawdown', 'VaR (95%)', 'VaR (99%)', 'CVaR (95%)'],
        'Value': [
            f"{risk_metrics['total_return_mean']:.2f}%",
            f"{risk_metrics['total_return_std']:.2f}%",
            f"{risk_metrics['sharpe_ratio']:.3f}",
            f"{risk_metrics['max_drawdown']:.2f}%",
            f"{risk_metrics['var_95']:.2f}%",
            f"{risk_metrics['var_99']:.2f}%",
            f"{risk_metrics['cvar_95']:.2f}%"
        ]
    }
    risk_df = pd.DataFrame(risk_data)
    
    return [
        html.H3("📊 Portfolio Simulation Results", className="mb-4"),
        
        # Key metrics
        metrics_cards,
        
        # Charts section
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_evolution)
            ], width=8),
            dbc.Col([
                dcc.Graph(figure=fig_pie)
            ], width=4)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_distribution)
            ], width=8),
            dbc.Col([
                html.H5("Risk Metrics", className="mb-3"),
                dash_table.DataTable(
                    data=risk_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in risk_df.columns],
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'Value'},
                            'fontWeight': 'bold'
                        }
                    ]
                )
            ], width=4)
        ]),
        
        # Summary
        dbc.Alert([
            html.H5("✅ Simulation Summary"),
            html.P(f"Successfully analyzed {n_paths:,} scenarios over {time_horizon} years using {len(selected_assets)} assets."),
            html.P(f"Portfolio shows expected return of {risk_metrics['total_return_mean']:.1f}% with volatility of {risk_metrics['volatility_annualized']:.1f}%.")
        ], color="success", className="mt-4")
    ]

# =====================================================
# RUN THE APP
# =====================================================

# The page is built on each load, after every component builder above exists
app.layout = serve_layout

if __name__ == '__main__':
    port = int(os.getenv('PORT_FUTURES', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)