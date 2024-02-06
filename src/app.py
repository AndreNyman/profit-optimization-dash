import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import base64
from io import BytesIO
import plotly.express as px
from dash import dash_table
width1 = 2

optimization_history = []


# Initialize Dash app
#app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Define app layout
app.layout = html.Div([
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label='Problem Definition', children=[
        html.Br(),
        html.Br(),
        dbc.Row([
                dbc.Col([
                html.H1("Dynamic Pricing and Profit Optimization",style={"color": "#1c3e4d", "font": "bold", 'margin': '20px'}),
                ]),
                dbc.Col([
                html.H1("The Model", style={"color": "#1c3e4d", "font-weight": "bold", 'margin': '20px'}),
                ]),
        ]),
        html.Br(),
        
        dbc.Row([
                dbc.Col([
                dcc.Markdown('''
                This application utilizes non-linear programming algorithms in scipy minimize to dynamically set prices and optimize total profit 
                            
                depending on the marketing-adjusted demand function, the production function and cost function of the product.
                
                There are two models in the app, the baseline and the labor-augmented model. The problem in the baseline model is to maximize profits 
                            
                with respect to prices and marketing expenditure, given the demand and cost functions, which are formed as a non-linear optimization 
                            
                case with the added constraint that marketing expenses should not exceed the calibrated max parameter. 

                ''', mathjax=True, style={'margin': '20px'}),

                html.Br(),

                dcc.Markdown('''
                These models can optimize profits with respect to prices, marketing and labor. In a dynamic setup, a firm could estimate its products' demand, 
                            
                cost and production functions by fitting different model specifications to their data. These equations can be dynamic and ingested to the 
                            
                optimization module, which fetches the optimal values for that instance. An example could be in a dynamic pricing setting, where the demand function
                            
                is dependent on the time of day, the season, expectations (forecasts), the price (incl. discounts) and other factors. The models can also be adjusted to account 
                            
                for different types of functions and more complex problems, e.g., quasi-linear demand functions or demand equations that are dependent on more variables.
                             
                In the case of non-smooth or non-differentiable problems, non-gradient iterative methods such as the Nelder-Mead method should be adopted.

                ''', mathjax=True, style={'margin': '20px'}),

                html.Br(),

                dcc.Markdown('''
                
                In the baseline model, we maximize profits (5) subject to the marketing-adjusted demand function (1), cost function (3) and 
                            
                marketing expenditure constraint (6):

                ''', mathjax=True, style={'margin': '20px'}),

                html.Br(),

                dcc.Markdown('''
                The labor-augmented model builds on the baseline model by adopting a production process and a labor cost function. 
                
                These functions are simplified but can be adjusted to yield non-linearities or more complex structures.
                            
                The problem is still to maximize profits, but in this model, with respect to prices, marketing and labor. 
                            
                Furthermore, the model assumes a market clearing condition, where it is assumed that the produced amount equals demand, 
                            
                meaning that inventory complexities are excluded.
                
                ''', mathjax=True, style={'margin': '20px'}),

                html.Br(),

                dcc.Markdown('''
                Note that there are also constraints added to prices and marketing, ensuring that they are non-negative. Similarly, a penalty term 
                
                has been added to quantity, enforcing non-negative values.
                
                Some methods are unable to handle the constraints, if so, the constraints are excluded.

                ''', mathjax=True, style={'margin': '20px'}),
            ], width=6),

                dbc.Col([
                
                dcc.Markdown('''
                1) Marketing-adjusted demand function: $D_t = C_d - Beta_p * P_t^{2} + (MarketingMultiplier * M_t - Beta_m * M_t^{2})$
                
                2) Revenue: $R_t = D_t * P_t$
                
                3) Costs: $Cost_t = VariableCost_t * Quantity_t + FixedCost + M_t$
                
                4) Quantity: $Quantity_t = C_d - beta_p * P_t^{2} + (MarketingMultiplier * M_t - beta_m * M_t^{2})$
                
                5) Profit (to maximize): $Profit_t = R_t - Cost_t$
                            
                6) Marketing expenditure inequality constraint: $M_t <= selected maximum$

                ''', mathjax=True, style={'margin': '20px'}),

                

                html.Br(),

                dcc.Markdown('''
                The labor-augmented model adds: 
                            
                7) The production function: $Quantity_t = productivity * Labor_t$
                
                8) The labor cost function: $LaborCosts_t = WageParameter * Labor_t$
                
                9) Market clearing condition: $Quantity_t = D_t$
                
                ''', mathjax=True, style={'margin': '20px'}),
            ], width=6)
        ]),
        

        html.Br(),

        
        
        ]),

        dcc.Tab(label='Model Solution and Simulation', children=[
        html.Br(),
        html.Div([
            html.H3("Parameters", style={"color": "#1c3e4d", "font-weight": "bold"}),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Label("Variable Cost: "),
                    dcc.Input(id='variable-cost', type='number', value=5, step=0.1),
                    html.Label("Fixed Cost: "),
                    dcc.Input(id='fixed-cost', type='number', value=1000, step=10),
                    html.Label("Constant Demand: "),
                    dcc.Input(id='constant-demand', type='number', value=250, step=10),
                    html.Label("Price Coefficient: "),
                    dcc.Input(id='beta-p', type='number', value=0.1, step=0.05),
                    html.Label("Marketing Quadr. Coef.: "),
                    dcc.Input(id='beta-m', type='number', value=0.1, step=0.05),
                    html.Label("Marketing Coefficient: "),
                    dcc.Input(id='marketing-multiplier', type='number', value=5, step=0.05),
                    
                ], width=1),
                dbc.Col([
                    html.Label("Max Marketing Exp.:"),
                    dcc.Input(id='max-marketing-expenditure', type='number', value=100, step=0.1),
                    html.Label("Initial Guess, Price: "),
                    dcc.Input(id='initial-state-price', type='number', value=25, step=0.1),
                    html.Label("Initial Guess, Marketing: "),
                    dcc.Input(id='initial-state-marketing', type='number', value=50, step=0.1),
                    html.Label("Linspace Start: "),
                    dcc.Input(id='linspace-start', type='number', value=0, step=0.1),
                    html.Label("Linspace End: "),
                    dcc.Input(id='linspace-end', type='number', value=50, step=0.1),
                    html.Label("Linspace Num: "),
                    dcc.Input(id='linspace-num', type='number', value=50, step=0.1),
                ], width=1),
            ]),
        ], style={'margin': '20px'}),




    
        html.Br(),
        #html.Label("Optimization Algorithm: ", style={'margin': '20px'}),
        html.H3("Optimization Method", style={"color": "#1c3e4d", "font-weight": "bold", 'margin': '20px'}),
        html.Div([
            
            dcc.Dropdown(
                id='optimization-algorithm',
                options=[
                    {'label': 'L-BFGS-B', 'value': 'L-BFGS-B'},
                    {'label': 'COBYLA', 'value': 'COBYLA'},
                    {'label': 'SLSQP', 'value': 'SLSQP'},
                    {'label': 'Nelder-Mead', 'value': 'Nelder-Mead'},
                    {'label': 'Powell', 'value': 'Powell'},
                    {'label': 'CG', 'value': 'CG'},
                    {'label': 'BFGS', 'value': 'BFGS'},
                    {'label': 'Newton-CG', 'value': 'Newton-CG'},
                    {'label': 'TNC', 'value': 'TNC'},
                    {'label': 'trust-constr', 'value': 'trust-constr'},
                    {'label': 'dogleg', 'value': 'dogleg'},
                    {'label': 'trust-ncg', 'value': 'trust-ncg'},
                    {'label': 'trust-exact', 'value': 'trust-exact'},
                    {'label': 'trust-krylov', 'value': 'trust-krylov'}
                ],
                value='trust-constr',
                style={'width': '395px'}
            ),
        ], style={'margin': '20px'}),
        html.Div([html.Button('Run Optimization', id='run-optimization-btn', n_clicks=0)], style={'margin': '20px'}),
        #html.Div(id='result-container'),
        #html.Br(),

        html.Div(children=[
        html.Div(id="output_algorithm"),
        html.Br(),
        #html.Div(id="output_min_investment_per_Stock"),
        #html.Div(id="output_max_nr_customers"),
        #html.Div(id="output_weight_retention"),
        html.Div(id="output_optimal_price"),
        html.Br(),
        html.Div(id="output_optimal_quantity"),
        html.Br(),
        html.Div(id="output_optimal_marketing"),
        html.Br(),
        html.Div(id="output_optimal_profit"),
        ], style={'padding': 1, 'flex': 1,'margin': '20px'}),
        # html.Div([dcc.Graph(id='profit-surface', style={'height': '70vh'})], style={'margin': '20px'}),

        # html.Div([
        # dcc.Graph(id='convergence-plot')
        # ], style={'margin': '20px'})


        dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='profit-surface', style={'height': '70vh'})
            ])
        ], width=6),
        dbc.Col([
            html.Div([
                dcc.Graph(id='iteration-trace-plot', style={'height': '70vh'})
            ])
        ], width=6)
        ]),
        dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='convergence-plot', style={'height': '70vh'})
            ])
        ], width=12)
        ]),

        ]), # end of 2nd tab

        # start of 3rd tab
        dcc.Tab(label='Data Ingestion and Simulating Multiple Products', children=[ 
        html.Div([
        html.Br(),
        html.H3("Upload your data:",style={"color": "#1c3e4d", "font-weight": "bold"}),
        html.Br(),
        dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Excel File'),
        multiple=False
        ),
        html.Br(),
        html.Div(id='table-container'),
        html.Br(),
        html.Button('Run Optimization', id='run-optimization-btn2', n_clicks=0),
        
        html.Br(),
        html.Hr(),  # Add a horizontal line
        html.H3("Optimal Values per Product",style={"color": "#1c3e4d", "font-weight": "bold"}),
        html.Br(),
        html.Div(id='optimal-values-table'),
        html.Br(),
        html.Div(id='content-container'),
        ], style={'margin': '20px'}),

        ]), # end of 3rd tab

        # start of 4th tab
        dcc.Tab(label='Labor-Augmented Model', children=[
        html.Br(),
        html.H3("Profit animations, holding inputs constant",style={"color": "#1c3e4d", "font-weight": "bold", 'margin': '20px'}),
        html.Br(),
        html.Div([html.Button('Run Optimization', id='run-optimization-btn3', n_clicks=0)], style={'margin': '20px'}),
        html.Br(),
        html.Div(children=[
        html.Div(id="output_optimal_price2"),
        html.Br(),
        #html.Div(id="output_min_investment_per_Stock"),
        #html.Div(id="output_max_nr_customers"),
        #html.Div(id="output_weight_retention"),
        html.Div(id="output_optimal_marketing2"),
        html.Br(),
        html.Div(id="output_optimal_labor"),
        html.Br(),
        html.Div(id="output_optimal_quantity2"),
        html.Br(),
        html.Div(id="output_optimal_profit2"),
        ], style={'padding': 1, 'flex': 1,'margin': '20px'}),
        html.Br(),
        dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='profit-animation-holding-prices-constant', style={'height': '70vh'})
            ])
        ], width=4),
        dbc.Col([
            html.Div([
                dcc.Graph(id='profit-animation-holding-labor-constant', style={'height': '70vh'})
            ])
        ], width=4),
        dbc.Col([
            html.Div([
                dcc.Graph(id='profit-animation-holding-marketing-constant', style={'height': '70vh'})
            ])
        ], width=4)
        ]),


        ]), # end of 4th tab
    ]),

])

# FUNCTIONS FOR THE LABOR-AUGMENTED MODEL

# Production function
def production_function(labor_input, productivity=5):
    return labor_input * productivity

# Labor cost function
def labor_cost_function(labor_input, wage_rate=0.4):
    return labor_input * wage_rate

# Demand function
def demand_function_v1(price, marketing_expenditure, constant_demand, beta_p, beta_m, marketing_multiplier):
    return constant_demand - beta_p * price**2 + (marketing_multiplier * marketing_expenditure - beta_m * marketing_expenditure**2)

# Function to calculate profit, quantity, and demand at a given price with a demand function
def calculate_profit_quantity_demand(price_marketing_labor, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, labor_cost_function, production_function):
    price, marketing_expenditure, labor_input = price_marketing_labor
    demand = constant_demand - beta_p * price**2
    adjusted_demand = demand + (marketing_multiplier * marketing_expenditure - beta_m * marketing_expenditure**2)
    quantity = production_function(labor_input)  # Update quantity calculation using production function
    if quantity < 0:
        penalty = abs(quantity) * (price + 1) * 1000  # Penalty for negative quantity
    else:
        penalty = 0
    revenue = price * quantity
    labor_cost = labor_cost_function(labor_input)  # Calculate labor cost
    cost = variable_cost * quantity + fixed_cost + marketing_expenditure + labor_cost  # Update total cost calculation
    profit = revenue - cost - penalty
    return -profit  # Returning the negative profit to be minimized


# Function to find the optimal price, marketing expenditure, and labor input
def find_optimal_price_and_marketing(variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, labor_cost_function, production_function, optimization_algorithm, initial_state_price, initial_state_marketing, initial_state_labor, max_marketing_expenditure):
    

    def constraint_function(x):
        price, marketing_expenditure, labor_input = x
        quantity_produced = production_function(labor_input)
        demand = constant_demand - beta_p * price**2 + (marketing_multiplier * marketing_expenditure - beta_m * marketing_expenditure**2)
        return np.array([
            max_marketing_expenditure - marketing_expenditure,  # Marketing expenditure constraint
            price,  # Price >= 0 constraint
            marketing_expenditure,  # Marketing expenditure >= 0 constraint
            labor_input,  # Labor input >= 0 constraint
            demand - quantity_produced  # Quantity produced should equal demand constraint
        ])

    # Define the constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},  # Price >= 0 constraint
        {'type': 'ineq', 'fun': lambda x: x[1]},  # Marketing expenditure >= 0 constraint
        {'type': 'ineq', 'fun': lambda x: x[2]},  # Labor input >= 0 constraint
        {'type': 'ineq', 'fun': lambda x: max_marketing_expenditure - x[1]},  # Marketing expenditure constraint
        {'type': 'eq', 'fun': lambda x: demand_function_v1(x[0], x[1], constant_demand, beta_p, beta_m, marketing_multiplier) - production_function(x[2])}  # Quantity produced should equal demand constraint
    ]

    initial_state_labor =1 
    result = minimize(calculate_profit_quantity_demand, x0=[initial_state_price, initial_state_marketing, initial_state_labor],
                      args=(variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, labor_cost_function, production_function),
                      method=optimization_algorithm,
                      constraints=constraints)
    optimal_price, optimal_marketing, optimal_labor = result.x
    optimal_quantity = production_function(optimal_labor)
    optimal_profit = -result.fun  # Extracting the negative profit from the result
    return optimal_price, optimal_marketing, optimal_labor, optimal_quantity, optimal_profit



# FUNCTIONS FOR THE BASELINE MODEL

# Function to calculate profit, quantity, and demand at a given price with a demand function
def calculate_profit_quantity_demand_m(price_and_marketing, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier):
    price, marketing_expenditure = price_and_marketing
    demand = constant_demand - beta_p * price**2
    adjusted_demand = demand + (marketing_multiplier * marketing_expenditure - beta_m * marketing_expenditure**2)
    quantity = adjusted_demand  # Simplified assumption for quantity calculation
    # next generate a penalty term to penalize negative quantity results, as we can't add quantity as a direct constraint to be non-negative
    if quantity < 0:
        penalty = abs(quantity)*(price+1)*1000 # where the last term is the penalty factor
    else:
        penalty = 0
    revenue = price * quantity
    cost = variable_cost * quantity + fixed_cost + marketing_expenditure
    profit = revenue - cost - penalty
    return -profit  # Returning the negative profit to be minimized

# Function to find the optimal price and marketing expenditure
def find_optimal_price_and_marketing_m(variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, optimization_algorithm, initial_state_price, initial_state_marketing, max_marketing_expenditure):
    
    
    # Define the constraint function
    def constraint_function2(x):
        price, marketing_expenditure = x
        return np.array([
            max_marketing_expenditure - marketing_expenditure,  # Marketing expenditure constraint
            price,  # Price >= 0 constraint
            marketing_expenditure  # Marketing expenditure >= 0 constraint
        ])


    # Define the constraint
    constraint2 = {'type': 'ineq', 'fun': constraint_function2}

    result = minimize(calculate_profit_quantity_demand_m, x0=[initial_state_price, initial_state_marketing],
                      args=(variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier),
                      method=optimization_algorithm,
                      constraints=[constraint2])
    optimal_price, optimal_marketing = result.x
    optimal_quantity = constant_demand - beta_p * optimal_price**2 + (marketing_multiplier * optimal_marketing - beta_m * optimal_marketing**2)
    optimal_profit = -result.fun  # Extracting the negative profit from the result
    return optimal_price, optimal_marketing, optimal_quantity, optimal_profit


@app.callback(
    Output('output_algorithm', 'children'),
    [
        Input('run-optimization-btn', 'n_clicks'),
    ],
    [
        State('variable-cost', 'value'),
        State('fixed-cost', 'value'),
        State('constant-demand', 'value'),
        State('beta-p', 'value'),
        State('beta-m', 'value'),
        State('marketing-multiplier', 'value'),
        State('optimization-algorithm', 'value')
    ]
)
def update_info_texts(n_clicks, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, optimization_algorithm):
    if n_clicks > 0:
        if optimization_algorithm is not None:
            optimization_algorithm = 'The currently running algorithm is: \n{}.'.format(optimization_algorithm)
        
        if variable_cost is not None:
            variable_cost = 'The upper limit on number of selected customers is set at \n{}.'.format(variable_cost)
        
        return optimization_algorithm
    else:
        None

# Callback to generate profit surface plot
@app.callback(
    Output('profit-surface', 'figure'),
    Output('output_optimal_price', 'children'),
    Output('output_optimal_quantity', 'children'),
    Output('output_optimal_marketing', 'children'),
    Output('output_optimal_profit', 'children'),
    Output('iteration-trace-plot', 'figure'),
    [Input('run-optimization-btn', 'n_clicks')],
    [State('variable-cost', 'value'),
     State('fixed-cost', 'value'),
     State('constant-demand', 'value'),
     State('beta-p', 'value'),
     State('beta-m', 'value'),
     State('marketing-multiplier', 'value'),
     State('optimization-algorithm', 'value'),
     State('linspace-start', 'value'),
     State('linspace-end', 'value'),
     State('linspace-num', 'value'),
     State('initial-state-price', 'value'),
     State('initial-state-marketing', 'value'),
     State('max-marketing-expenditure', 'value')]
)
def generate_profit_surface(n_clicks, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, optimization_algorithm, linspace_start, linspace_end, linspace_num, initial_state_price, initial_state_marketing, max_marketing_expenditure):
    if n_clicks > 0:
        demand_function_params = (constant_demand, beta_p, beta_m, marketing_multiplier)
        # Default optimization algorithm
        # optimization_algorithm = 'L-BFGS-B'
        optimal_price, optimal_marketing, optimal_quantity, optimal_profit = find_optimal_price_and_marketing_m(
            variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, optimization_algorithm,
            initial_state_price, initial_state_marketing, max_marketing_expenditure)

        # Generate profit surface plot
        prices = np.linspace(linspace_start, linspace_end, linspace_num)
        marketing_expenditures = np.linspace(linspace_start, linspace_end, linspace_num)

        Z = np.array([[-calculate_profit_quantity_demand_m([p, m], variable_cost, fixed_cost, constant_demand, beta_p,
                                                          beta_m, marketing_multiplier) for p in prices] for m in
                      marketing_expenditures])
        X, Y = np.meshgrid(prices, marketing_expenditures)

        fig = go.Figure()

        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, name='Profit Surface')
        )

        fig.add_trace(
            go.Scatter3d(x=[initial_state_price], y=[initial_state_marketing], z=[-calculate_profit_quantity_demand_m([initial_state_price, initial_state_marketing], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)],
                         mode='markers', marker=dict(size=5, color='green'), name='Initial Point')
        )

        fig.add_trace(
            go.Scatter3d(x=[optimal_price], y=[optimal_marketing], z=[optimal_profit],
                         mode='markers', marker=dict(size=5, color='red'), name='Optimal Point')
        )

        # Add iteration points
        for iteration_point in optimization_history:
            price, marketing_expenditure = iteration_point
            profit = -calculate_profit_quantity_demand_m(iteration_point, variable_cost, fixed_cost, constant_demand,
                                                       beta_p, beta_m, marketing_multiplier)
            fig.add_trace(
                go.Scatter3d(x=[price], y=[marketing_expenditure], z=[profit],
                             mode='markers', marker=dict(size=3, color='blue'), name='Iteration Point', showlegend=False)
            )


        # Generate traces for arrows showing the path of iteration points
        iteration_trace = go.Scatter3d(
            x=[point[0] for point in optimization_history],
            y=[point[1] for point in optimization_history],
            z=[-calculate_profit_quantity_demand_m(point, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier) for point in optimization_history],
            mode='lines+markers',
            line=dict(color='blue', width=5),
            marker=dict(size=6, color='blue'),
            name='Iteration Path'
        )
        
        initial_guess_trace = go.Scatter3d(
            x=[initial_state_price],
            y=[initial_state_marketing],
            z=[-calculate_profit_quantity_demand_m([initial_state_price, initial_state_marketing], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)],
            mode='markers',
            marker=dict(size=5, color='green'),
            name='Initial Guess'
        )

        # Add iteration path traces to the figure
        fig.add_trace(iteration_trace)
        fig.add_trace(initial_guess_trace)


        fig.update_layout(
            title='Profit Surface',
            scene=dict(aspectmode="cube"),
            scene_xaxis_title='Price',
            scene_yaxis_title='Marketing Expenditure',
            scene_zaxis_title='Profit',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        if optimal_price is not None:
            optimal_price = 'Optimal price:                 \n{:.2f}.'.format(optimal_price)
        if optimal_quantity is not None:
            optimal_quantity = 'Optimal quantity:              \n{:.2f}.'.format(optimal_quantity)
        if optimal_marketing is not None:
            optimal_marketing = 'Optimal marketing expenditure: \n{:.2f}.'.format(optimal_marketing)
        if optimal_profit is not None:
            optimal_profit = 'Maximized profit:              \n{:.2f}.'.format(optimal_profit)



        # create the trace plot of the solution iterations

        # Retrieve X, Y, and Z axes ranges from the surface plot
        # x_range = fig['layout']['scene']['xaxis']['range']
        # y_range = fig['layout']['scene']['yaxis']['range']
        # z_range = fig['layout']['scene']['zaxis']['range']
        x_range = [min(prices), max(prices)]
        y_range = [min(marketing_expenditures), max(marketing_expenditures)]
        z_range = [np.min(Z), np.max(Z)]
       

        iteration_trace_plot = go.Figure()
        iteration_trace_plot.add_trace(
            go.Scatter3d(x=[initial_state_price], y=[initial_state_marketing], z=[-calculate_profit_quantity_demand_m([initial_state_price, initial_state_marketing], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)],
                         mode='markers', marker=dict(size=5, color='green'), name='Initial Point')
        )

        iteration_trace_plot.add_trace(
            go.Scatter3d(x=[optimal_price], y=[optimal_marketing], z=[optimal_profit],
                         mode='markers', marker=dict(size=5, color='red'), name='Optimal Point')
        )

        # Add iteration points
        for iteration_point in optimization_history:
            price, marketing_expenditure = iteration_point
            profit = -calculate_profit_quantity_demand_m(iteration_point, variable_cost, fixed_cost, constant_demand,
                                                       beta_p, beta_m, marketing_multiplier)
            iteration_trace_plot.add_trace(
                go.Scatter3d(x=[price], y=[marketing_expenditure], z=[profit],
                             mode='markers', marker=dict(size=3, color='blue'), name='Iteration Point', showlegend=False)
            )


        # Add iteration path traces to the figure
        iteration_trace_plot.add_trace(iteration_trace)
        iteration_trace_plot.add_trace(initial_guess_trace)

        # Set the ranges of X, Y, and Z axes to match the surface plot
        iteration_trace_plot.update_layout(
            scene=dict(
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
                aspectmode="cube"
            )
        )
        iteration_trace_plot.update_layout(
            title='Optimization iteration trace',
            scene=dict(aspectmode="cube"),
            scene_xaxis_title='Price',
            scene_yaxis_title='Marketing Expenditure',
            scene_zaxis_title='Profit',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )


        return fig, optimal_price, optimal_quantity, optimal_marketing, optimal_profit, iteration_trace_plot
    else:
        return go.Figure(), None, None, None, None, go.Figure()


@app.callback(
    Output('convergence-plot', 'figure'),
    [Input('run-optimization-btn', 'n_clicks')],
    [State('variable-cost', 'value'),
     State('fixed-cost', 'value'),
     State('constant-demand', 'value'),
     State('beta-p', 'value'),
     State('beta-m', 'value'),
     State('marketing-multiplier', 'value'),
     State('optimization-algorithm', 'value'),
     State('linspace-start', 'value'),
     State('linspace-end', 'value'),
     State('linspace-num', 'value'),
     State('initial-state-price', 'value'),
     State('initial-state-marketing', 'value'),
     State('max-marketing-expenditure', 'value')]
)
def update_convergence_plot(n_clicks, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, optimization_algorithm, linspace_start, linspace_end, linspace_num, initial_state_price, initial_state_marketing, max_marketing_expenditure):
    
    global optimization_history  # Use the global optimization history variable

    if n_clicks > 0:
        def objective_function(price_and_marketing):
            return calculate_profit_quantity_demand_m(price_and_marketing, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)

        initial_guess = [initial_state_price, initial_state_marketing]  # Initial guess for price and marketing expenditure
        optimization_history = []  # Clear previous optimization history
        result = minimize(objective_function, initial_guess, method=optimization_algorithm, callback=record_optimization_history)

        # Extract optimization history
        iterations = list(range(len(optimization_history)))
        objective_values = [-calculate_profit_quantity_demand_m(x, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier) for x in optimization_history]

        # Create convergence plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=objective_values, mode='lines+markers', name='Objective Function'))
        fig.update_layout(title='Optimization Convergence',
                          xaxis_title='Iteration',
                          yaxis_title='Objective Function Value',
                          )
        return fig
    else:
        return go.Figure()

# Callback function to record optimization history
def record_optimization_history(*args):
    if len(args) == 1:
        x = args[0]
    elif len(args) == 2:
        x, _ = args
    else:
        raise ValueError("Unexpected number of arguments in record_optimization_history")

    optimization_history.append(x)















# PRICE AND MARKETING SIMULATION TAB

# Start by defining the necessary functions, note that these are simplified versions of the individual product simulation functions above

# Function to calculate profit, quantity, and demand at a given price with a demand function
def calculate_profit_quantity_demand2(price_and_marketing, variable_cost, fixed_cost, demand_function_params):
    price, marketing_expenditure = price_and_marketing
    demand_function_product = lambda price, marketing_expenditure: demand_function2(price, marketing_expenditure, *demand_function_params)
    quantity, _ = demand_function_product(price, marketing_expenditure)
    revenue = price * quantity
    cost = variable_cost * quantity + fixed_cost + marketing_expenditure
    profit = revenue - cost
    return -profit  # Returning the negative profit to be minimized

# Demand function for each product
def demand_function2(price, marketing_expenditure, constant_demand, beta_p, beta_m, marketing_multiplier):
    demand = constant_demand - beta_p * price**2
    adjusted_demand = demand + (marketing_multiplier * marketing_expenditure - beta_m * marketing_expenditure**2)
    return adjusted_demand, demand

# Demand function, returning only adjusted demand
def demand_function(price, marketing_expenditure, constant_demand, beta_p, beta_m, marketing_multiplier):
    return constant_demand - beta_p * price**2 + (marketing_multiplier * marketing_expenditure - beta_m * marketing_expenditure**2)

# Function to find the optimal price, marketing expenditure, quantity, and profit for a given product
def find_optimal_price_and_marketing2(product_data, price_range=(0, 500), marketing_range=(0, 500)):
    variable_cost = product_data['VariableCost']
    fixed_cost = product_data['FixedCost']
    demand_function_params = (
        product_data['Constant_Demand'],
        product_data['Beta_p'],
        product_data['Beta_m'],
        product_data['MarketingMultiplier']
    )

    result = minimize(calculate_profit_quantity_demand2, x0=[50, 25],
                      args=(variable_cost, fixed_cost, demand_function_params),
                      bounds=[price_range, marketing_range], method='L-BFGS-B')
    optimal_price, optimal_marketing = result.x
    optimal_quantity, _ = demand_function2(optimal_price, optimal_marketing, *demand_function_params)
    optimal_profit = -result.fun  # Extracting the negative profit from the result
    return optimal_price, optimal_marketing, optimal_quantity, optimal_profit

# Function to calculate profit for the labor-augmented model
def calculate_profit(price_labor_marketing, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier):
    price, labor_input, marketing_expenditure = price_labor_marketing
    demand = demand_function(price, marketing_expenditure, constant_demand, beta_p, beta_m, marketing_multiplier)
    quantity = min(production_function(labor_input), demand)
    revenue = price * quantity
    labor_cost = labor_cost_function(labor_input)
    total_cost = variable_cost * quantity + fixed_cost + labor_cost + marketing_expenditure
    profit = revenue - total_cost
    return -profit  # Negative because we want to maximize profit


# Callback to run optimization when the button is clicked
@app.callback(
    Output('content-container', 'children'),
    Output('optimal-values-table', 'children'),
    [Input('run-optimization-btn2', 'n_clicks')],
    [State('table', 'data'),

     ]
)
def run_optimization(n_clicks, data):
    if n_clicks > 0:
        df = pd.DataFrame(data)
        return generate_product_layout(df)

# Modify the existing update_table callback to make the table editable
@app.callback(
    Output('table-container', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_table(contents, filename):
    if contents is None:
        return []

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_excel(BytesIO(decoded))

    return dash_table.DataTable(
        id='table',
        columns=[{'name': col, 'id': col, 'editable': True} for col in df.columns],
        data=df.to_dict('records'),
        export_format='csv',  # Default export format
        export_headers='display',  # Include headers in the exported file
        merge_duplicate_headers=True,  # Merge duplicate headers if they exist
        style_table={'overflowX': 'auto'},  # Enable horizontal scrolling
        style_data_conditional=[  # Conditional styling for odd and even rows
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'row_index': 'even'},
                'backgroundColor': 'white'
            }
        ]
    )

def generate_product_layout(df):
    product_layouts = []
    product_results = []

    # Data Type Conversion
    df['VariableCost'] = df['VariableCost'].astype(float)
    df['FixedCost'] = df['FixedCost'].astype(float)
    df['Constant_Demand'] = df['Constant_Demand'].astype(float)
    df['Beta_p'] = df['Beta_p'].astype(float)
    df['Beta_m'] = df['Beta_m'].astype(float)
    df['MarketingMultiplier'] = df['MarketingMultiplier'].astype(float)

    for i, product_data in enumerate(df.iterrows()):
        product_data = product_data[1]
        result = find_optimal_price_and_marketing2(product_data)

        # Extract data from result
        optimal_price, optimal_marketing, optimal_quantity, optimal_profit = result

        demand_function_params = (
            product_data['Constant_Demand'],
            product_data['Beta_p'],
            product_data['Beta_m'],
            product_data['MarketingMultiplier']
        )

        product_results.append({
            'Product': f'Product {i+1}',
            'Optimal Price': optimal_price,
            'Optimal Marketing': optimal_marketing,
            'Optimal Quantity': optimal_quantity,
            'Optimal Profit': optimal_profit
        })

        # Plot surface for each product
        prices = np.linspace(1, 50, 50)
        marketing_expenditures = np.linspace(0, 50, 50)

        Z = np.array([[-calculate_profit_quantity_demand2([p, m], product_data['VariableCost'], product_data['FixedCost'], demand_function_params) for p in prices] for m in marketing_expenditures])
        X, Y = np.meshgrid(prices, marketing_expenditures)

        fig = go.Figure()

        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, name=f'Product {i+1} Profit')
        )

        fig.add_trace(
            go.Scatter3d(x=[optimal_price], y=[optimal_marketing], z=[optimal_profit],
                         mode='markers', marker=dict(size=5, color='red'), name='Optimal Point')
        )

        fig.update_layout(
            title=f'Profit Surface for Product {i+1}',
            scene=dict(aspectmode="cube"),
            scene_xaxis_title='Price',
            scene_yaxis_title='Marketing Expenditure',
            scene_zaxis_title='Profit'
        )

        # Plot demand functions
        demand_fig = px.scatter(x=prices, y=demand_function2(prices, 0, *demand_function_params)[0], title=f'Demand Function for Product {i+1}')
        demand_fig.update_layout(showlegend=False)

        product_layouts.append(
            html.Div([
                html.Div(dcc.Graph(figure=fig), style={'width': '60%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(figure=demand_fig), style={'width': '40%', 'display': 'inline-block'}),
            ]),
        )
        # Add a break between product rows
        product_layouts.append(html.Br())

    # Create a table for displaying optimal values per product
    table_columns = [
        {'name': col, 'id': col} for col in product_results[0].keys()
    ]
    # optimal_values_table = dash_table.DataTable(
    #     id='optimal-values-table',
    #     columns=table_columns,
    #     data=product_results
    # )
    optimal_values_table = dash_table.DataTable(
    id='optimal-values-table',
    columns=table_columns,
    data=product_results,
    export_format='csv',  # Default export format
    export_headers='display',  # Include headers in the exported file
    merge_duplicate_headers=True,  # Merge duplicate headers if they exist
    style_table={'overflowX': 'auto'},  # Enable horizontal scrolling
    style_data_conditional=[  # Conditional styling for odd and even rows
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        },
        {
            'if': {'row_index': 'even'},
            'backgroundColor': 'white'
        }
    ]
    )
    # product_layouts.append(html.Hr())  # Add a horizontal line
    # product_layouts.append(html.H3("Optimal Values per Product"))
    # product_layouts.append(optimal_values_table)

    return product_layouts, optimal_values_table






# LABOR-AUGMENTED MODEL SIMULATION

# Callback to generate profit surface plot
@app.callback(
    Output('profit-animation-holding-prices-constant', 'figure'),
    Output('profit-animation-holding-labor-constant', 'figure'),
    Output('profit-animation-holding-marketing-constant', 'figure'),
    Output('output_optimal_price2', 'children'),
    Output('output_optimal_marketing2', 'children'),
    Output('output_optimal_labor', 'children'),
    Output('output_optimal_quantity2', 'children'),
    Output('output_optimal_profit2', 'children'),
    [Input('run-optimization-btn3', 'n_clicks')],
    [State('variable-cost', 'value'),
     State('fixed-cost', 'value'),
     State('constant-demand', 'value'),
     State('beta-p', 'value'),
     State('beta-m', 'value'),
     State('marketing-multiplier', 'value'),
     State('optimization-algorithm', 'value'),
     State('linspace-start', 'value'),
     State('linspace-end', 'value'),
     State('linspace-num', 'value'),
     State('initial-state-price', 'value'),
     State('initial-state-marketing', 'value'),
     State('max-marketing-expenditure', 'value')]
)
def generate_profit_surface2(n_clicks, variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, optimization_algorithm, linspace_start, linspace_end, linspace_num, initial_state_price, initial_state_marketing, max_marketing_expenditure):
    
    if n_clicks > 0:
        # Parameters
        variable_cost = 2
        fixed_cost = 500
        constant_demand = 250
        beta_p = 0.1
        beta_m = 0.1
        marketing_multiplier = 5
        initial_state_labor = 1
        optimization_algorithm = 'trust-constr'
        initial_state_price = 1
        initial_state_marketing = 1
        initial_state_labor = 1
        max_marketing_expenditure = 51
        optimal_price, optimal_marketing, optimal_labor, optimal_quantity, optimal_profit = find_optimal_price_and_marketing(variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier, labor_cost_function, production_function, optimization_algorithm, initial_state_price, initial_state_marketing, initial_state_labor, max_marketing_expenditure)
        
        
        # CREATING THE PRICES-CONSTANT FIGURE
        # Number of frames for the animation
        num_frames = 40

        # Create data for animation frames
        frames = []
        for k in range(num_frames):
            price = k * 50 / (num_frames - 1)  # Vary price from 0 to 50
            
            Z = np.zeros((50, 50))
            for i, labor_input in enumerate(np.linspace(0, 50, 50)):
                for j, marketing_expenditure in enumerate(np.linspace(0, 50, 50)):
                    Z[j, i] = -calculate_profit([price, labor_input, marketing_expenditure], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)
            
            frames.append(go.Frame(data=[go.Surface(z=Z, x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50))]))

        # Create the initial profit surface
        Z = np.zeros((50, 50))
        for i, labor_input in enumerate(np.linspace(0, 50, 50)):
            for j, marketing_expenditure in enumerate(np.linspace(0, 50, 50)):
                Z[j, i] = -calculate_profit([optimal_price, labor_input, marketing_expenditure], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)


        # Create the animation figure
        fig = go.Figure(frames=frames)
        fig.update_layout(title='Profit Surface Animation with Varying Prices',
                        scene=dict(xaxis_title='Labor Input', yaxis_title='Marketing Expenditure', zaxis_title='Profit'),
                        updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])])

        # Add initial frame
        #fig.add_trace(go.Surface(z=np.zeros((50, 50)), x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50), visible=True))
        fig.add_trace(go.Surface(z=Z, x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50)))



        # CREATE THE FIGURE FOR LABOR-CONSTANT VISUALIZATION
        # Create data for animation frames
        frames = []
        for k in range(num_frames):
            marketing_expenditure = k * 50 / (num_frames - 1)  # Vary marketing expenditure from 0 to 50
            Z = np.zeros((50, 50))
            for i, price in enumerate(np.linspace(0, 50, 50)):
                for j, labor_input in enumerate(np.linspace(0, 50, 50)):
                    Z[j, i] = -calculate_profit([price, labor_input, marketing_expenditure], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)
            frames.append(go.Frame(data=[go.Surface(z=Z, x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50))]))

        # Create the animation figure
        fig2 = go.Figure(frames=frames)
        fig2.update_layout(title='Profit Surface Animation with Varying Marketing Expenditure',
                        scene=dict(xaxis_title='Prices', yaxis_title='Labor', zaxis_title='Profit'),
                        updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])])

        # Create the initial profit surface
        Z = np.zeros((50, 50))
        for i, price_input in enumerate(np.linspace(0, 50, 50)):
            for j, marketing_expenditure in enumerate(np.linspace(0, 50, 50)):
                Z[j, i] = -calculate_profit([price_input, optimal_labor, marketing_expenditure], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)

        # Add initial frame
        #fig2.add_trace(go.Surface(z=np.zeros((50, 50)), x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50), visible=True))
        fig2.add_trace(go.Surface(z=Z, x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50)))


        # CREATE FIGURE THAT HOLDS MARKETING CONSTANT
        # Create data for animation frames
        frames = []
        for k in range(num_frames):
            labor_input = k * 50 / (num_frames - 1)  # Vary labor input from 0 to 50
            
            Z = np.zeros((50, 50))
            for i, price in enumerate(np.linspace(0, 50, 50)):
                for j, marketing_expenditure in enumerate(np.linspace(0, 50, 50)):
                    Z[j, i] = -calculate_profit([price, labor_input, marketing_expenditure], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)
            
            frames.append(go.Frame(data=[go.Surface(z=Z, x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50))]))

        # Create the animation figure
        fig3 = go.Figure(frames=frames)
        fig3.update_layout(title='Profit Surface Animation with Varying Labor Input',
                        scene=dict(xaxis_title='Prices', yaxis_title='Marketing Expenditure', zaxis_title='Profit'),
                        updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])])


        # Create the initial profit surface
        Z = np.zeros((50, 50))
        for i, price_input in enumerate(np.linspace(0, 50, 50)):
            for j, labor_input in enumerate(np.linspace(0, 50, 50)):
                Z[j, i] = -calculate_profit([price_input, labor_input, optimal_marketing], variable_cost, fixed_cost, constant_demand, beta_p, beta_m, marketing_multiplier)

        # Add initial frame
        #fig3.add_trace(go.Surface(z=np.zeros((50, 50)), x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50), visible=True))
        fig3.add_trace(go.Surface(z=Z, x=np.linspace(0, 50, 50), y=np.linspace(0, 50, 50)))



        if optimal_price is not None:
            optimal_price = 'Optimal price:                 \n{:.2f}.'.format(optimal_price)
        if optimal_quantity is not None:
            optimal_quantity = 'Optimal quantity:              \n{:.2f}.'.format(optimal_quantity)
        if optimal_marketing is not None:
            optimal_marketing = 'Optimal marketing expenditure: \n{:.2f}.'.format(optimal_marketing)
        if optimal_profit is not None:
            optimal_profit = 'Maximized profit:              \n{:.2f}.'.format(optimal_profit)
        if optimal_labor is not None:
            optimal_labor = 'Optimal labor:              \n{:.2f}.'.format(optimal_labor)


        return fig, fig2, fig3, optimal_price, optimal_marketing, optimal_labor, optimal_quantity, optimal_profit
    else:
        return go.Figure(), go.Figure(), go.Figure(), None, None, None, None, None



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
