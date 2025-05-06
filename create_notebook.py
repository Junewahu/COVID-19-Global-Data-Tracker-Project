import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# COVID-19 Global Data Analysis\n",
                "\n",
                "This notebook provides a comprehensive analysis of global COVID-19 data, including interactive features for custom analysis, advanced visualizations, and detailed insights."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup and Data Loading"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "from datetime import datetime, timedelta\n",
                "from ipywidgets import interact, widgets, Layout\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set plot style\n",
                "plt.style.use('seaborn')\n",
                "sns.set_palette('husl')\n",
                "\n",
                "# Display settings\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.max_rows', 100)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Loading and Initial Exploration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the dataset\n",
                "df = pd.read_csv('../data/owid-covid-data.csv')\n",
                "\n",
                "# Display basic information\n",
                "print(\"Dataset Shape:\", df.shape)\n",
                "print(\"\\nColumns:\")\n",
                "print(df.columns.tolist())\n",
                "print(\"\\nFirst few rows:\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Cleaning and Preprocessing"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Convert date column to datetime\n",
                "df['date'] = pd.to_datetime(df['date'])\n",
                "\n",
                "# Check for missing values\n",
                "missing_values = df.isnull().sum()\n",
                "print(\"Missing values per column:\")\n",
                "print(missing_values[missing_values > 0])\n",
                "\n",
                "# Handle missing values\n",
                "numeric_columns = df.select_dtypes(include=[np.number]).columns\n",
                "df[numeric_columns] = df[numeric_columns].fillna(0)\n",
                "\n",
                "# Calculate additional metrics\n",
                "df['death_rate'] = (df['total_deaths'] / df['total_cases'] * 100).round(2)\n",
                "df['vaccination_rate'] = (df['total_vaccinations'] / df['population'] * 100).round(2)\n",
                "df['cases_per_million'] = (df['total_cases'] / df['population'] * 1000000).round(2)\n",
                "df['deaths_per_million'] = (df['total_deaths'] / df['population'] * 1000000).round(2)\n",
                "df['recovery_rate'] = (df['total_cases'] - df['total_deaths']) / df['total_cases'] * 100\n",
                "\n",
                "# Calculate rolling averages\n",
                "df['new_cases_7d_avg'] = df.groupby('location')['new_cases'].transform(lambda x: x.rolling(7, min_periods=1).mean())\n",
                "df['new_deaths_7d_avg'] = df.groupby('location')['new_deaths'].transform(lambda x: x.rolling(7, min_periods=1).mean())\n",
                "\n",
                "# Filter out non-country entities\n",
                "excluded_locations = ['World', 'European Union', 'International']\n",
                "df = df[~df['location'].isin(excluded_locations)]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Interactive Country Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def analyze_country(country, metric='total_cases', date_range=None):\n",
                "    country_data = df[df['location'] == country].copy()\n",
                "    \n",
                "    if date_range:\n",
                "        country_data = country_data[(country_data['date'] >= date_range[0]) & \n",
                "                                   (country_data['date'] <= date_range[1])]\n",
                "    \n",
                "    # Create figure with secondary y-axis\n",
                "    fig = go.Figure()\n",
                "    \n",
                "    # Add main metric\n",
                "    fig.add_trace(go.Scatter(x=country_data['date'], y=country_data[metric],\n",
                "                             name=metric.replace('_', ' ').title(),\n",
                "                             mode='lines'))\n",
                "    \n",
                "    # Add 7-day moving average if applicable\n",
                "    if 'new_' in metric:\n",
                "        avg_metric = metric.replace('new_', 'new_') + '_7d_avg'\n",
                "        fig.add_trace(go.Scatter(x=country_data['date'], y=country_data[avg_metric],\n",
                "                                 name='7-day Moving Average',\n",
                "                                 mode='lines',\n",
                "                                 line=dict(dash='dash')))\n",
                "    \n",
                "    fig.update_layout(title=f'{metric.replace(\"_\", \" \").title()} in {country}',\n",
                "                      xaxis_title='Date',\n",
                "                      yaxis_title=metric.replace('_', ' ').title(),\n",
                "                      hovermode='x unified')\n",
                "    fig.show()\n",
                "\n",
                "# Create interactive widgets\n",
                "countries = sorted(df['location'].unique())\n",
                "metrics = ['total_cases', 'total_deaths', 'total_vaccinations', 'death_rate', \n",
                "          'vaccination_rate', 'cases_per_million', 'deaths_per_million', \n",
                "          'new_cases', 'new_deaths', 'recovery_rate']\n",
                "\n",
                "# Date range slider\n",
                "date_range = widgets.DateRangeSlider(\n",
                "    value=[df['date'].min(), df['date'].max()],\n",
                "    min=df['date'].min(),\n",
                "    max=df['date'].max(),\n",
                "    step=timedelta(days=1),\n",
                "    description='Date Range:',\n",
                "    layout=Layout(width='100%')\n",
                ")\n",
                "\n",
                "interact(analyze_country,\n",
                "         country=widgets.Dropdown(options=countries, description='Country:'),\n",
                "         metric=widgets.Dropdown(options=metrics, description='Metric:'),\n",
                "         date_range=date_range)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Global Trends Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate global totals\n",
                "global_data = df.groupby('date').agg({\n",
                "    'total_cases': 'sum',\n",
                "    'total_deaths': 'sum',\n",
                "    'total_vaccinations': 'sum',\n",
                "    'population': 'sum',\n",
                "    'new_cases': 'sum',\n",
                "    'new_deaths': 'sum'\n",
                "}).reset_index()\n",
                "\n",
                "# Calculate global rates\n",
                "global_data['global_death_rate'] = (global_data['total_deaths'] / global_data['total_cases'] * 100).round(2)\n",
                "global_data['global_vaccination_rate'] = (global_data['total_vaccinations'] / global_data['population'] * 100).round(2)\n",
                "global_data['global_cases_per_million'] = (global_data['total_cases'] / global_data['population'] * 1000000).round(2)\n",
                "global_data['global_deaths_per_million'] = (global_data['total_deaths'] / global_data['population'] * 1000000).round(2)\n",
                "\n",
                "# Create subplots\n",
                "fig = make_subplots(rows=2, cols=2,\n",
                "                    subplot_titles=('Total Cases and Deaths', 'Daily New Cases and Deaths',\n",
                "                                   'Global Death Rate', 'Global Vaccination Rate'))\n",
                "\n",
                "# Add traces\n",
                "fig.add_trace(go.Scatter(x=global_data['date'], y=global_data['total_cases'],\n",
                "                         name='Total Cases'), row=1, col=1)\n",
                "fig.add_trace(go.Scatter(x=global_data['date'], y=global_data['total_deaths'],\n",
                "                         name='Total Deaths'), row=1, col=1)\n",
                "\n",
                "fig.add_trace(go.Scatter(x=global_data['date'], y=global_data['new_cases'],\n",
                "                         name='New Cases'), row=1, col=2)\n",
                "fig.add_trace(go.Scatter(x=global_data['date'], y=global_data['new_deaths'],\n",
                "                         name='New Deaths'), row=1, col=2)\n",
                "\n",
                "fig.add_trace(go.Scatter(x=global_data['date'], y=global_data['global_death_rate'],\n",
                "                         name='Death Rate'), row=2, col=1)\n",
                "\n",
                "fig.add_trace(go.Scatter(x=global_data['date'], y=global_data['global_vaccination_rate'],\n",
                "                         name='Vaccination Rate'), row=2, col=2)\n",
                "\n",
                "fig.update_layout(height=800, title_text=\"Global COVID-19 Trends\")\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Regional Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def analyze_region(region, metric='total_cases'):\n",
                "    region_data = df[df['continent'] == region]\n",
                "    \n",
                "    # Calculate regional totals\n",
                "    regional_totals = region_data.groupby('date')[metric].sum().reset_index()\n",
                "    \n",
                "    # Plot regional trend\n",
                "    fig = go.Figure()\n",
                "    fig.add_trace(go.Scatter(x=regional_totals['date'], y=regional_totals[metric],\n",
                "                             name=f'Total {metric.replace(\"_\", \" \").title()}',\n",
                "                             mode='lines'))\n",
                "    \n",
                "    # Add individual country trends\n",
                "    for country in region_data['location'].unique():\n",
                "        country_data = region_data[region_data['location'] == country]\n",
                "        fig.add_trace(go.Scatter(x=country_data['date'], y=country_data[metric],\n",
                "                                 name=country,\n",
                "                                 mode='lines',\n",
                "                                 visible='legendonly'))\n",
                "    \n",
                "    fig.update_layout(title=f'{metric.replace(\"_\", \" \").title()} in {region}',\n",
                "                      xaxis_title='Date',\n",
                "                      yaxis_title=metric.replace('_', ' ').title(),\n",
                "                      hovermode='x unified')\n",
                "    fig.show()\n",
                "\n",
                "# Create interactive widget\n",
                "regions = sorted(df['continent'].unique())\n",
                "interact(analyze_region,\n",
                "         region=widgets.Dropdown(options=regions, description='Region:'),\n",
                "         metric=widgets.Dropdown(options=metrics, description='Metric:'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Vaccination Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get latest vaccination data\n",
                "latest_data = df.sort_values('date').groupby('location').last()\n",
                "vaccination_data = latest_data[['total_vaccinations', 'population', 'vaccination_rate']]\n",
                "\n",
                "# Create vaccination analysis dashboard\n",
                "fig = make_subplots(rows=2, cols=2,\n",
                "                    subplot_titles=('Top 10 Countries by Vaccination Rate',\n",
                "                                   'Vaccination Rate Distribution',\n",
                "                                   'Vaccination Progress by Region',\n",
                "                                   'Vaccination vs Death Rate'))\n",
                "\n",
                "# Top 10 countries\n",
                "top_vaccinated = vaccination_data.nlargest(10, 'vaccination_rate')\n",
                "fig.add_trace(go.Bar(x=top_vaccinated.index, y=top_vaccinated['vaccination_rate'],\n",
                "                     name='Vaccination Rate'), row=1, col=1)\n",
                "\n",
                "# Distribution\n",
                "fig.add_trace(go.Histogram(x=vaccination_data['vaccination_rate'],\n",
                "                          name='Distribution'), row=1, col=2)\n",
                "\n",
                "# Regional progress\n",
                "regional_vaccination = df.groupby(['continent', 'date'])['vaccination_rate'].mean().reset_index()\n",
                "for region in regional_vaccination['continent'].unique():\n",
                "    region_data = regional_vaccination[regional_vaccination['continent'] == region]\n",
                "    fig.add_trace(go.Scatter(x=region_data['date'], y=region_data['vaccination_rate'],\n",
                "                             name=region), row=2, col=1)\n",
                "\n",
                "# Vaccination vs Death Rate\n",
                "fig.add_trace(go.Scatter(x=latest_data['vaccination_rate'],\n",
                "                         y=latest_data['death_rate'],\n",
                "                         mode='markers',\n",
                "                         text=latest_data.index,\n",
                "                         name='Countries'), row=2, col=2)\n",
                "\n",
                "fig.update_layout(height=800, title_text=\"Vaccination Analysis Dashboard\")\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Comparative Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compare_countries(countries, metrics, date_range=None):\n",
                "    country_data = df[df['location'].isin(countries)].copy()\n",
                "    \n",
                "    if date_range:\n",
                "        country_data = country_data[(country_data['date'] >= date_range[0]) & \n",
                "                                   (country_data['date'] <= date_range[1])]\n",
                "    \n",
                "    fig = make_subplots(rows=len(metrics), cols=1,\n",
                "                        subplot_titles=[m.replace('_', ' ').title() for m in metrics])\n",
                "    \n",
                "    for i, metric in enumerate(metrics, 1):\n",
                "        for country in countries:\n",
                "            country_metric = country_data[country_data['location'] == country]\n",
                "            fig.add_trace(go.Scatter(x=country_metric['date'],\n",
                "                                     y=country_metric[metric],\n",
                "                                     name=country),\n",
                "                         row=i, col=1)\n",
                "    \n",
                "    fig.update_layout(height=300*len(metrics),\n",
                "                      title_text=\"Country Comparison Dashboard\",\n",
                "                      showlegend=True)\n",
                "    fig.show()\n",
                "\n",
                "# Create interactive widget for comparison\n",
                "interact(compare_countries,\n",
                "         countries=widgets.SelectMultiple(options=countries, description='Countries:'),\n",
                "         metrics=widgets.SelectMultiple(options=metrics, description='Metrics:'),\n",
                "         date_range=date_range)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Key Insights and Conclusions"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Key Findings:\n",
                "1. Global Trends\n",
                "   - Overall case and death trends\n",
                "   - Vaccination progress\n",
                "   - Regional variations\n",
                "   - Impact of vaccination on death rates\n",
                "\n",
                "2. Country Comparisons\n",
                "   - Success stories and challenges\n",
                "   - Vaccination strategies\n",
                "   - Healthcare system impacts\n",
                "   - Regional patterns\n",
                "\n",
                "3. Vaccination Progress\n",
                "   - Global vaccination rates\n",
                "   - Country-specific achievements\n",
                "   - Remaining challenges\n",
                "   - Correlation with case/death rates\n",
                "\n",
                "4. Recommendations\n",
                "   - Policy implications\n",
                "   - Future monitoring needs\n",
                "   - Areas for improvement\n",
                "   - Regional cooperation opportunities"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('notebooks/covid_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1) 