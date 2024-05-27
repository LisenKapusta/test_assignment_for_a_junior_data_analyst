# test_assignment_for_a_junior_data_analyst

#### Prerequisites
Ensure you have the necessary libraries installed:
```bash
pip install pandas matplotlib seaborn plotly graphviz statsmodels
```

#### Data Preparation
The data is read from a CSV file and formatted. The transaction dates are converted to a uniform format.
```bash
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from graphviz import Digraph
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv("data.csv", sep=";")

months = {
    "январь": "01",
    "февраль": "02",
    "март": "03",
    "апрель": "04",
    "май": "05",
    "июнь": "06",
    "июль": "07",
    "август": "08",
    "сентябрь": "09",
    "октябрь": "10",
    "ноябрь": "11",
    "декабрь": "12",
}

def format_date(row):
    for rus, eng in months.items():
        if rus in row:
            return row.replace(rus, eng) + "-01"

df["transaction_date"] = df["transaction_date"].apply(format_date)
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%m %Y-%d")
```
#### Customer Classification
Classify customers based on their monthly transaction activity.
```bash
def classify_customers(df: pd.DataFrame):
    df.sort_values("transaction_date", inplace=True)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    monthly_activity = df.groupby(
        [df["client_id"], df["transaction_date"].dt.to_period("M")]
    ).size()

    customer_classifications = defaultdict(lambda: defaultdict(str))

    for (client_id, month), count in monthly_activity.items():
        previous_month = (month - 1).asfreq("M")

        previous_status = customer_classifications[client_id][previous_month]

        if count > 0:
            if previous_status == "":
                current_status = "новые/возвращенные"
            elif previous_status in ["новые/возвращенные", "постоянные"]:
                current_status = "постоянные"
            else:
                current_status = "нерегулярные"
        else:
            if previous_status in ["постоянные", "нерегулярные"]:
                current_status = "неактивные"
            else:
                current_status = previous_status

        customer_classifications[client_id][month] = current_status

    classification_df = pd.DataFrame.from_dict(
        {
            (i, j): customer_classifications[i][j]
            for i in customer_classifications.keys()
            for j in customer_classifications[i].keys()
        },
        orient="index",
    )

    classification_df.index = pd.MultiIndex.from_tuples(classification_df.index)
    classification_df.index.names = ["client_id", "month"]
    classification_df.columns = ["status"]

    return classification_df
```

#### Transition State Classification
Determine detailed transition states for customers.
```bash
def classify_transition_states(classified_data: pd.DataFrame):
    detailed_classifications = defaultdict(dict)

    for client_id, data in classified_data.groupby(level=0):
        sorted_months = sorted(data.index.get_level_values(1))
        previous_status = None

        for month in sorted_months:
            current_status = data.loc[(client_id, month), "status"]

            if previous_status:
                if current_status == "постоянные":
                    if previous_status in ["постоянные"]:
                        detailed_status = "постоянные"
                    elif previous_status in ["нерегулярные", "новые/возвращенные"]:
                        detailed_status = "уходящие постоянные"
                elif current_status == "нерегулярные":
                    if previous_status in ["постоянные"]:
                        detailed_status = "нерегулярные"
                    else:
                        detailed_status = "разовые"
                elif current_status == "неактивные":
                    if previous_status in ["постоянные", "нерегулярные"]:
                        detailed_status = "отток"
                    else:
                        detailed_status = "потерянные"
            else:
                detailed_status = (
                    "неактивный" if current_status == "неактивные" else "новый"
                )

            detailed_classifications[client_id][month] = detailed_status
            previous_status = current_status

    detailed_classification_df = pd.DataFrame.from_dict(
        {
            (i, j): detailed_classifications[i][j]
            for i in detailed_classifications
            for j in detailed_classifications[i]
        },
        orient="index",
    )
    detailed_classification_df.index = pd.MultiIndex.from_tuples(
        detailed_classification_df.index
    )
    detailed_classification_df.index.names = ["client_id", "month"]
    detailed_classification_df.columns = ["detailed_status"]

    return detailed_classification_df

classified_data = classify_customers(df)
transition_classified_data = classify_transition_states(classified_data)
```

####Visualizing Customer Lifecycle
Create a visualization of the customer lifecycle using Graphviz.

```bash
def create_client_lifecycle():
    dot = Digraph(comment="Жизненный цикл клиента")

    states = [
        "новые/возвращенные",
        "постоянные",
        "нерегулярные",
        "разовые",
        "уходящие",
        "уходящие постоянные",
        "отток",
        "потерянные",
        "неактивные",
    ]

    for state in states:
        dot.node(state, state)

    transitions = [
        ("новые/возвращенные", "постоянные"),
        ("новые/возвращенные", "нерегулярные"),
        ("постоянные", "нерегулярные"),
        ("нерегулярные", "разовые"),
        ("разовые", "уходящие"),
        ("постоянные", "уходящие постоянные"),
        ("уходящие", "отток"),
        ("отток", "потерянные"),
        ("потерянные", "неактивные"),
        ("уходящие постоянные", "отток"),
    ]

    for start, end in transitions:
        dot.edge(start, end)

    dot.render("client_lifecycle", view=True)

create_client_lifecycle()
```
#### Visualizing Customer Transitions

Sankey Diagram
A Sankey diagram is used to visualize the transitions between different customer statuses over time.

```bash
create_client_lifecycle()

transition_counts = (
    transition_classified_data.groupby(["month", "detailed_status"])
    .size()
    .unstack(fill_value=0)
)

labels = sorted(set(transition_classified_data["detailed_status"]))
label_to_index = {label: i for i, label in enumerate(labels)}

sources = []
targets = []
values = []

for i in range(len(transition_counts.index) - 1):
    current_data = transition_counts.iloc[i]
    next_data = transition_counts.iloc[i + 1]
    for current_status, count in current_data.items():
        if count > 0:
            current_index = label_to_index[current_status]
            for next_status, next_count in next_data.items():
                if next_count > 0:
                    next_index = label_to_index[next_status]
                    sources.append(current_index)
                    targets.append(next_index)
                    values.append(min(count, next_count))

fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b'],
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=[('#d62728', '#2ca02c')[i % 2] for i in range(len(sources))]
            )
        )
    ]
)

fig.update_layout(title_text="Миграция клиентов по сегментам и периодам", font_size=15)
fig.show()
```
#### Heatmap
A heatmap is used to show the migration of customers across different segments over time.
```bash
pivot_table = transition_classified_data.pivot_table(
    index="month", columns="detailed_status", aggfunc="size", fill_value=0
)
pivot_table.rename_axis("month", axis="columns", inplace=True)
pivot_table.index.name = None
pivot_table.rename(columns={"новый": "новые/возвращенные"}, inplace=True)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Миграция клиентов по сегментам")
plt.xlabel("Статусы")
plt.ylabel("Месяц")
plt.show()
```
#### Forecasting Customer Segments

Forecast Setup
The forecasting process starts from April 2024 and predicts the number of clients in different segments for the next six months.
```bash
start_forecast = "2024-04"
forecast_months = pd.date_range(start=start_forecast, periods=6, freq="M")

last_known_value_date = pd.to_datetime("2024-03")
last_known_value = pivot_table.loc[last_known_value_date, "новые/возвращенные"]

forecast_values = [
    last_known_value * (1 + 0.05) ** i for i in range(len(forecast_months))
]

forecast_df = pd.DataFrame(
    {
        "month": forecast_months,
        "новые/возвращенные": forecast_values,
        "постоянные": [np.nan] * len(forecast_months),
        "нерегулярные": [np.nan] * len(forecast_months),
    }
).set_index("month")

result_df = pd.concat([pivot_table, forecast_df], sort=False)
```
#### Exponential Smoothing for Forecasting
Using the Exponential Smoothing method to forecast the customer count for the next six months
```bash
groups = ["новые/возвращенные", "постоянные"]
results = {}

for group in groups:
    data = pivot_table[group].dropna()

    if data.min() <= 0:
        model = ExponentialSmoothing(
            data, seasonal="add", seasonal_periods=12, initialization_method="estimated"
        ).fit()
    else:
        model = ExponentialSmoothing(
            data, seasonal="mul", seasonal_periods=12, initialization_method="estimated"
        ).fit()

    results[group] = model.forecast(6)
```
#### Visualization of Forecasting Results
Visualizing the forecasted customer segments.
```bash
fig, axes = plt.subplots(nrows=len(results), ncols=1, figsize=(10, 15), sharex=True)
fig.suptitle("Прогноз количества клиентов по группам")

for ax, (group, forecast) in zip(axes, results.items()):
    historical_data = pivot_table[group].dropna()

    if isinstance(historical_data.index, pd.PeriodIndex):
        historical_data.index = historical_data.index.to_timestamp()

    ax.plot(
        historical_data.index,
        historical_data,
        label="Исторические данные",
        marker="o",
        linestyle="-",
    )

    forecast.index = forecast.index.to_timestamp()

    ax.plot(forecast.index, forecast, label="Прогноз", marker="o", linestyle="--")
    ax.set_title(f'Прогноз для группы "{group}"')
    ax.set_xlabel("Дата")
    ax.set_ylabel("Количество клиентов")
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```
#### Heatmap of Complete Data
Plotting a heatmap of the combined historical and forecasted data.
```bash
complete_data = pd.concat([pivot_table, forecast_df], sort=False).fillna(0)

plt.figure(figsize=(25, 10))
sns.heatmap(complete_data.T, cmap="coolwarm", annot=True, fmt=".0f")
plt.title("Тепловая карта прогноза по группам клиентов")
plt.xlabel("Месяц")
plt.ylabel("Группы")
plt.show()
```
#### Additional Forecasts
Generating forecasts by filling NaN values with the mean and applying the Exponential Smoothing method.
```bash
groups = ["новые/возвращенные", "постоянные"]
results = {}

for group in groups:
    data = pivot_table[group].fillna(pivot_table[group].mean())

    if data.min() <= 0:
        model = ExponentialSmoothing(
            data, seasonal="add", seasonal_periods=12, initialization_method="estimated"
        ).fit()
    else:
        model = ExponentialSmoothing(
            data, seasonal="mul", seasonal_periods=12, initialization_method="estimated"
        ).fit()

    forecast = model.forecast(6)

    results[group] = forecast

for group, forecast in results.items():
    print(f"Прогноз для {group}:")
    print(forecast)
    print("\n")
```
