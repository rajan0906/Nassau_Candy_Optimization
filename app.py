import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

st.set_page_config(page_title="Nassau Candy Factory Reallocation", layout="wide")


FACTORY_COORDS = {
    "Lot's O' Nuts": {"lat": 32.881893, "lon": -111.768036},
    "Wicked Choccy's": {"lat": 32.076176, "lon": -81.088371},
    "Sugar Shack": {"lat": 48.119140, "lon": -96.181150},
    "Secret Factory": {"lat": 41.446333, "lon": -90.565487},
    "The Other Factory": {"lat": 35.117500, "lon": -89.971107},
}

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory",
}

REQUIRED_COLUMNS = [
    "Order ID", "Order Date", "Ship Date", "Ship Mode", "Region",
    "Product Name", "Sales", "Units", "Gross Profit", "Cost",
]
CATEGORICAL_FEATURES = ["Product Name", "Origin Factory", "Region", "Ship Mode"]
NUMERIC_FEATURES = ["Order Day Number", "Order Month", "Order Weekday"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FACTORIES = list(FACTORY_COORDS)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse the CSV's DD-MM-YYYY dates explicitly; never guess the format."""
    parsed = pd.to_datetime(series.astype(str).str.strip(), format="%d-%m-%Y", errors="coerce")
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"{bad_count:,} date value(s) are not in DD-MM-YYYY format.")
    return parsed


@st.cache_data(show_spinner=False)
def read_default_csv() -> pd.DataFrame:
    possible_files = [
        Path("Nassau Candy Distributor.csv"),
        Path("Nassau Candy Distributor (1).csv"),
        Path.home() / "Downloads" / "Nassau Candy Distributor.csv",
        Path.home() / "Downloads" / "Nassau Candy Distributor (1).csv",
    ]
    for file_name in possible_files:
        if file_name.exists():
            return pd.read_csv(file_name)
    raise FileNotFoundError("Upload the Nassau Candy Distributor CSV in the sidebar.")


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df["Order Date"] = parse_dates(df["Order Date"])
    df["Ship Date"] = parse_dates(df["Ship Date"])
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Origin Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP)

    for column in ["Sales", "Units", "Gross Profit", "Cost"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=CATEGORICAL_FEATURES + ["Order Date", "Ship Date", "Lead Time", "Gross Profit"]
    ).copy()
    df = df[df["Lead Time"] >= 0].copy()
    if len(df) < 100:
        raise ValueError("At least 100 valid rows are required to train and evaluate the models.")

    first_order_date = df["Order Date"].min()
    df["Order Day Number"] = (df["Order Date"] - first_order_date).dt.days
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Weekday"] = df["Order Date"].dt.dayofweek
    return df.reset_index(drop=True)


def load_data() -> pd.DataFrame:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    try:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file else read_default_csv()
        return prepare_data(raw_df)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        st.stop()


def make_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        sparse_threshold=0,
    )
    return Pipeline([( "preprocess", preprocessor), ("model", model)])


@st.cache_resource(show_spinner=False)
def train_models(data: pd.DataFrame):
    x = data[MODEL_FEATURES]
    y = data["Lead Time"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }

    fitted_models, results = {}, []
    baseline_prediction = np.full(len(y_test), y_train.mean())
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_prediction)))
    baseline_mae = float(mean_absolute_error(y_test, baseline_prediction))

    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        results.append({
            "Model": name,
            "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "MAE": float(mean_absolute_error(y_test, prediction)),
            "R2": float(r2_score(y_test, prediction)),
        })
        fitted_models[name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(["RMSE", "MAE"], ascending=True)
    best_name = metrics_df.iloc[0]["Model"]
    best_metrics = metrics_df.iloc[0].to_dict()
    best_metrics["Baseline RMSE"] = baseline_rmse
    best_metrics["Baseline MAE"] = baseline_mae
    return fitted_models[best_name], best_name, best_metrics, metrics_df


def predict_lead_time(model, product, factory, region, ship_mode, order_date) -> float:
    first_order_date = st.session_state["first_order_date"]
    order_timestamp = pd.Timestamp(order_date)
    scenario = pd.DataFrame([{
        "Product Name": product,
        "Origin Factory": factory,
        "Region": region,
        "Ship Mode": ship_mode,
        "Order Day Number": (order_timestamp - first_order_date).days,
        "Order Month": order_timestamp.month,
        "Order Weekday": order_timestamp.dayofweek,
    }])
    return max(0.0, float(model.predict(scenario)[0]))


def average_profit(data, product, region, ship_mode) -> float:
    subset = data.loc[
        (data["Product Name"] == product)
        & (data["Region"] == region)
        & (data["Ship Mode"] == ship_mode),
        "Gross Profit",
    ]
    if subset.empty:
        subset = data.loc[data["Product Name"] == product, "Gross Profit"]
    return float(subset.mean()) if not subset.empty else float(data["Gross Profit"].mean())


def scenario_table(data, model, product, region, ship_mode, order_date, priority, test_r2):
    current_factory = PRODUCT_FACTORY_MAP[product]
    current_lead = predict_lead_time(
        model, product, current_factory, region, ship_mode, order_date
    )
    base_profit = average_profit(data, product, region, ship_mode)
    speed_weight = priority / 100
    profit_weight = 1 - speed_weight
    rows = []

    for factory in FACTORIES:
        predicted_lead = predict_lead_time(
            model, product, factory, region, ship_mode, order_date
        )
        reduction_days = current_lead - predicted_lead
        reduction_pct = 100 * reduction_days / current_lead if current_lead else 0.0
        # This is an explicit planning estimate, not a historical causal profit result.
        profit_impact = base_profit * (reduction_pct / 100) * 0.25
        confidence = max(0.0, min(100.0, test_r2 * 100))
        score = (
            speed_weight * reduction_pct
            + profit_weight * (profit_impact / max(abs(base_profit), 1) * 100)
        )
        if factory == current_factory:
            recommendation = "Current Factory"
        elif test_r2 <= 0:
            recommendation = "Model Not Reliable"
        elif reduction_days <= 0:
            recommendation = "Do Not Reassign"
        else:
            recommendation = "Recommended"

        rows.append({
            "Product": product,
            "Region": region,
            "Ship Mode": ship_mode,
            "Factory": factory,
            "Current Factory": current_factory,
            "Predicted Lead Time": predicted_lead,
            "Lead Time Reduction Days": reduction_days,
            "Lead Time Reduction (%)": reduction_pct,
            "Estimated Profit Impact": profit_impact,
            "Scenario Confidence Score": confidence,
            "Optimization Score": score,
            "Recommendation": recommendation,
        })
    return pd.DataFrame(rows).sort_values("Optimization Score", ascending=False).reset_index(drop=True)


def build_recommendations(data, model, region, ship_mode, order_date, priority, test_r2):
    recommendations = []
    for product in sorted(data["Product Name"].unique()):
        table = scenario_table(
            data, model, product, region, ship_mode, order_date, priority, test_r2
        )
        viable = table.loc[table["Recommendation"] == "Recommended"]
        if not viable.empty:
            recommendations.append(viable.head(1))
    if not recommendations:
        return pd.DataFrame()
    return pd.concat(recommendations, ignore_index=True).sort_values(
        "Optimization Score", ascending=False
    )


@st.cache_data(show_spinner=False)
def create_route_clusters(data):
    route_df = data.groupby(["Origin Factory", "Region", "Product Name"], as_index=False).agg(
        Avg_Lead_Time=("Lead Time", "mean"),
        Total_Orders=("Order ID", "count"),
        Avg_Gross_Profit=("Gross Profit", "mean"),
    )
    cluster_count = min(3, len(route_df))
    if cluster_count < 2:
        route_df["Status"] = "Insufficient route variation"
        return route_df
    values = StandardScaler().fit_transform(
        route_df[["Avg_Lead_Time", "Total_Orders", "Avg_Gross_Profit"]]
    )
    route_df["Cluster"] = KMeans(
        n_clusters=cluster_count, random_state=42, n_init=10
    ).fit_predict(values)
    ranking = route_df.groupby("Cluster")["Avg_Lead_Time"].mean().sort_values().index.tolist()
    labels = {ranking[0]: "Optimal / Standard", ranking[-1]: "Congested / Slow Route"}
    for cluster in ranking[1:-1]:
        labels[cluster] = "High Volume / Monitor"
    route_df["Status"] = route_df["Cluster"].map(labels)
    return route_df


def display_table(frame):
    columns = [
        "Product", "Current Factory", "Factory", "Predicted Lead Time",
        "Lead Time Reduction (%)", "Estimated Profit Impact",
        "Scenario Confidence Score", "Optimization Score", "Recommendation",
    ]
    return frame[columns].rename(columns={"Factory": "Move To"})


df = load_data()
st.session_state["first_order_date"] = df["Order Date"].min()
model, winning_model_name, best_metrics, metrics_df = train_models(df)
test_r2 = float(best_metrics["R2"])

st.title("Factory Reallocation & Shipping Optimization Recommendation System")
st.caption("Nassau Candy Distributor")

median_lead_time = float(df["Lead Time"].median())
if median_lead_time > 60:
    st.warning(
        f"Data quality notice: the uploaded CSV has a median lead time of {median_lead_time:,.0f} days. "
        "Verify Order Date and Ship Date before using these outputs for an operational decision."
    )

st.sidebar.header("Optimization Simulator")
selected_product = st.sidebar.selectbox("Product", sorted(df["Product Name"].unique()))
selected_region = st.sidebar.selectbox("Destination Region", sorted(df["Region"].unique()))
selected_ship_mode = st.sidebar.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))
# The brief only requires product, region, ship mode, and priority controls.
# Use the latest observed order date internally for a consistent scenario forecast.
scenario_order_date = df["Order Date"].max().date()
priority = st.sidebar.slider("Optimization Priority: Speed vs Profit", 0, 100, 50)
max_recommendations = min(15, df["Product Name"].nunique())
top_n = st.sidebar.slider(
    "Top-N Recommendations", 3, max_recommendations, min(10, max_recommendations)
)

scenario_df = scenario_table(
    df, model, selected_product, selected_region, selected_ship_mode,
    scenario_order_date, priority, test_r2
)
current_factory = PRODUCT_FACTORY_MAP[selected_product]
best_recommendation = scenario_df.loc[
    scenario_df["Recommendation"] == "Recommended"
].head(1)
all_recommendations = build_recommendations(
    df, model, selected_region, selected_ship_mode, scenario_order_date, priority, test_r2
)

coverage = 100 * all_recommendations["Product"].nunique() / df["Product Name"].nunique() if not all_recommendations.empty else 0.0
avg_reduction = float(all_recommendations["Lead Time Reduction (%)"].mean()) if not all_recommendations.empty else 0.0
total_profit_impact = float(all_recommendations["Estimated Profit Impact"].sum()) if not all_recommendations.empty else 0.0

st.subheader("Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Lead Time Reduction", f"{avg_reduction:.1f}%")
k2.metric("Estimated Profit Impact", f"${total_profit_impact:,.2f}")
k3.metric("Scenario Confidence", f"{max(test_r2, 0) * 100:.1f}%")
k4.metric("Recommendation Coverage", f"{coverage:.1f}%")

st.subheader("Model Evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Model", winning_model_name)
m2.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
m3.metric("MAE", f"{best_metrics['MAE']:.2f}")
m4.metric("R2", f"{test_r2:.4f}")
if test_r2 <= 0:
    st.error(
        "The best model does not outperform the mean-lead-time baseline. "
        "Recommendations are disabled until the source dates or additional routing data are corrected."
    )
else:
    st.success("The selected model outperforms the mean-lead-time baseline on the held-out test set.")
with st.expander("Compare all trained models"):
    st.dataframe(metrics_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.4f}"}), use_container_width=True, hide_index=True)

st.subheader("Factory Optimization Simulator")
st.dataframe(
    display_table(scenario_df).style.format({
        "Predicted Lead Time": "{:.2f}", "Lead Time Reduction (%)": "{:.2f}",
        "Estimated Profit Impact": "${:,.2f}", "Scenario Confidence Score": "{:.1f}",
        "Optimization Score": "{:.1f}",
    }), use_container_width=True, hide_index=True,
)

fig_factory = px.bar(
    scenario_df.sort_values("Predicted Lead Time"), x="Factory", y="Predicted Lead Time",
    color="Recommendation", text="Predicted Lead Time",
    title=f"Predicted Performance Across Factories: {selected_product}",
)
fig_factory.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_factory.update_layout(yaxis_title="Predicted Lead Time (Days)")
st.plotly_chart(fig_factory, use_container_width=True)

st.subheader("What-If Scenario Analysis")
current_row = scenario_df.loc[scenario_df["Factory"] == current_factory].iloc[0]
comparison_rows = [{"Scenario": "Current Assignment", "Factory": current_factory, "Predicted Lead Time": current_row["Predicted Lead Time"]}]
if not best_recommendation.empty:
    row = best_recommendation.iloc[0]
    comparison_rows.append({"Scenario": "Recommended Assignment", "Factory": row["Factory"], "Predicted Lead Time": row["Predicted Lead Time"]})
comparison_df = pd.DataFrame(comparison_rows)
left, right = st.columns(2)
with left:
    st.dataframe(comparison_df.style.format({"Predicted Lead Time": "{:.2f}"}), use_container_width=True, hide_index=True)
with right:
    fig_what_if = px.bar(comparison_df, x="Scenario", y="Predicted Lead Time", color="Factory", text="Predicted Lead Time")
    fig_what_if.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig_what_if, use_container_width=True)

st.subheader("Recommendation Dashboard")
if all_recommendations.empty:
    st.info("No data-supported positive reassignment recommendation exists for the selected filters.")
else:
    top_recommendations = all_recommendations.head(top_n)
    st.dataframe(display_table(top_recommendations).style.format({
        "Predicted Lead Time": "{:.2f}", "Lead Time Reduction (%)": "{:.2f}",
        "Estimated Profit Impact": "${:,.2f}", "Scenario Confidence Score": "{:.1f}", "Optimization Score": "{:.1f}",
    }), use_container_width=True, hide_index=True)
    fig_gain = px.bar(top_recommendations, x="Product", y="Lead Time Reduction (%)", color="Factory", text="Lead Time Reduction (%)", title="Expected Efficiency Gains by Product")
    fig_gain.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_gain.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_gain, use_container_width=True)

st.subheader("Risk & Impact Panel")
risk1, risk2 = st.columns(2)
with risk1:
    if test_r2 <= 0:
        st.error("Model risk: do not use the forecast for an operational reassignment decision.")
    elif total_profit_impact >= 0:
        st.success(f"Estimated profit impact: ${total_profit_impact:,.2f}.")
    else:
        st.warning(f"Estimated profit risk: ${abs(total_profit_impact):,.2f}.")
with risk2:
    negative_options = int((scenario_df["Lead Time Reduction Days"] < 0).sum())
    if negative_options:
        st.warning(
            f"{negative_options} factory option(s) are slower than the current assignment."
        )
    else:
        st.success("No slower option was predicted for this scenario.")

st.subheader("Route & Product Clustering")
route_df = create_route_clusters(df)
fig_cluster = px.scatter(route_df, x="Total_Orders", y="Avg_Lead_Time", color="Status", size="Avg_Gross_Profit", hover_data=["Origin Factory", "Region", "Product Name"], title="Route Performance Clusters")
st.plotly_chart(fig_cluster, use_container_width=True)

st.subheader("Network Reallocation Map")
map_view = st.radio("Map view", ["Selected product scenario", "Top-N network recommendations"], horizontal=True)
if map_view == "Selected product scenario":
    map_recommendations = best_recommendation.copy()
    map_title = f"Selected Product Relocation Path: {selected_product}"
else:
    map_recommendations = all_recommendations.head(top_n).copy()
    map_title = f"Top-{top_n} Network Relocation Paths"

map_rows = []
for factory, coords in FACTORY_COORDS.items():
    source = not map_recommendations.empty and factory in map_recommendations["Current Factory"].values
    destination = not map_recommendations.empty and factory in map_recommendations["Factory"].values
    status = "Source & Destination" if source and destination else "Move Away" if source else "Recommended Destination" if destination else "Standard Facility"
    map_rows.append({"Factory": factory, "Lat": coords["lat"], "Lon": coords["lon"], "Status": status})

fig_map = px.scatter_geo(pd.DataFrame(map_rows), lat="Lat", lon="Lon", text="Factory", color="Status", projection="albers usa", color_discrete_map={"Standard Facility": "#4C78A8", "Move Away": "#E45756", "Recommended Destination": "#54A24B", "Source & Destination": "#F58518"})
for _, row in map_recommendations.iterrows():
    source, destination = FACTORY_COORDS[row["Current Factory"]], FACTORY_COORDS[row["Factory"]]
    fig_map.add_scattergeo(lon=[source["lon"], destination["lon"]], lat=[source["lat"], destination["lat"]], mode="lines", line={"width": 3, "color": "#111827"}, hovertemplate=f"<b>{row['Product']}</b><br>{row['Current Factory']} to {row['Factory']}<br>Lead-time reduction: {row['Lead Time Reduction (%)']:.2f}%<extra></extra>", showlegend=False)
fig_map.update_traces(marker={"size": 16, "line": {"width": 2, "color": "white"}}, textfont={"size": 13, "color": "black"}, selector={"mode": "markers+text"})
fig_map.update_geos(showland=True, landcolor="#F3F4F6", showcountries=True)
fig_map.update_layout(title=map_title, height=600, margin={"r": 0, "t": 40, "l": 0, "b": 0})
st.plotly_chart(fig_map, use_container_width=True)
