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

st.set_page_config(
    page_title="Nassau Candy Factory Reallocation",
    layout="wide",
)

st.title("Factory Reallocation & Shipping Optimization Recommendation System")
st.caption("Nassau Candy Distributor")


FACTORY_COORDS = {
    "Lot's O' Nuts": {"lat": 32.881893, "lon": -111.768036},
    "Wicked Choccy's": {"lat": 32.076176, "lon": -81.088371},
    "Sugar Shack": {"lat": 48.11914, "lon": -96.18115},
    "Secret Factory": {"lat": 41.446333, "lon": -90.565487},
    "The Other Factory": {"lat": 35.1175, "lon": -89.971107},
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
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Region",
    "Division",
    "Product Name",
    "Sales",
    "Units",
    "Gross Profit",
    "Cost",
]

MODEL_FEATURES = ["Product Name", "Origin Factory", "Region", "Ship Mode"]
FACTORIES = list(FACTORY_COORDS.keys())


def parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


@st.cache_data
def read_default_csv() -> pd.DataFrame:
    possible_files = [
        Path("Nassau Candy Distributor.csv"),
        Path("Nassau Candy Distributor (1).csv"),
        Path("Nassau_Candy_Distributor.csv"),
        Path("Nassau_Candy_Distributor__1_.csv"),
        Path.home() / "Downloads" / "Nassau Candy Distributor.csv",
        Path.home() / "Downloads" / "Nassau Candy Distributor (1).csv",
    ]
    for file_name in possible_files:
        if file_name.exists():
            return pd.read_csv(file_name)
    raise FileNotFoundError(
        "Put the Nassau Candy Distributor CSV in this app folder or upload it in the sidebar."
    )


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    df["Order Date"] = parse_dates(df["Order Date"])
    df["Ship Date"] = parse_dates(df["Ship Date"])
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Origin Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP)

    numeric_cols = ["Sales", "Units", "Gross Profit", "Cost", "Lead Time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=MODEL_FEATURES + ["Lead Time", "Gross Profit"]).copy()
    df = df[df["Lead Time"] >= 0].copy()

    q1 = df["Lead Time"].quantile(0.25)
    q3 = df["Lead Time"].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[df["Lead Time"].between(lower, upper)].copy()

    return df.reset_index(drop=True)


def load_data() -> pd.DataFrame:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        raw = read_default_csv()
    return prepare_data(raw)


@st.cache_resource
def train_models(data: pd.DataFrame):
    x = data[MODEL_FEATURES]
    y = data["Lead Time"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    results = []
    best_name = None
    best_pipeline = None
    best_score = -np.inf

    for name, model in models.items():
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_FEATURES),
            ],
            remainder="drop",
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

        selection_score = r2 - (rmse / max(y.mean(), 1))
        if selection_score > best_score:
            best_score = selection_score
            best_name = name
            best_pipeline = pipeline

    metrics_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    best_metrics = metrics_df[metrics_df["Model"] == best_name].iloc[0].to_dict()
    return best_pipeline, best_name, best_metrics, metrics_df


def predict_lead_time(model, product: str, factory: str, region: str, ship_mode: str) -> float:
    scenario = pd.DataFrame(
        [
            {
                "Product Name": product,
                "Origin Factory": factory,
                "Region": region,
                "Ship Mode": ship_mode,
            }
        ]
    )
    return max(float(model.predict(scenario)[0]), 0.0)


def average_profit(data: pd.DataFrame, product: str, region: str, ship_mode: str) -> float:
    subset = data[
        (data["Product Name"] == product)
        & (data["Region"] == region)
        & (data["Ship Mode"] == ship_mode)
    ]
    if subset.empty:
        subset = data[data["Product Name"] == product]
    if subset.empty:
        return float(data["Gross Profit"].mean())
    return float(subset["Gross Profit"].mean())


def scenario_table(
    data: pd.DataFrame,
    model,
    best_mae: float,
    product: str,
    region: str,
    ship_mode: str,
    priority: int,
) -> pd.DataFrame:
    current_factory = PRODUCT_FACTORY_MAP[product]
    current_lead = predict_lead_time(model, product, current_factory, region, ship_mode)
    base_profit = average_profit(data, product, region, ship_mode)
    weight_speed = priority / 100
    weight_profit = 1 - weight_speed
    rows = []

    for factory in FACTORIES:
        predicted_lead = predict_lead_time(model, product, factory, region, ship_mode)
        lead_reduction_days = current_lead - predicted_lead
        lead_reduction_pct = (
            (lead_reduction_days / current_lead) * 100 if current_lead > 0 else 0
        )
        profit_impact = base_profit * (lead_reduction_pct / 100) * 0.25
        confidence = max(0, min(100, 100 - (best_mae / max(predicted_lead, 1)) * 100))
        risk_reduction = lead_reduction_pct * 0.7 + confidence * 0.3
        score = (
            weight_speed * lead_reduction_pct
            + weight_profit * (profit_impact / max(abs(base_profit), 1) * 100)
            + 0.25 * risk_reduction
        )

        if factory == current_factory:
            recommendation = "Current Factory"
        elif lead_reduction_days <= 0:
            recommendation = "Do Not Reassign"
        elif profit_impact < 0:
            recommendation = "High Risk"
        else:
            recommendation = "Recommended"

        rows.append(
            {
                "Product": product,
                "Region": region,
                "Ship Mode": ship_mode,
                "Factory": factory,
                "Current Factory": current_factory,
                "Predicted Lead Time": predicted_lead,
                "Lead Time Reduction Days": lead_reduction_days,
                "Lead Time Reduction (%)": lead_reduction_pct,
                "Profit Impact": profit_impact,
                "Scenario Confidence Score": confidence,
                "Risk Reduction Score": risk_reduction,
                "Optimization Score": score,
                "Recommendation": recommendation,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("Optimization Score", ascending=False).reset_index(drop=True)


def build_all_recommendations(
    data: pd.DataFrame,
    model,
    selected_region: str,
    selected_ship_mode: str,
    priority: int,
    best_mae: float,
) -> pd.DataFrame:
    tables = []
    for product in sorted(data["Product Name"].dropna().unique()):
        table = scenario_table(
            data,
            model,
            best_mae,
            product,
            selected_region,
            selected_ship_mode,
            priority,
        )
        current_factory = PRODUCT_FACTORY_MAP[product]
        viable = table[
            (table["Factory"] != current_factory)
            & (table["Lead Time Reduction Days"] > 0)
            & (table["Profit Impact"] >= 0)
        ].copy()
        if not viable.empty:
            tables.append(viable.head(1))

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True).sort_values(
        "Optimization Score", ascending=False
    )


def create_route_clusters(data: pd.DataFrame) -> pd.DataFrame:
    route_df = (
        data.groupby(["Origin Factory", "Region", "Product Name"], as_index=False)
        .agg(
            Avg_Lead_Time=("Lead Time", "mean"),
            Total_Orders=("Order ID", "count"),
            Avg_Gross_Profit=("Gross Profit", "mean"),
        )
        .copy()
    )

    features = route_df[["Avg_Lead_Time", "Total_Orders", "Avg_Gross_Profit"]]
    scaled = StandardScaler().fit_transform(features)
    k = min(3, len(route_df))
    route_df["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(
        scaled
    )

    cluster_summary = route_df.groupby("Cluster")["Avg_Lead_Time"].mean().sort_values()
    labels = {}
    ordered_clusters = list(cluster_summary.index)
    for position, cluster in enumerate(ordered_clusters):
        if position == 0:
            labels[cluster] = "Optimal / Standard"
        elif position == len(ordered_clusters) - 1:
            labels[cluster] = "Congested / Slow Route"
        else:
            labels[cluster] = "High Volume / Monitor"

    route_df["Status"] = route_df["Cluster"].map(labels)
    return route_df


def format_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Product",
        "Current Factory",
        "Factory",
        "Predicted Lead Time",
        "Lead Time Reduction (%)",
        "Profit Impact",
        "Scenario Confidence Score",
        "Risk Reduction Score",
        "Optimization Score",
        "Recommendation",
    ]
    return df[cols].rename(columns={"Factory": "Move To"})


df = load_data()
model, winning_model_name, best_metrics, metrics_df = train_models(df)
st.session_state["model"] = model

st.sidebar.header("Optimization Simulator")
selected_product = st.sidebar.selectbox(
    "Product", sorted(df["Product Name"].dropna().unique())
)
selected_region = st.sidebar.selectbox("Destination Region", sorted(df["Region"].unique()))
selected_ship_mode = st.sidebar.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))
optimization_priority = st.sidebar.slider(
    "Optimization Priority: Speed vs Profit",
    min_value=0,
    max_value=100,
    value=50,
    help="0 = prioritize profit stability, 100 = prioritize lead-time reduction",
)
top_n = st.sidebar.slider("Top-N Recommendations", 3, 15, 10)

scenario_df = scenario_table(
    df,
    model,
    float(best_metrics["MAE"]),
    selected_product,
    selected_region,
    selected_ship_mode,
    optimization_priority,
)

current_factory = PRODUCT_FACTORY_MAP[selected_product]
recommended_df = scenario_df[
    (scenario_df["Factory"] != current_factory)
    & (scenario_df["Lead Time Reduction Days"] > 0)
    & (scenario_df["Profit Impact"] >= 0)
].copy()
best_recommendation = (
    recommended_df.iloc[0] if not recommended_df.empty else scenario_df.iloc[0]
)

all_recommendations = build_all_recommendations(
    df,
    model,
    selected_region,
    selected_ship_mode,
    optimization_priority,
    float(best_metrics["MAE"]),
)

coverage = (
    all_recommendations["Product"].nunique() / df["Product Name"].nunique() * 100
    if not all_recommendations.empty
    else 0
)
avg_reduction = (
    all_recommendations["Lead Time Reduction (%)"].mean()
    if not all_recommendations.empty
    else 0
)
total_profit_impact = (
    all_recommendations["Profit Impact"].sum() if not all_recommendations.empty else 0
)
avg_confidence = (
    all_recommendations["Scenario Confidence Score"].mean()
    if not all_recommendations.empty
    else scenario_df["Scenario Confidence Score"].mean()
)

st.subheader("Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Lead Time Reduction", f"{avg_reduction:.1f}%")
kpi2.metric("Profit Impact Stability", f"${total_profit_impact:,.2f}")
kpi3.metric("Scenario Confidence", f"{avg_confidence:.1f}%")
kpi4.metric("Recommendation Coverage", f"{coverage:.1f}%")

st.subheader("Model Evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Model", winning_model_name)
m2.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
m3.metric("MAE", f"{best_metrics['MAE']:.2f}")
m4.metric("R2", f"{best_metrics['R2']:.4f}")
with st.expander("Compare all trained models"):
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.subheader("Factory Optimization Simulator")
st.dataframe(
    format_recommendations(scenario_df).style.format(
        {
            "Predicted Lead Time": "{:.2f}",
            "Lead Time Reduction (%)": "{:.2f}",
            "Profit Impact": "${:,.2f}",
            "Scenario Confidence Score": "{:.1f}",
            "Risk Reduction Score": "{:.1f}",
            "Optimization Score": "{:.1f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

fig_factory = px.bar(
    scenario_df.sort_values("Predicted Lead Time"),
    x="Factory",
    y="Predicted Lead Time",
    color="Recommendation",
    text="Predicted Lead Time",
    title=f"Predicted Performance Across Factories: {selected_product}",
)
fig_factory.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_factory.update_layout(yaxis_title="Predicted Lead Time (Days)")
st.plotly_chart(fig_factory, use_container_width=True)

st.subheader("What-If Scenario Analysis")
current_row = scenario_df[scenario_df["Factory"] == current_factory].iloc[0]
comparison_df = pd.DataFrame(
    [
        {
            "Scenario": "Current Assignment",
            "Factory": current_factory,
            "Predicted Lead Time": current_row["Predicted Lead Time"],
        },
        {
            "Scenario": "Recommended Assignment",
            "Factory": best_recommendation["Factory"],
            "Predicted Lead Time": best_recommendation["Predicted Lead Time"],
        },
    ]
)
left, right = st.columns([1, 1])
with left:
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
with right:
    fig_what_if = px.bar(
        comparison_df,
        x="Scenario",
        y="Predicted Lead Time",
        color="Factory",
        text="Predicted Lead Time",
        title="Current vs Recommended Assignment",
    )
    fig_what_if.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig_what_if, use_container_width=True)

st.subheader("Recommendation Dashboard")
if all_recommendations.empty:
    st.info("No positive, profit-stable reassignment recommendations for this filter.")
else:
    top_recommendations = all_recommendations.head(top_n)
    st.dataframe(
        format_recommendations(top_recommendations).style.format(
            {
                "Predicted Lead Time": "{:.2f}",
                "Lead Time Reduction (%)": "{:.2f}",
                "Profit Impact": "${:,.2f}",
                "Scenario Confidence Score": "{:.1f}",
                "Risk Reduction Score": "{:.1f}",
                "Optimization Score": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_gain = px.bar(
        top_recommendations,
        x="Product",
        y="Lead Time Reduction (%)",
        color="Factory",
        title="Expected Efficiency Gains by Product",
        text="Lead Time Reduction (%)",
    )
    fig_gain.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_gain.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_gain, use_container_width=True)

st.subheader("Risk & Impact Panel")
risk_col1, risk_col2 = st.columns(2)
with risk_col1:
    if total_profit_impact >= 0:
        st.success(f"Profit impact alert: stable positive impact of ${total_profit_impact:,.2f}.")
    else:
        st.error(f"Profit impact alert: potential loss of ${abs(total_profit_impact):,.2f}.")
with risk_col2:
    high_risk_count = int(
        (scenario_df["Scenario Confidence Score"] < 60).sum()
        + (scenario_df["Lead Time Reduction Days"] < 0).sum()
    )
    if high_risk_count > 0:
        st.warning(f"High-risk reassignment warning: {high_risk_count} scenario(s) need review.")
    else:
        st.success("No high-risk warning for the selected scenario set.")

st.subheader("Route & Product Clustering")
route_df = create_route_clusters(df)
fig_cluster = px.scatter(
    route_df,
    x="Total_Orders",
    y="Avg_Lead_Time",
    color="Status",
    size="Avg_Gross_Profit",
    hover_data=["Origin Factory", "Region", "Product Name"],
    title="Route Performance Clusters",
)
st.plotly_chart(fig_cluster, use_container_width=True)

st.subheader("Network Reallocation Map")
map_view = st.radio(
    "Map view",
    ["Selected product scenario", "Top-N network recommendations"],
    horizontal=True,
)

if map_view == "Selected product scenario":
    map_recommendations = recommended_df.head(1).copy()
    map_title = f"Selected Product Relocation Path: {selected_product}"
else:
    map_recommendations = all_recommendations.head(top_n).copy()
    map_title = f"Top-{top_n} Network Relocation Paths"

map_rows = []
for factory, coords in FACTORY_COORDS.items():
    status = "Standard Facility"
    if not map_recommendations.empty:
        is_source = factory in map_recommendations["Current Factory"].values
        is_destination = factory in map_recommendations["Factory"].values
        if is_source and is_destination:
            status = "Source & Destination"
        elif is_source:
            status = "Move Away"
        elif is_destination:
            status = "Recommended Destination"
    map_rows.append(
        {
            "Factory": factory,
            "Lat": coords["lat"],
            "Lon": coords["lon"],
            "Status": status,
        }
    )

map_df = pd.DataFrame(map_rows)
fig_map = px.scatter_geo(
    map_df,
    lat="Lat",
    lon="Lon",
    text="Factory",
    color="Status",
    projection="albers usa",
    color_discrete_map={
        "Standard Facility": "#4C78A8",
        "Move Away": "#E45756",
        "Recommended Destination": "#54A24B",
        "Source & Destination": "#F58518",
    },
)

if not map_recommendations.empty:
    for _, row in map_recommendations.iterrows():
        source = FACTORY_COORDS[row["Current Factory"]]
        destination = FACTORY_COORDS[row["Factory"]]
        fig_map.add_scattergeo(
            lon=[source["lon"], destination["lon"]],
            lat=[source["lat"], destination["lat"]],
            mode="lines",
            line=dict(
                width=max(2, min(8, abs(row["Lead Time Reduction (%)"]) / 4)),
                color="#111827",
            ),
            opacity=0.75,
            hovertemplate=(
                f"<b>{row['Product']}</b><br>"
                f"{row['Current Factory']} to {row['Factory']}<br>"
                f"Lead-time reduction: {row['Lead Time Reduction (%)']:.2f}%<br>"
                f"Profit impact: ${row['Profit Impact']:,.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
else:
    st.info("No positive reassignment path for this map view. Current allocation is best under the selected filters.")

fig_map.update_traces(
    marker=dict(size=16, line=dict(width=2, color="white")),
    textfont=dict(size=13, color="black"),
)
fig_map.update_geos(showland=True, landcolor="#F3F4F6", showcountries=True)
fig_map.update_layout(
    title=map_title,
    height=600,
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
)
st.plotly_chart(fig_map, use_container_width=True)
