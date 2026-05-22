import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background: #0F1117;
    }

    .stApp {
        background: linear-gradient(135deg, #0F1117 0%, #1a1d2e 50%, #0F1117 100%);
    }

    /* Title */
    h1 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.4rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 0 !important;
    }

    h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #E8EAED !important;
    }

    p, li, label {
        color: #9AA0AC !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2235 0%, #252a40 100%);
        border: 1px solid #2d3250;
        border-radius: 16px;
        padding: 20px !important;
        box-shadow: 0 4px 24px rgba(102, 126, 234, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }

    [data-testid="metric-container"] label {
        color: #667eea !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E8EAED !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e2235;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #2d3250;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9AA0AC !important;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        margin-top: 8px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1e2235;
        border: 2px dashed #2d3250;
        border-radius: 16px;
        padding: 20px;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2d3250;
    }

    /* Success / Warning / Error boxes */
    .stSuccess {
        background: #0d2818 !important;
        border-left: 4px solid #00c853 !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background: #2a1f0a !important;
        border-left: 4px solid #ffab00 !important;
        border-radius: 8px !important;
    }

    .stError {
        background: #2a0a0a !important;
        border-left: 4px solid #ff1744 !important;
        border-radius: 8px !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background: #1e2235 !important;
        border-color: #2d3250 !important;
        border-radius: 10px !important;
    }

    /* Caption */
    .stCaption {
        color: #667eea !important;
        font-size: 0.8rem !important;
    }

    /* Divider */
    hr {
        border-color: #2d3250 !important;
        margin: 24px 0 !important;
    }

    /* Spinner */
    .stSpinner {
        color: #667eea !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1e2235; }
    ::-webkit-scrollbar-thumb { background: #667eea; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────────
CHART_THEME = {
    'paper_bgcolor': '#1e2235',
    'plot_bgcolor':  '#1e2235',
    'font':          dict(color='#9AA0AC', family='DM Sans'),
    'title_font':    dict(color='#E8EAED', size=15, family='DM Sans'),
    'gridcolor':     '#2d3250',
    'linecolor':     '#2d3250',
}

def style_chart(fig, height=420):
    fig.update_layout(
        paper_bgcolor=CHART_THEME['paper_bgcolor'],
        plot_bgcolor=CHART_THEME['plot_bgcolor'],
        font=CHART_THEME['font'],
        title_font=CHART_THEME['title_font'],
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=CHART_THEME['gridcolor'], linecolor=CHART_THEME['linecolor']),
        yaxis=dict(gridcolor=CHART_THEME['gridcolor'], linecolor=CHART_THEME['linecolor']),
    )
    return fig

# ── HEADER ───────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🛒 Sales Intelligence Dashboard")
    st.markdown("<p style='color:#9AA0AC;margin-top:-10px;font-size:1rem'>Discover hidden patterns using Machine Learning — Association Rules · Clustering · Forecasting · Feature Importance</p>", unsafe_allow_html=True)

st.divider()

# ── FILE UPLOAD ──────────────────────────────────────────────
st.markdown("### 📂 Upload Your Sales Data")
uploaded_file = st.file_uploader("Upload Excel file (xlsx)", type=["xlsx"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style='background:#1e2235;border:2px dashed #2d3250;border-radius:16px;padding:40px;text-align:center;margin-top:20px'>
        <p style='color:#667eea;font-size:1.1rem;font-weight:600'>👆 Upload your Sales_Data.xlsx to begin</p>
        <p style='color:#9AA0AC;font-size:0.85rem'>Supports any retail sales Excel file with Transaction, Product, Amount, Store and Date columns</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    sales = pd.read_excel(file)
    sales['Date']      = pd.to_datetime(sales['Date'])
    sales['Month']     = sales['Date'].dt.month
    sales['Weekday']   = sales['Date'].dt.day_name()
    sales['YearMonth'] = sales['Date'].dt.to_period('M').astype(str)
    sales['Weekend']   = sales['Weekday'].isin(['Saturday', 'Sunday'])
    sales              = sales.drop_duplicates()
    return sales

sales     = load_data(uploaded_file)
sales_pos = sales[sales['Net Amount'] > 0]

st.success(f"✅  Data loaded successfully — **{len(sales):,} rows** | **{sales['Store No_'].nunique()} stores** | **{sales['Item No_'].nunique():,} unique products**")

# ── KPI CARDS ────────────────────────────────────────────────
st.markdown("### 📊 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Total Revenue",     f"AED {sales_pos['Net Amount'].sum():,.0f}")
k2.metric("🧾 Transactions",      f"{sales_pos['Transaction No_'].nunique():,}")
k3.metric("📦 Unique Products",   f"{sales['Item No_'].nunique():,}")
k4.metric("🏪 Stores",            f"{sales['Store No_'].nunique()}")
k5.metric("↩️ Total Returns",     f"AED {abs(sales[sales['Net Amount']<0]['Net Amount'].sum()):,.0f}")

st.divider()

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔗  Association Rules",
    "🏪  Store Clustering",
    "📈  Revenue Trends",
    "📦  Pareto Analysis",
    "🏠  Dept Heatmap",
    "🔮  Forecast",
    "🌲  Feature Importance"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — ASSOCIATION RULES
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔗 Market Basket Analysis")
    st.markdown("<p>Discovers which products are <b>bought together</b> using the Apriori algorithm. Higher lift = stronger pattern.</p>", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Parameters")
    col1, col2 = st.columns(2)
    with col1:
        min_count = st.slider("🔢 Minimum Transactions", min_value=10, max_value=200, value=50, step=10, help="How many times must a combo appear?")
    with col2:
        min_lift = st.slider("📈 Minimum Lift Score", min_value=1.0, max_value=15.0, value=3.0, step=0.5, help="Higher = stronger rules only. Lift > 5 is very strong.")

    if st.button("🔍  Find Hidden Patterns", type="primary"):
        with st.spinner("🔄 Running Apriori algorithm... please wait 15 seconds"):
            try:
                sc = sales[(sales['Net Amount'] > 0) & (sales['Item Category'] != 'Service')].copy()
                basket = (sc.groupby(['Transaction No_', 'Subgroup Desc'])['Quantity'].sum().unstack(fill_value=0).gt(0).astype('bool'))
                total_transactions = basket.shape[0]
                min_support = min_count / total_transactions
                st.caption(f"ℹ️  Support = {min_support:.4f}  ({min_count} combos out of {total_transactions:,} transactions)")

                freq  = apriori(basket, min_support=min_support, use_colnames=True)
                rules = association_rules(freq, metric='lift', min_threshold=min_lift)
                rules['IF']   = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['THEN'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                rules['rule'] = rules['IF'] + '  →  ' + rules['THEN']
                rules = rules.sort_values('lift', ascending=False)

                st.success(f"✅  Found **{len(rules)} rules** with lift ≥ {min_lift}")

                # Bar chart
                top10 = rules.head(10)
                fig1 = px.bar(top10, x='lift', y='rule', orientation='h',
                              color='lift', color_continuous_scale='Purples',
                              title='🏆 Top 10 Rules by Lift Score',
                              labels={'lift': 'Lift Score', 'rule': ''})
                fig1 = style_chart(fig1, height=440)
                fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig1, use_container_width=True)

                # Bubble chart
                fig2 = px.scatter(rules, x='support', y='confidence',
                                  size='lift', color='lift',
                                  color_continuous_scale='Plasma',
                                  hover_data=['IF', 'THEN'],
                                  title='🫧 Support vs Confidence (bubble size = lift strength)',
                                  labels={'support': 'Support', 'confidence': 'Confidence'})
                fig2 = style_chart(fig2, height=420)
                st.plotly_chart(fig2, use_container_width=True)

                # Rules table
                st.markdown("#### 📋 All Rules")
                st.dataframe(rules[['IF', 'THEN', 'support', 'confidence', 'lift']].round(4).reset_index(drop=True), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {e}. Try lowering the Minimum Transactions value.")

# ════════════════════════════════════════════════════════════
# TAB 2 — STORE CLUSTERING
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏪 Store Segmentation")
    st.markdown("<p>Groups stores into segments based on Revenue, Quantity and Transactions using <b>KMeans Clustering</b>.</p>", unsafe_allow_html=True)

    n_clusters = st.slider("🔢 Number of Store Groups", min_value=2, max_value=5, value=3, step=1, help="3 = High, Mid, Low segments")

    store_data = sales_pos.groupby('Store No_').agg(
        Revenue      = ('Net Amount',      'sum'),
        Quantity     = ('Quantity',        'sum'),
        Transactions = ('Transaction No_', 'nunique')
    ).reset_index()
    store_data.columns = ['Store', 'Revenue', 'Quantity', 'Transactions']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(store_data[['Revenue', 'Quantity', 'Transactions']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    store_data['Cluster'] = kmeans.fit_predict(scaled).astype(str)

    fig3 = px.scatter(store_data, x='Revenue', y='Transactions',
                      color='Cluster', size='Quantity', text='Store',
                      title=f'🏪 Store Segmentation — {n_clusters} Groups (bubble size = quantity sold)',
                      color_discrete_sequence=['#667eea', '#f093fb', '#4ecdc4', '#ff6b6b', '#ffd93d'])
    fig3.update_traces(textposition='top center', textfont=dict(color='#E8EAED', size=11))
    fig3 = style_chart(fig3, height=520)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### 📋 Store Summary")
    st.dataframe(store_data.sort_values('Revenue', ascending=False).reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — REVENUE TRENDS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Revenue Trends")

    col1, col2 = st.columns(2)
    with col1:
        monthly = sales_pos.groupby('YearMonth')['Net Amount'].sum().reset_index()
        fig4 = px.line(monthly, x='YearMonth', y='Net Amount', markers=True,
                       title='📅 Monthly Revenue — drag slider to zoom',
                       labels={'Net Amount': 'Revenue (AED)', 'YearMonth': 'Month'})
        fig4.update_traces(line_color='#667eea', line_width=2.5,
                           marker=dict(color='#f093fb', size=9, line=dict(color='#667eea', width=2)))
        fig4 = style_chart(fig4, height=360)
        fig4.update_layout(hovermode='x unified', xaxis=dict(rangeslider=dict(visible=True, bgcolor='#1e2235')))
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        week = sales_pos.copy()
        week['Day Type'] = week['Weekend'].map({True: 'Weekend', False: 'Weekday'})
        week_avg = week.groupby('Day Type')['Net Amount'].mean().reset_index()
        fig5 = px.bar(week_avg, x='Day Type', y='Net Amount',
                      color='Day Type',
                      color_discrete_map={'Weekend': '#f093fb', 'Weekday': '#667eea'},
                      title='📅 Average Sale — Weekend vs Weekday',
                      labels={'Net Amount': 'Avg Revenue (AED)'})
        fig5.update_layout(showlegend=False)
        fig5 = style_chart(fig5, height=360)
        st.plotly_chart(fig5, use_container_width=True)

    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    selected_store = st.selectbox("🏪 Filter by Store", options=['All Stores'] + list(sales_pos['Store No_'].unique()))

    if selected_store == 'All Stores':
        filtered_sales = sales_pos
    else:
        filtered_sales = sales_pos[sales_pos['Store No_'] == selected_store]

    day_sales = filtered_sales.groupby('Weekday')['Net Amount'].sum().reindex(day_order).reset_index()
    fig6 = px.bar(day_sales, x='Weekday', y='Net Amount',
                  color='Net Amount',
                  color_continuous_scale='Purples',
                  title=f'📅 Revenue by Day of Week — {selected_store}',
                  labels={'Net Amount': 'Revenue (AED)'})
    fig6.update_layout(hovermode='x unified')
    fig6 = style_chart(fig6, height=380)
    st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — PARETO
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📦 Pareto Analysis — 80/20 Rule")
    st.markdown("<p>Which <b>small % of products</b> drive <b>80% of revenue?</b></p>", unsafe_allow_html=True)

    pareto = (sales_pos.groupby('Search Description')['Net Amount']
              .sum().sort_values(ascending=False).reset_index())
    pareto['Cumulative %'] = pareto['Net Amount'].cumsum() / pareto['Net Amount'].sum() * 100
    top80 = pareto[pareto['Cumulative %'] <= 80]

    p1, p2, p3 = st.columns(3)
    p1.metric("📦 Total Products",           f"{len(pareto):,}")
    p2.metric("🎯 Products for 80% Revenue", f"{len(top80):,}")
    p3.metric("📊 That is only",             f"{len(top80)/len(pareto)*100:.1f}% of products")

    fig7 = go.Figure()
    fig7.add_trace(go.Bar(x=list(range(len(pareto))), y=pareto['Net Amount'],
                          name='Revenue', marker_color='#667eea', opacity=0.8))
    fig7.add_trace(go.Scatter(x=list(range(len(pareto))), y=pareto['Cumulative %'],
                              name='Cumulative %', yaxis='y2',
                              line=dict(color='#f093fb', width=2.5)))
    fig7.add_hline(y=80, line_dash='dash', line_color='#ff6b6b',
                   annotation_text='80% revenue line',
                   annotation_font_color='#ff6b6b', yref='y2')
    fig7.update_layout(
        title='📦 Pareto Chart — Products vs Revenue',
        yaxis2=dict(overlaying='y', side='right', range=[0, 105],
                    gridcolor='#2d3250', linecolor='#2d3250', color='#9AA0AC'),
        paper_bgcolor='#1e2235', plot_bgcolor='#1e2235',
        font=dict(color='#9AA0AC', family='DM Sans'),
        title_font=dict(color='#E8EAED', size=15),
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor='#2d3250', linecolor='#2d3250'),
        yaxis=dict(gridcolor='#2d3250', linecolor='#2d3250'),
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("#### 🏆 Top 10 Products by Revenue")
    st.dataframe(pareto.head(10).round(2), use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — HEATMAP
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🏠 Department Affinity Heatmap")
    st.markdown("<p>Shows which departments are <b>bought together</b> in the same transaction.</p>", unsafe_allow_html=True)

    if st.button("🔥  Load Heatmap", type="primary"):
        with st.spinner("🔄 Building heatmap... please wait"):
            dept = (sales_pos.groupby(['Transaction No_', 'Department Desc'])['Quantity']
                    .sum().unstack().fillna(0))
            corr = dept.corr()
            fig8 = px.imshow(corr, color_continuous_scale='Purples',
                             title='🏠 Department Co-Purchase Correlation', aspect='auto')
            fig8.update_layout(paper_bgcolor='#1e2235', plot_bgcolor='#1e2235',
                               font=dict(color='#9AA0AC'), title_font=dict(color='#E8EAED', size=15),
                               height=620, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig8, use_container_width=True)

    st.info("💡 Dark purple = strongly bought together  |  White = no relationship  |  Look for **Bedroom + Mattress** and **Living + Sofa**")

# ════════════════════════════════════════════════════════════
# TAB 6 — FORECAST
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🔮 Revenue Forecast")
    st.markdown("<p>Predicts future revenue using <b>Facebook Prophet</b> based on past patterns.</p>", unsafe_allow_html=True)

    from prophet import Prophet

    forecast_data = sales_pos.groupby('Date')['Net Amount'].sum().reset_index()
    forecast_data.columns = ['ds', 'y']

    col1, col2 = st.columns(2)
    with col1:
        store_option = st.selectbox("🏪 Select Store", options=['All Stores'] + list(sales['Store No_'].unique()))
    with col2:
        forecast_days = st.slider("📅 Forecast Days Ahead", min_value=7, max_value=90, value=30, step=7)

    if store_option != 'All Stores':
        store_sales = sales_pos[sales_pos['Store No_'] == store_option]
        forecast_data = store_sales.groupby('Date')['Net Amount'].sum().reset_index()
        forecast_data.columns = ['ds', 'y']

    if st.button("🔮  Generate Forecast", type="primary"):
        with st.spinner("🔄 Forecasting revenue... please wait"):
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(forecast_data)
            future   = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            fig_f = px.line(forecast, x='ds', y='yhat',
                            title=f'🔮 Revenue Forecast — Next {forecast_days} Days ({store_option})',
                            labels={'ds': 'Date', 'yhat': 'Predicted Revenue (AED)'})
            fig_f.update_traces(line_color='#667eea', line_width=2.5)
            fig_f.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                              mode='lines', name='Upper Bound',
                              line=dict(dash='dash', color='#4ecdc4', width=1))
            fig_f.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                              mode='lines', name='Lower Bound',
                              line=dict(dash='dash', color='#ff6b6b', width=1))
            fig_f = style_chart(fig_f, height=460)
            st.plotly_chart(fig_f, use_container_width=True)

            future_only = forecast[forecast['ds'] > forecast_data['ds'].max()]
            st.markdown("#### 📊 Forecast Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("💰 Total Predicted Revenue", f"AED {future_only['yhat'].sum():,.0f}")
            c2.metric("📅 Daily Average",            f"AED {future_only['yhat'].mean():,.0f}")
            c3.metric("🏆 Peak Day",                 f"{future_only.loc[future_only['yhat'].idxmax(), 'ds'].strftime('%d %b %Y')}")

# ════════════════════════════════════════════════════════════
# TAB 7 — RANDOM FOREST
# ════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### 🌲 Feature Importance — What Drives Sales?")
    st.markdown("<p>Uses <b>Random Forest</b> to find which factors have the biggest impact on revenue.</p>", unsafe_allow_html=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    if st.button("🌲  Run Random Forest Analysis", type="primary"):
        with st.spinner("🔄 Training model on 144,000 rows... please wait 30 seconds"):
            rf_data              = sales_pos.copy()
            rf_data['DayOfWeek'] = rf_data['Date'].dt.dayofweek
            rf_data['MonthNum']  = rf_data['Date'].dt.month
            rf_data['IsWeekend'] = rf_data['Weekend'].astype(int)
            rf_data['Store_enc'] = rf_data['Store No_'].astype('category').cat.codes
            rf_data['Dept_enc']  = rf_data['Department Desc'].astype('category').cat.codes
            rf_data['Cat_enc']   = rf_data['Item Category'].astype('category').cat.codes

            features = ['DayOfWeek', 'MonthNum', 'IsWeekend', 'Store_enc', 'Dept_enc', 'Cat_enc', 'Quantity']
            X = rf_data[features]
            y = rf_data['Net Amount']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)

            importance = pd.DataFrame({
                'Feature'   : ['Day of Week', 'Month', 'Is Weekend', 'Store', 'Department', 'Category', 'Quantity'],
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            score = rf_model.score(X_test, y_test)
            st.metric("🎯 Model Accuracy", f"{score*100:.1f}%")

            fig_rf = px.bar(importance, x='Importance', y='Feature',
                            orientation='h', color='Importance',
                            color_continuous_scale='Greens',
                            title='🌲 What Factors Affect Sales Amount Most?',
                            labels={'Importance': 'Importance Score', 'Feature': ''})
            fig_rf = style_chart(fig_rf, height=400)
            fig_rf.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_rf, use_container_width=True)

            top_feature = importance.iloc[0]['Feature']
            st.success(f"🏆 **Most important factor:** {top_feature} has the biggest impact on revenue")
            st.markdown("""
            | Factor | What it means |
            |---|---|
            | **Quantity** | How many items bought — directly drives revenue |
            | **Department** | Which department the product belongs to |
            | **Store** | Which store location — affects spending behavior |
            | **Category** | Product category — Furniture vs Homeware etc |
            | **Month** | Seasonal patterns — which month drives more sales |
            | **Is Weekend** | Whether it is a weekend — spending changes |
            | **Day of Week** | Specific day — Saturday vs Monday behavior |
            """)

# ════════════════════════════════════════════════════════════
# STRATEGIC INSIGHTS
# ════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 💡 Strategic Insights & Recommendations")
st.markdown("<p>Key business actions based on the analysis above</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🛒 Product Strategy")
    st.success("**🎁 Bundle Opportunity** — Sheets + Duvet Covers have lift of 14.9x. Create a Bedding Bundle offer immediately.")
    st.success("**🛋️ Furniture Set** — Coffee Table + Side Table bought together 15x more than random. Always display together in store.")
    st.warning("**📉 Slow Movers** — Products below median quantity need promotions or should be removed from inventory.")

with col2:
    st.markdown("#### 🏪 Store Strategy")
    st.success("**🏆 Top Stores** — ALAIN and DHMAL are high performers. Use their strategies as benchmark.")
    st.error("**🚨 Urgent Attention** — OMNWEB has only 18 transactions. BKOFM has only 29 transactions. Immediate review needed.")
    st.warning("**📅 Weekend Focus** — Saturday and Sunday drive highest revenue. Schedule promotions on weekends.")

st.divider()

col3, col4 = st.columns(2)
with col3:
    st.markdown("#### 📦 Inventory Strategy")
    st.success("**✅ Pareto Confirmed** — Small % of products drive 80% of revenue. Never go out of stock on top products.")
    st.warning("**↩️ Return Rate Alert** — OMNWEB has 14% return rate — 6x company average. Investigate quality and delivery.")

with col4:
    st.markdown("#### 📅 Timing Strategy")
    st.success("**🟢 Best Days** — Saturday and Sunday = highest revenue. Launch new products on weekends.")
    st.error("**🔴 March Drop Alert** — Revenue dropped 34% in March vs February. Needs urgent investigation.")

st.divider()
st.markdown("<p style='text-align:center;color:#2d3250;font-size:0.8rem'>Built with Python · Streamlit · Plotly · mlxtend · scikit-learn · Prophet · Random Forest</p>", unsafe_allow_html=True)