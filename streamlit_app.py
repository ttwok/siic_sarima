import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
import glob
from datetime import datetime

# Everything is accessible via the st.secrets dict:

st.write("DB username:", st.secrets["db_username"])
st.write("DB password:", st.secrets["db_password"])

# And the root-level secrets are also accessible as environment variables:

import os

st.write(
    "Has environment variables been set:",
    os.environ["db_username"] == st.secrets["db_username"],
)

# 콜건수 예측 코드
@st.cache_data
def load_data(folder_path):
    csv_files = glob.glob(f'{folder_path}/*.csv')

    def extract_date(file_path):
        file_name = file_path.split('/')[-1]
        date_str = file_name.split('_')[-1].split('.')[0]
        return datetime.strptime(date_str, '%Y-%m')

    latest_file = max(csv_files, key=extract_date)
    df = pd.read_csv(latest_file)
    df = df[['ds', 'y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    df.index.freq = 'MS'  # 월별 데이터로 주기 설정
    return df

@st.cache_data
def fit_model(df_log):
    order = (1, 1, 1)
    seasonal_order = (2, 0, 1, 12)
    model = sm.tsa.SARIMAX(df_log, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fit_result = model.fit(disp=False, maxiter=1000)
    return fit_result

def call_forecast():
    folder_path = 'month_call_total'
    df = load_data(folder_path)

    if not df.empty:
        df_log = np.log(df)

        try:
            predict = fit_model(df_log)
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            predict = None

        if predict:
            predict_mean = np.exp(predict.get_forecast(7).predicted_mean)
            conf_int = predict.get_forecast(7).conf_int()
            conf_int_lb = np.exp(conf_int['lower y'])
            conf_int_ub = np.exp(conf_int['upper y'])

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df, color='blue', label='Actual Values')
            ax.plot(predict_mean, label=f'Predicted Values (model = (1,1,1),(2,0,1,12))', color='red', linestyle='--', alpha=0.5)
            ax.fill_between(predict_mean.index, conf_int_lb, conf_int_ub, color='red', alpha=0.1, label='95% Confidence Interval')

            for x, y in zip(df.index, df['y']):
                ax.text(x, y, f'{int(y)}', color='blue', fontsize=8, ha='center', va='bottom')

            for x, y in zip(predict_mean.index, predict_mean):
                ax.text(x, y, f'{int(y)}', color='red', fontsize=8, ha='center', va='bottom')

            ax.set_ylim(20000, 120000)
            ax.legend(loc='upper right')
            ax.set_title('SIIC phone call : total values')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

            st.pyplot(fig)

            max_date = df.index.max()
            start_control_date = max_date - pd.DateOffset(months=17)
            min_date = max_date - pd.DateOffset(months=36)
            max_forecast_date = max_date + pd.DateOffset(months=0)

            if 'start_date' not in st.session_state:
                st.session_state.start_date = start_control_date

            start_date = st.slider('control month', min_date.date(), max_forecast_date.date(), value=st.session_state.start_date.date(), format="YYYY-MM")

            st.session_state.start_date = pd.to_datetime(start_date)

            df_filtered = df[st.session_state.start_date:max_date]

            y_min = min(df_filtered['y'].min(), predict_mean.min())
            y_max = max(df_filtered['y'].max(), predict_mean.max())
            y_range_margin = (y_max - y_min) * 0.1

            fig_filtered = go.Figure()

            fig_filtered.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['y'], mode='lines+markers+text', name='Actual Values',
                                              text=df_filtered['y'].astype(int), textposition='bottom center', textfont=dict(size=10)))

            fig_filtered.add_trace(go.Scatter(x=predict_mean.index, y=predict_mean, mode='lines+markers+text', name='Predicted Values',
                                              line=dict(dash='dash', color='red'), text=predict_mean.astype(int), textposition='top center', textfont=dict(size=10)))

            fig_filtered.add_trace(go.Scatter(x=predict_mean.index, y=conf_int_lb, fill=None, mode='lines', line_color='red', showlegend=False))
            fig_filtered.add_trace(go.Scatter(x=predict_mean.index, y=conf_int_ub, fill='tonexty', mode='lines', line_color='red', fillcolor='rgba(255, 0, 0, 0.1)', showlegend=False))

            fig_filtered.update_layout(
                title='forcasting M+7',
                yaxis=dict(range=[y_min - y_range_margin, y_max + y_range_margin]),
                xaxis=dict(tickformat='%Y-%m'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=40, r=40, t=20, b=40),
                hovermode='x unified',
                height=300
            )

            st.plotly_chart(fig_filtered)
        else:
            st.error("Prediction model fitting failed.")
    else:
        st.error("Failed to retrieve data from Google Sheets.")

# 사이드바에 탭 추가
st.sidebar.title('SIIC Dashboard')

# Main tabs
main_tabs = ["SIIC Management", "SIIC Reporting"]
main_selected_tab = st.sidebar.radio("Select a category", main_tabs)

if main_selected_tab == "SIIC Management":
    sub_tabs = ["SIIC 운영현황", "SIIC 운영실적", "SIIC 수요예측"]
    sub_selected_tab = st.sidebar.radio("SIIC Management", sub_tabs)
    
    if sub_selected_tab == "SIIC 운영현황":
        st.header("SIIC 운영현황")
        st.write("여기에 SIIC 운영현황에 대한 내용을 작성하세요.")

    elif sub_selected_tab == "SIIC 운영실적":
        st.header("SIIC 운영실적")
        st.write("여기에 SIIC 운영실적에 대한 내용을 작성하세요.")

    elif sub_selected_tab == "SIIC 수요예측":
        st.header("SIIC 콜 처리량 수요예측")
        call_forecast()

elif main_selected_tab == "SIIC Reporting":
    sub_tabs = ["SIIC 월간보고", "SIIC 일간보고"]
    sub_selected_tab = st.sidebar.radio("SIIC Reporting", sub_tabs)

    if sub_selected_tab == "SIIC 월간보고":
        st.header("SIIC 월간보고")
        st.write("여기에 SIIC 월간보고에 대한 내용을 작성하세요.")

    elif sub_selected_tab == "SIIC 일간보고":
        st.header("SIIC 일간보고")
        st.write("여기에 SIIC 일간보고에 대한 내용을 작성하세요.")
