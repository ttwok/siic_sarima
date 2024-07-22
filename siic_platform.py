import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.dates as mdates
from datetime import datetime

# 데이터 로드
df = pd.read_csv('month_call_total/month_call_2024-07.csv')

if not df.empty:
    df = df[['ds', 'y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    df.index.freq = 'MS'  # 월별 데이터로 주기 설정

    # 로그 변환
    df_log = np.log(df)

    # 전체 데이터에 대한 모델 예측
    order = (1, 1, 1)
    seasonal_order = (2, 0, 1, 12)

    model_df_log = sm.tsa.SARIMAX(df_log, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)

    try:
        predict = model_df_log.fit(disp=False, maxiter=1000)  # 최대 반복 횟수 증가
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        predict = None

    if predict:
        predict_mean = np.exp(predict.get_forecast(7).predicted_mean)
        conf_int = predict.get_forecast(7).conf_int()
        conf_int_lb = np.exp(conf_int['lower y'])
        conf_int_ub = np.exp(conf_int['upper y'])

        # 전체 예측 결과와 실제 데이터 시각화 (고정)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df, color='blue', label='Actual Values')
        ax.plot(predict_mean, label=f'Predicted Values (model = (1,1,1),(2,0,1,12))', color='red', linestyle='--', alpha=0.5)
        ax.fill_between(predict_mean.index, conf_int_lb, conf_int_ub, color='red', alpha=0.1, label='95% Confidence Interval')

        # 실제 값의 숫자를 플롯에 추가 (정수로 표시)
        for x, y in zip(df.index, df['y']):
            ax.text(x, y, f'{int(y)}', color='blue', fontsize=8, ha='center', va='bottom')

        # 예측 값의 숫자를 플롯에 추가 (정수로 표시)
        for x, y in zip(predict_mean.index, predict_mean):
            ax.text(x, y, f'{int(y)}', color='red', fontsize=8, ha='center', va='bottom')

        # y축 범위 설정
        ax.set_ylim(20000, 120000)
        ax.legend(loc='upper right')
        ax.set_title('Predicted vs Actual Values')

        # 격자 설정
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

        st.pyplot(fig)

        # 최종 예측 날짜에서 -15개월과 +7개월 범위 계산
        max_date = df.index.max()
        min_date = max_date - pd.DateOffset(months=17)
        max_forecast_date = max_date + pd.DateOffset(months=7)

        # Interactive date range selectors
        start_year = st.selectbox('Start year', list(range(min_date.year, max_forecast_date.year + 1)), index=0)  # Default to min_date year
        start_month = st.selectbox('Start month', list(range(1, 13)), index=min_date.month - 1)  # Default to min_date month
        end_year = st.selectbox('End year', list(range(min_date.year, max_forecast_date.year + 1)), index=max_forecast_date.year - min_date.year)  # Default to max_forecast_date year
        end_month = st.selectbox('End month', list(range(1, 13)), index=max_forecast_date.month - 1)  # Default to max_forecast_date month
        start_date = pd.to_datetime(f'{start_year}-{start_month:02d}-01')
        end_date = pd.to_datetime(f'{end_year}-{end_month:02d}-01')

        if start_date > end_date:
            st.error("Error: End date must fall after start date.")
        else:
            # 필터링된 데이터
            df_filtered = df[start_date:end_date]

            # 필터링된 데이터의 y 값의 최소 및 최대 값을 계산
            y_min = min(df_filtered['y'].min(), predict_mean.min())
            y_max = max(df_filtered['y'].max(), predict_mean.max())
            y_range_margin = (y_max - y_min) * 0.1  # 여유를 두기 위해 10% 추가

            # 동적으로 업데이트되는 시각화
            fig_filtered, ax_filtered = plt.subplots(figsize=(14, 5))
            ax_filtered.plot(df_filtered, color='blue', label='Actual Values')
            ax_filtered.fill_between(predict_mean.index, conf_int_lb, conf_int_ub, color='red', alpha=0.1, label='95% Confidence Interval')
            
            # 필터링된 날짜 범위 내의 예측 값 필터링
            predict_mean_filtered = predict_mean[start_date:end_date]
            conf_int_lb_filtered = conf_int_lb[start_date:end_date]
            conf_int_ub_filtered = conf_int_ub[start_date:end_date]
            
            ax_filtered.plot(predict_mean_filtered, label=f'Predicted Values (model = (1,1,1),(2,0,1,12))', color='red', linestyle='--', alpha=0.5)

            # 실제 값의 숫자를 플롯에 추가 (정수로 표시)
            for x, y in zip(df_filtered.index, df_filtered['y']):
                ax_filtered.text(x, y, f'{int(y)}', color='blue', fontsize=12, ha='center', va='bottom', rotation=10)

            # 예측 값의 숫자를 플롯에 추가 (정수로 표시)
            for x, y in zip(predict_mean_filtered.index, predict_mean_filtered):
                ax_filtered.text(x, y, f'{int(y)}', color='red', fontsize=12, ha='center', va='bottom', rotation=10)

            # y축 범위 설정
            ax_filtered.set_ylim(y_min - y_range_margin, y_max + y_range_margin)
            ax_filtered.legend(loc='upper right')
            ax_filtered.set_title('Filtered Predicted vs Actual Values')

            # x축 레이블 설정
            ax_filtered.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax_filtered.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax_filtered.get_xticklabels(), rotation=45, ha='center')

            # 격자 설정
            ax_filtered.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

            st.pyplot(fig_filtered)
    else:
        st.error("Prediction model fitting failed.")
else:
    st.error("Failed to retrieve data from Google Sheets.")
