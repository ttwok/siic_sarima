import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

import gspread
from google.oauth2.service_account import Credentials

# 구글 스프레드시트에 연결하여 데이터를 가져오는 함수
def get_google_sheet_data(sheet_url, sheet_name):
    # 서비스 계정 인증 정보 및 스코프 설정
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    creds = Credentials.from_service_account_file('client_secret.json', scopes=SCOPES)
    
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).worksheet(sheet_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# 구글 스프레드시트 URL 및 시트 이름
sheet_url = 'https://docs.google.com/spreadsheets/d/15YI4kOklCi5VqPNOyR5KyQ-aBOluLQ25CNsdnscF3_A/edit?gid=0#gid=0'
sheet_name = 'month_call_total'

st.title('Month Call Total Analysis')

df = get_google_sheet_data(sheet_url, sheet_name)
if not df.empty:
    df = df[['ds', 'y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    df.index.freq = 'MS'  # 월별 데이터로 주기 설정

    df_log = np.log(df)
    train_log, test_log = train_test_split(df_log, test_size=0.2, shuffle=False)

    # 모델 생성 및 학습
    order = (1, 1, 1)
    seasonal_order = (2, 0, 1, 12)
    
    # 초기 파라미터 설정
    start_params = [0.1] * (order[0] + order[1] + order[2] + seasonal_order[0] + seasonal_order[1] + seasonal_order[2] + 1)
    
    model_mape = sm.tsa.SARIMAX(train_log, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    
    try:
        history_mape = model_mape.fit(start_params=start_params, disp=False)
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        history_mape = None

    if history_mape:
        # 예측
        pred_mape = history_mape.get_forecast(steps=len(test_log)).predicted_mean

        # 신뢰 구간
        conf_int_mape = history_mape.get_forecast(steps=len(test_log)).conf_int()
        pred_mape_lb = np.exp(conf_int_mape['lower y'])
        pred_mape_ub = np.exp(conf_int_mape['upper y'])

        # 예측 값 시리즈 생성
        pred_values_mape = pred_mape.values
        pred_index = test_log.index[-len(pred_values_mape):]
        pred_series_mape = pd.Series(pred_values_mape, index=pred_index)
        pred_series_mape = np.exp(pred_series_mape.dropna())
        test_log_clean = np.exp(test_log.dropna())

        # 전체 데이터에 대한 모델 예측
        model_df_log = sm.tsa.SARIMAX(df_log, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        try:
            predict = model_df_log.fit(start_params=start_params, disp=False)
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            predict = None

        if predict:
            predict_mean = np.exp(predict.get_forecast(7).predicted_mean)
            conf_int = predict.get_forecast(7).conf_int()
            conf_int_lb = np.exp(conf_int['lower y'])
            conf_int_ub = np.exp(conf_int['upper y'])

            # 시각화
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(predict_mean, label=f'model = (1,1,1),(2,0,1,12)', color='r', linestyle='--', alpha=0.5)
            ax.fill_between(predict_mean.index, conf_int_lb, conf_int_ub, color='r', alpha=0.1, label='95% Confidence Interval')
            ax.plot(df, color='gray')

            # 예측 값의 숫자를 플롯에 추가 (정수로 표시)
            for x, y in zip(predict_mean.index, predict_mean):
                ax.text(x, y, f'{int(y)}', color='r', fontsize=8, ha='center', va='bottom')

            # y축 범위 설정
            ax.set_ylim(20000, 120000)

            ax.legend(loc='upper left')
            ax.set_title('Predicted vs Actual Values')
            
            st.pyplot(fig)
        else:
            st.error("Prediction model fitting failed.")
    else:
        st.error("Initial model fitting failed.")
else:
    st.error("Failed to retrieve data from Google Sheets.")
