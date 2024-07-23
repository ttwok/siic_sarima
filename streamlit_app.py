import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
import glob
from datetime import datetime

import plotly.express as px
from plotly.subplots import make_subplots

# 요일 이름 매핑
day_name_map = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}

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


    elif sub_selected_tab == "SIIC 운영실적":
        st.header("SIIC 운영실적")
        st.write("SIIC 콜 처리 현황입니다.")
        st.write(" :green[*raw data*]는 :blue[*csv 파일*]로 다운로드 가능합니다.")
        
        df_daycall = pd.read_csv('pusan_mall_2024-07-22.csv')
        df_daycall = df_daycall[df_daycall['총처리호'] != 0]
        
        # '날짜' 컬럼을 datetime 형식으로 변환
        df_daycall['날짜'] = pd.to_datetime(df_daycall['날짜'], format='%Y-%m-%d')
        
        # 요일 컬럼 추가
        df_daycall['요일'] = df_daycall['날짜'].dt.weekday.map(day_name_map)
        
        # 요일 컬럼을 순서 지정된 카테고리로 변환
        weekday_order = ['월요일', '화요일', '수요일', '목요일', '금요일']
        df_daycall['요일'] = pd.Categorical(df_daycall['요일'], categories=weekday_order, ordered=True)
        
        # 일자별 데이터 처리
        df_daycall_daily = df_daycall.groupby(['날짜', '요일'])['총처리호'].sum().reset_index()
        df_daycall_daily = df_daycall_daily[df_daycall_daily['총처리호'] != 0]  # 처리호가 0인 행 제거
        df_daycall_daily = df_daycall_daily.sort_values(by='날짜')
        
        # raw data
        # st.dataframe(df_daycall)
        
        # 탭 생성
        c1, c2, c3 = st.tabs(['전체 콜 현황','부서별', '업체별'])
        
        # 월별 선차트
        with c1:
            c1.info('2024년도 "콜 현황 조회" 입니다')
        
            df_daycall['월'] = df_daycall['날짜'].dt.to_period('M')
            df_month_call = df_daycall.groupby('월')['총처리호'].sum().reset_index()
            df_month_call['월'] = df_month_call['월'].astype(str)  # Period를 문자열로 변환
            df_month_call = df_month_call.sort_values(by='월', ascending=False)
        
            fig3 = px.bar(df_month_call, x='월', y='총처리호', title='월별 총 콜 처리현황')
            st.plotly_chart(fig3, use_container_width=True)
        
        
            fig2 = px.bar(df_daycall_daily, x='날짜', y='총처리호', title='일자별 총 콜 처리현황')
            st.plotly_chart(fig2, use_container_width=True)
        
            fig = px.line(df_daycall_daily, x='날짜', y='총처리호', color='요일', title='요일별 콜 처리현황')
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        
        
        mall_name = df_daycall['쇼핑몰명'].unique()
        team_name = df_daycall['팀명'].unique()
        
        with c2:
            st.info('부서별 "콜 상세 현황 조회" 입니다.')
            # 부서 선택 멀티셀렉트
            selected_teams = st.multiselect('조회 대상 부서를 선택하세요', [*team_name])
            # 선택한 부서의 데이터 필터링
            df_selected_teams = df_daycall[df_daycall['팀명'].isin(selected_teams)]
        
            # 서브 탭 생성
            t1, t2 = st.tabs(['월별', '일자별'])
        
            with t1:
                # 부서별 월별 데이터 집계
                df_selected_teams['월'] = df_selected_teams['날짜'].dt.to_period('M')
                df_monthly = df_selected_teams.groupby(['월', '팀명'])['총처리호'].sum().reset_index()
                df_monthly['월'] = df_monthly['월'].astype(str)  # Period를 문자열로 변환
        
                # 데이터프레임을 pivot하여 부서별 데이터 구성
                df_pivot_monthly = df_monthly.pivot_table(index='월', columns='팀명', values='총처리호', aggfunc='sum').reset_index()
        
                # 데이터프레임을 melt하여 긴 형식으로 변환
                df_melted_monthly = df_pivot_monthly.melt(id_vars='월', value_vars=df_pivot_monthly.columns[1:], var_name='부서', value_name='처리호')
                df_melted_monthly = df_melted_monthly.dropna()  # NaN 값 제거
        
                # 월별 부서별 그룹 바 차트 생성
                fig_bar_monthly = px.bar(df_melted_monthly, x='월', y='처리호', color='부서', barmode='group', title='월별 부서별 콜 처리현황')
                fig_bar_monthly.update_layout(yaxis_title='처리호', xaxis_title='월', xaxis={'categoryorder': 'category ascending'}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_bar_monthly, use_container_width=True)
        
                # 월별 부서별 누적 바 차트 생성
                fig_stack_bar_monthly = px.bar(df_melted_monthly, x='월', y='처리호', color='부서', barmode='stack', title='월별 부서별 콜 처리현황 (누적)')
                fig_stack_bar_monthly.update_layout(yaxis_title='처리호', xaxis_title='월', xaxis={'categoryorder': 'category ascending'}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_stack_bar_monthly, use_container_width=True)
        
                # 결과 데이터프레임 출력
                st.write('*raw data*')
                st.dataframe(df_pivot_monthly)
        
            with t2:
                # 마지막 데이터 일자에서부터 14일간의 데이터만 필터링
                last_date = df_selected_teams['날짜'].max()
                start_date = last_date - pd.Timedelta(days=10)
                df_filtered = df_selected_teams[(df_selected_teams['날짜'] >= start_date) & (df_selected_teams['날짜'] <= last_date)]
        
                # 날짜별, 팀명별 총처리호 집계
                df_pivot = df_filtered.pivot_table(index='날짜', columns='팀명', values='총처리호', aggfunc='sum').reset_index()
        
                # 날짜를 내림차순으로 정렬
                df_pivot = df_pivot.sort_values(by='날짜', ascending=False)
        
                # 데이터프레임을 melt하여 긴 형식으로 변환
                df_melted = df_pivot.melt(id_vars='날짜', value_vars=df_pivot.columns[1:], var_name='부서', value_name='처리호')
                df_melted = df_melted.dropna()  # NaN 값 제거
        
                # 일자별 부서별 그룹 바 차트 생성
                fig_bar = px.bar(df_melted, x='날짜', y='처리호', color='부서', barmode='group', title='일자별 부서별 콜 처리현황')
                fig_bar.update_layout(yaxis_title='처리호', xaxis_title='날짜', xaxis={'categoryorder':'category ascending'}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_bar, use_container_width=True)
        
                # 영역형 차트 추가
                fig_area = px.area(df_melted, x='날짜', y='처리호', color='부서', title='일자별 부서별 콜 처리 현황 (영역형)')
                fig_area.update_layout(yaxis_title='처리호', xaxis_title='날짜', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_area, use_container_width=True)
        
                # 결과 데이터프레임 출력
                st.write('*raw data*')
                st.dataframe(df_pivot)
        
        
        with c3:
            st.info('업체별 "콜 상세 현황 조회" 입니다.')
            selected_malls = st.multiselect('업체를 선택하세요', [*mall_name])
            df_selected_malls = df_daycall[df_daycall['쇼핑몰명'].isin(selected_malls)]
            
            # 월별 추이
            df_selected_malls['월'] = df_selected_malls['날짜'].dt.to_period('M')
            df_monthly_mall = df_selected_malls.groupby(['월', '쇼핑몰명'])['총처리호'].sum().reset_index()
            df_monthly_mall['월'] = df_monthly_mall['월'].astype(str)  # Period를 문자열로 변환
        
            fig_mall_monthly = px.line(df_monthly_mall, x='월', y='총처리호', color='쇼핑몰명', title='월별 업체별 콜 처리 추이')
            fig_mall_monthly.update_layout(yaxis_title='처리호', xaxis_title='월', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_mall_monthly, use_container_width=True)
        
            # 일별 현황
            last_date_mall = df_selected_malls['날짜'].max()
            start_date_mall = last_date_mall - pd.Timedelta(days=90)
            df_daily_mall = df_selected_malls[(df_selected_malls['날짜'] >= start_date_mall) & (df_selected_malls['날짜'] <= last_date_mall)]
        
            fig_mall_daily = px.area(df_daily_mall, x='날짜', y='총처리호', color='쇼핑몰명', title='일별 업체별 콜 처리 현황 (90일)')
            fig_mall_daily.update_layout(yaxis_title='처리호', xaxis_title='날짜', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_mall_daily, use_container_width=True)
        
            # raw data (월별)
            st.write('*raw data (월별)*')
            st.dataframe(df_monthly_mall)
        
            # raw data (일별)
            st.write('*raw data (일별)*')
            st.dataframe(df_daily_mall[['날짜','쇼핑몰명','총처리호']])

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
