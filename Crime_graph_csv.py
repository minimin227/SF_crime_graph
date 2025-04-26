import pandas as pd 
import streamlit as st
import plotly.express as px
import os
import subprocess
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import gdown
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

@st.cache_data
def load_data():
    file_id = "14j0NrXfHmGl3jLGDaUKL3KSJX6ab0lvH"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path = "train.csv"

    # 파일 없으면 다운로드
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    df = pd.read_csv(output_path)
    return df

# 데이터 집계 함수 (캐싱)
# @st.cache_data
# def load_data():
#     data_folder = 'Data'
#     df = pd.read_csv(os.path.join(data_folder, '/home/uho317/SQL_project/Data/train.csv'))
#     return df

# 데이터 로딩
df = load_data()
SEVERITY_RULES = {
    3.5: ["homicide", "attempted homicide", "armed with a gun", "rape", "carjacking", "aggravated assault"],
    3.0: ["armed", "deadly weapon", "forcible", "resisting arrest", "kidnapping", "bank robbery"],
    2.5: ["burglary", "grand theft", "narcotics", "extortion", "domestic violence"],
    2.0: ["probation violation", "trespassing", "drunk", "vandalism", "disorderly", "forgery", "possession"],
    1.5: ["petty theft", "shoplifting", "traffic", "suspicious", "loitering", "lost property"],
}
@st.cache_data
def preprocess_descript_features(df):
    # Dates열 분리
    df['Dates'] = pd.to_datetime(df['Dates'])
    df['Year'] = df['Dates'].dt.year
    df['Month'] = df['Dates'].dt.month
    df['Day'] = df['Dates'].dt.day
    df['Hour'] = df['Dates'].dt.hour

    # 대분류 카테고리 매핑
    category_map = {
        "ASSAULT": "Violent",
        "ROBBERY": "Violent",
        "KIDNAPPING": "Violent",
        "SUICIDE": "Violent",
        "SEX OFFENSES FORCIBLE": "Violent",

        "LARCENY/THEFT": "Property",
        "BURGLARY": "Property",
        "VEHICLE THEFT": "Property",
        "VANDALISM": "Property",
        "STOLEN PROPERTY": "Property",
        "FRAUD": "Property",
        "ARSON": "Property",
        "FORGERY/COUNTERFEITING": "Property",
        "BAD CHECKS": "Property",
        "EXTORTION": "Property",
        "EMBEZZLEMENT": "Property",

        "DRUG/NARCOTIC": "Substance",
        "DRUNKENNESS": "Substance",
        "LIQUOR LAWS": "Substance",
        "DRIVING UNDER THE INFLUENCE": "Substance",

        "PROSTITUTION": "Sexual",
        "SEX OFFENSES NON FORCIBLE": "Sexual",
        "PORNOGRAPHY/OBSCENE MAT": "Sexual",

        "DISORDERLY CONDUCT": "Behavior",
        "LOITERING": "Behavior",
        "TRESPASS": "Behavior",
        "FAMILY OFFENSES": "Behavior",
        "RUNAWAY": "Behavior",
        "SUSPICIOUS OCC": "Behavior",
        "GAMBLING": "Behavior",

        "WARRANTS": "Legal",
        "OTHER OFFENSES": "Legal",
        "SECONDARY CODES": "Legal",
        "WEAPON LAWS": "Legal",
        "BRIBERY": "Legal",
        "NON-CRIMINAL": "Legal",
        "RECOVERED VEHICLE": "Legal",
        "TREA": "Other",

        "MISSING PERSON": "Other"
    }
    df['L_Category'] = df['Category'].map(category_map)
    # resolution_score_mapping
    resolution_scores = {
        'ARREST, BOOKED': 3.0,
        'JUVENILE BOOKED': 3.0,
        'ARREST, CITED': 2.5,
        'JUVENILE CITED': 2.5,
        'PROSECUTED FOR LESSER OFFENSE': 2.5,
        'DISTRICT ATTORNEY REFUSES TO PROSECUTE': 2.0,
        'PROSECUTED BY OUTSIDE AGENCY': 2.0,
        'JUVENILE DIVERTED': 2.0,
        'EXCEPTIONAL CLEARANCE': 2.0,
        'NOT PROSECUTED': 1.5,
        'COMPLAINANT REFUSES TO PROSECUTE': 1.5,
        'JUVENILE ADMONISHED': 1.5,
        'CLEARED-CONTACT JUVENILE FOR MORE INFO': 1.5,
        'NONE': 1.0,
        'UNFOUNDED': 1.0,
        'PSYCHOPATHIC CASE': 1.0,
        'LOCATED': 1.0,
    }
    df['ResolutionScore'] = df['Resolution'].map(resolution_scores).fillna(1.5) 
    # 범죄심각도 수치화

    def auto_score(text: str) -> float:
        text = text.lower()
        for score, keywords in SEVERITY_RULES.items():
            for keyword in keywords:
                if keyword in text:
                    return score
        return None # 기본값
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(df['Descript'].tolist())

    # 클러스터링
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(X)

    # 규칙 기반 스코어 부여
    df['Score'] = df['Descript'].apply(auto_score)
    df['Cluster'] = clusters

    # 클러스터별 평균 점수 계산
    cluster_scores = df.dropna(subset=['Score']).groupby('Cluster')['Score'].mean()

    # 결측 점수 보정
    def fill_missing_score(row):
        if pd.isna(row['Score']):
            return cluster_scores.get(row['Cluster'], 2.0)
        return row['Score']

    df['FinalScore'] = df.apply(fill_missing_score, axis=1)

    return df
# 데이터 로딩
df = load_data()

# 스코어 및 클러스터링 전처리 적용
df = preprocess_descript_features(df)


L_Category_list = df['L_Category'].unique().tolist()
Category_list = df['Category'].unique().tolist()
PdDistrict_list = df['PdDistrict'].unique().tolist()
Year_list = df['Year'].unique().tolist()
Month_list = df['Month'].unique().tolist()
Day_list = df['Day'].unique().tolist()
Hour_list = df['Hour'].unique().tolist()
Weekday_list = df['DayOfWeek'].unique().tolist()

# L_Category 선택
select_all_lcat = st.checkbox("전체 대분류(L_Category) 선택")
selected_lcat = st.multiselect(
    "대분류(L_Category) 선택", L_Category_list,
    default=L_Category_list if select_all_lcat else []
)
# Category 선택
select_all_cat = st.checkbox("전체 범죄유형(Category) 선택")
selected_cat = st.multiselect(
    "범죄유형(Category) 선택", Category_list,
    default=Category_list if select_all_cat else []
)
# 경찰서 관할구 선택
select_all_pd = st.checkbox("전체 관할구(PdDistrict) 선택")
selected_pd = st.multiselect(
    "관할구(PdDistrict) 선택", PdDistrict_list,
    default=PdDistrict_list if select_all_pd else []
)
# 연도 선택
select_all_year = st.checkbox("전체 연도(Year) 선택")
selected_year = st.multiselect(
    "연도(Year) 선택", sorted(Year_list),
    default=Year_list if select_all_year else []
)
# 월 선택
select_all_month = st.checkbox("전체 월(Month) 선택")
selected_month = st.multiselect(
    "월(Month) 선택", sorted(Month_list),
    default=Month_list if select_all_month else []
)
# 일 선택
select_all_day = st.checkbox("전체 일(Day) 선택")
selected_day = st.multiselect(
    "일(Day) 선택", sorted(Day_list),
    default=Day_list if select_all_day else []
)
# 시간 선택
select_all_hour = st.checkbox("전체 시간(Hour) 선택")
selected_hour = st.multiselect(
    "시간(Hour) 선택", sorted(Hour_list),
    default=Hour_list if select_all_hour else []
)
# 요일 선택
select_all_weekday = st.checkbox("전체 요일(Weekday) 선택")
selected_weekday = st.multiselect(
    "요일(Weekday) 선택", Weekday_list,
    default=Weekday_list if select_all_weekday else []
)

import streamlit as st
import pandas as pd

# 필터 함수 정의
@st.cache_data
def filter_crime_data(df, selected_lcat, selected_cat, selected_pd, selected_year,
                      selected_month, selected_day, selected_hour):
    filtered_df = df.copy()
    selected_columns = ['L_Category', 'Category', 'PdDistrict', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek']

    if selected_lcat:
        filtered_df = filtered_df[filtered_df['L_Category'].isin(selected_lcat)]
    else:
        selected_columns.remove('L_Category')

    if selected_cat:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_cat)]
    else:
        selected_columns.remove('Category')

    if selected_pd:
        filtered_df = filtered_df[filtered_df['PdDistrict'].isin(selected_pd)]
    else:
        selected_columns.remove('PdDistrict')

    if selected_year:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_year)]
    else:
        selected_columns.remove('Year')

    if selected_month:
        filtered_df = filtered_df[filtered_df['Month'].isin(selected_month)]
    else:
        selected_columns.remove('Month')

    if selected_day:
        filtered_df = filtered_df[filtered_df['Day'].isin(selected_day)]
    else:
        selected_columns.remove('Day')

    if selected_hour:
        filtered_df = filtered_df[filtered_df['Hour'].isin(selected_hour)]
    else:
        selected_columns.remove('Hour')

    if selected_weekday:
        filtered_df = filtered_df[filtered_df['DayOfWeek'].isin(selected_weekday)]
    else:
        selected_columns.remove('DayOfWeek')

    return filtered_df, selected_columns

# 위에서 사용자가 선택한 필터 입력값들을 바탕으로
filtered_df, selected_columns = filter_crime_data(
    df,
    selected_lcat, selected_cat, selected_pd,
    selected_year, selected_month, selected_day, selected_hour
)

# 그룹화 및 정규화된 위험지수 계산
try:
    if len(selected_columns) == 0:
        st.warning("선택된 필터가 없어 그룹화할 수 없습니다.")
        df_group = pd.DataFrame()
    else:
        df_group = filtered_df.groupby(selected_columns).agg(
            Counts=('FinalScore', 'count'),
            Severity_sum=('FinalScore', 'sum'),
            Severity_mean=('FinalScore', 'mean'),
            ResolutionScore_sum=('ResolutionScore', 'sum'),
            ResolutionScore_mean=('ResolutionScore', 'mean'),
        ).reset_index()

except Exception as e:
    st.error(f"그룹화 중 오류 발생: {e}")
    df_group = pd.DataFrame()


# 시각화 설정 옵션 제공
st.subheader("그래프 설정")
columns_for_x_and_color = ['없음', 'L_Category', 'Category', 'PdDistrict', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek']
metrics = ['Counts', 'Severity_sum', 'Severity_mean', 'ResolutionScore_sum', 'ResolutionScore_mean']
graph_types = ['Bar']

# 사용자 선택 입력
metric = st.multiselect('그래프를 표시할 값 선택', metrics, default=['Counts'])
x_axis = st.selectbox('X축 선택', columns_for_x_and_color[1:], index=0)
color_axis = st.selectbox('Color 기준 선택', columns_for_x_and_color, index=0)
graph_type = st.selectbox('그래프 유형 선택', graph_types, index=0)


if st.button('그래프 생성하기'):
    if len(metric) >= 1: 
        rows = len(metric)
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=metric
        )

    if len(metric) >= 1 and not df_group.empty:
        rows = len(metric)
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=metric
        )

        for i, m in enumerate(metric):
            if color_axis == '없음':
                # 색상 기준 없음
                fig.add_trace(
                    go.Bar(x=df_group[x_axis], y=df_group[m], name=m),
                    row=i+1, col=1
                )
            else:
                # 색상 기준 있음
                for category in df_group[color_axis].dropna().unique():
                    df_filtered = df_group[df_group[color_axis] == category]
                    fig.add_trace(
                        go.Bar(
                            x=df_filtered[x_axis],
                            y=df_filtered[m],
                            name=str(category),
                            showlegend=(i == 0)  # 첫 row에만 범례 표시
                        ),
                        row=i+1, col=1
                    )
            fig.update_yaxes(title_text=m, row=i+1, col=1)

        barmode = st.radio(
            "막대그래프 표시 방식 선택",
            ("group", "overlay"),
            index=0
        )

        fig.update_layout(height=500 * rows, title_text=f"{x_axis} 기준 지표별 Subplot 비교", barmode=barmode)
        st.plotly_chart(fig)
    else:
        st.warning("1개 이상의 지표를 선택하고, 필터링된 데이터가 있어야 시각화할 수 있습니다.")


 
# 히트맵 시각화 섹션
st.subheader("피벗 테이블 히트맵")
if st.button('히트맵 생성하기'):
    if x_axis != '없음' and color_axis != '없음' and len(metric) == 1:
        pivot_metric = metric[0]

        try:
            pivot_df = df_group.pivot_table(
                index=color_axis,
                columns=x_axis,
                values=pivot_metric,
                aggfunc='mean'
            )

            fig_heatmap = px.imshow(
                pivot_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels=dict(color=pivot_metric),
                title=f"{pivot_metric}에 대한 히트맵 ({color_axis} vs {x_axis})"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

        except Exception as e:
            st.warning(f"피벗 테이블 생성 중 오류 발생: {e}")
    else:
        st.info("히트맵은 X축, Color 기준이 모두 선택되고 통계값(metric)이 1개일 때만 표시됩니다.")

    if x_axis != '없음' and color_axis != '없음' and len(metric) == 1:
        pivot_metric = metric[0]

        try:
            pivot_df = df_group.pivot_table(
                index=color_axis,
                columns=x_axis,
                values=pivot_metric,
                aggfunc='mean'
            )

            # 행(row)별 정규화 (각 row를 0~1 범위로)
            normalized_pivot_df = pivot_df.copy()
            normalized_pivot_df = normalized_pivot_df.apply(
                lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) if np.nanmax(x) != np.nanmin(x) else 0,
                axis=1
            )

            fig_heatmap = px.imshow(
                normalized_pivot_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels=dict(color=f"{pivot_metric} (Normalized)"),
                title=f"{pivot_metric}에 대한 히트맵 (행별 정규화, {color_axis} vs {x_axis})"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

        except Exception as e:
            st.error(f"히트맵 생성 중 오류 발생: {e}")

    if x_axis != '없음' and color_axis != '없음' and len(metric) == 1:
        pivot_metric = metric[0]

        try:
            pivot_df = df_group.pivot_table(
                index=color_axis,
                columns=x_axis,
                values=pivot_metric,
                aggfunc='mean'
            )

            # 행(row)별 정규화 (각 row를 0~1 범위로)
            normalized_pivot_df = pivot_df.copy()
            normalized_pivot_df = normalized_pivot_df.apply(
                lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) if np.nanmax(x) != np.nanmin(x) else 0,
                axis=0
            )

            fig_heatmap = px.imshow(
                normalized_pivot_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels=dict(color=f"{pivot_metric} (Normalized)"),
                title=f"{pivot_metric}에 대한 히트맵 (열별 정규화, {color_axis} vs {x_axis})"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

        except Exception as e:
            st.error(f"히트맵 생성 중 오류 발생: {e}")

st.dataframe(df_group, use_container_width=True) 



# 지도 표시 여부를 Streamlit 세션 상태로 관리
# if 'show_map' not in st.session_state:
#     st.session_state['show_map'] = False

# # 지도 On/Off 토글 버튼
# if st.button("지도 보기 / 숨기기"):
#     st.session_state['show_map'] = not st.session_state['show_map']

# # 현재 상태에 따라 지도 표시
# if st.session_state['show_map']:
#     st.subheader("선택된 범죄 위치 지도")

#     if not filtered_df.empty:
#         center_lat = filtered_df['Y'].mean()
#         center_lon = filtered_df['X'].mean()
#     else:
#         center_lat, center_lon = 37.77, -122.42  # 샌프란시스코 기본값

#     m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

#     for idx, row in filtered_df.iterrows():
#         if pd.notnull(row['Y']) and pd.notnull(row['X']):
#             popup_text = f"Category: {row['Category']}<br>Resolution: {row['Resolution']}"
#             folium.CircleMarker(
#                 location=[row['Y'], row['X']],
#                 radius=3,
#                 color='blue',
#                 fill=True,
#                 fill_color='blue',
#                 popup=popup_text
#             ).add_to(m)

#     st_folium(m, width=700, height=500)
# else:
#     st.info("지도는 현재 숨겨져 있습니다.")

# 지도 표시 여부를 Streamlit 세션 상태로 관리
if 'show_map' not in st.session_state:
    st.session_state['show_map'] = False

# 지도 On/Off 토글 버튼
if st.button("지도 보기 / 숨기기"):
    st.session_state['show_map'] = not st.session_state['show_map']

# 현재 상태에 따라 지도 표시
if st.session_state['show_map']:
    st.subheader("선택된 범죄 위치 지도")

    if not filtered_df.empty:
        center_lat = filtered_df['Y'].mean()
        center_lon = filtered_df['X'].mean()
    else:
        center_lat, center_lon = 37.77, -122.42  # 샌프란시스코 기본값

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # --- 색상 매핑 준비
    if color_axis != '없음' and color_axis in filtered_df.columns:
        unique_values = filtered_df[color_axis].dropna().unique()
        colormap = cm.get_cmap('plasma')

        color_mapping = {val: mcolors.to_hex(colormap(i / len(unique_values))) for i, val in enumerate(unique_values)}
    else:
        color_mapping = {}

    # --- 지도에 점 추가
    for idx, row in filtered_df.iterrows():
        if pd.notnull(row['Y']) and pd.notnull(row['X']):
            popup_text = f"Category: {row['Category']}<br>Resolution: {row['Resolution']}"

            # 색상 결정
            color = color_mapping.get(row[color_axis], 'blue') if color_axis != '없음' else 'blue'

            folium.CircleMarker(
                location=[row['Y'], row['X']],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                popup=popup_text
            ).add_to(m)

    st_folium(m, width=700, height=500)
else:
    st.info("지도는 현재 숨겨져 있습니다.")