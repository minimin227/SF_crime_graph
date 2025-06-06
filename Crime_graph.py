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
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pymysql
import geopandas as gpd
import pydeck as pdk

st.set_page_config(layout="wide")

DB_CONFIG = {
    "host": st.secrets["DB_HOST"],
    "port": int(st.secrets["DB_PORT"]),
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "db": st.secrets["DB_NAME"],
    "charset": "utf8mb4"
}

# --- 데이터 로딩 함수 ---
@st.cache_data
def load_sql_data(sql):
    try:
        conn = pymysql.connect(**DB_CONFIG)
        df = pd.read_sql(sql, conn)
        conn.close()
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"❌ SQL 실행 오류: {e}")
        return pd.DataFrame()

@st.cache_data
def load_severity_mapping():
    file_id = "1FVDEr7SLX5uYZgPB7aTlez6_hI5cVIv0"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path = "Descript_Severity.csv"

    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    return pd.read_csv(output_path)

@st.cache_data
def preprocess_data(df, severity_df):
    # 필수 컬럼 확인
    required_cols = ['dates', 'category', 'resolution']
    if not all(col in df.columns for col in required_cols):
        return None  # 전처리 불가

    # dates 처리
    df['dates'] = pd.to_datetime(df['dates'])
    df['year'] = df['dates'].dt.year
    df['month'] = df['dates'].dt.month
    df['day'] = df['dates'].dt.day
    df['hour'] = df['dates'].dt.hour

    # 대분류 매핑
    category_map = {
        "ASSAULT": "Violent",
        "ROBBERY": "Violent",
        "KIDNAPPING": "Violent",
        "SUICIDE": "Violent",
        
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

        "SEX OFFENSES FORCIBLE": "Sexual",
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
    df['l_category'] = df['category'].map(category_map)

    # # resolution Score 매핑
    # resolution_scores = {
    #     'NONE': 1.0, 'UNFOUNDED': 5, 'EXCEPTIONAL CLEARANCE': 4.5, 'CLEARED-CONTACT JUVENILE FOR MORE INFO': 4.5
    #     , 'NOT PROSECUTED': 3.5, 'DISTRICT ATTORNEY REFUSES TO PROSECUTE': 3.5, 'PROSECUTED FOR LESSER OFFENSE': 4
    # }
    # df['resolution'] = df['resolution'].map(resolution_scores).fillna(5)

    # Descript 정리 후 Severity 매핑
    # df['Descript'] = df['Descript'].str.strip().replace(r'\s+', ' ', regex=True)
    # df = df.merge(severity_df, on='dates', how='left')
    # df = df.drop(columns=['Descript'])
    # df['severity_per_resolution'] = df['Severity_Score']/df['resolution']
    df = df.loc[:, ~df.columns.duplicated()]

    return df

# --- 메인 앱 시작 ---
st.title("🚔 San Francisco Crime 데이터 분석")

# --- SQL 입력 창 ---
st.sidebar.header("🔎 SQL 쿼리 입력")
st.sidebar.write("SELECT t.*\n, r.resolution_score\n, d.severity_score\n, d.severity_score/r.resolution_score AS severity_per_resolution \nFROM train t \nJOIN resolution_score r \nON t.resolution=r.resolution \nJOIN descript_severity d \nON t.d_code=d.d_code \nWHERE dates BETWEEN '2015-01-01' AND '2015-05-14';")
default_sql = "SELECT t.*\n, r.resolution_score\n, d.severity_score\n, d.severity_score/r.resolution_score AS severity_per_resolution \nFROM train t \nJOIN resolution_score r \nON t.resolution=r.resolution \nJOIN descript_severity d \nON t.d_code=d.d_code \nWHERE dates BETWEEN '2015-01-01' AND '2015-05-14';"
user_sql = st.sidebar.text_area("SQL 입력:", default_sql, height=150)

# --- 데이터 로드 및 캐싱 ---
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

if st.sidebar.button("데이터 불러오기"):
    df_raw = load_sql_data(user_sql)
    if df_raw.empty:
        st.warning("❌ 데이터가 없습니다.")
    else:
        severity_df = load_severity_mapping()
        processed_df = preprocess_data(df_raw, severity_df)

        if processed_df is not None:
            st.session_state["df"] = processed_df
            st.success(f"✅ {len(processed_df):,}건 전처리 완료")
        else:
            st.session_state["df"] = df_raw
            st.info("⚡ 전처리 없이 원본 데이터 표시 (필수 컬럼 부족)")

# --- 데이터 출력 ---
df = st.session_state["df"]

if df.empty:
    st.warning("데이터를 먼저 불러오세요.")
    st.stop()

st.subheader("📄 데이터 미리보기")
st.dataframe(df, use_container_width=True)

L_Category_list = df['l_category'].unique().tolist()
Category_list = df['category'].unique().tolist()
PdDistrict_list = df['pddistrict'].unique().tolist()
Year_list = df['year'].unique().tolist()
Month_list = df['month'].unique().tolist()
Day_list = df['day'].unique().tolist()
Hour_list = df['hour'].unique().tolist()
Weekday_list = df['dayofweek'].unique().tolist()

# l_category 선택
select_all_lcat = st.checkbox("전체 대분류(l_category) 선택")
selected_lcat = st.multiselect(
    "대분류(l_category) 선택", L_Category_list,
    default=L_Category_list if select_all_lcat else []
)
# category 선택
select_all_cat = st.checkbox("전체 범죄유형(category) 선택")
selected_cat = st.multiselect(
    "범죄유형(category) 선택", Category_list,
    default=Category_list if select_all_cat else []
)
# 경찰서 관할구 선택
select_all_pd = st.checkbox("전체 관할구(pddistrict) 선택")
selected_pd = st.multiselect(
    "관할구(pddistrict) 선택", PdDistrict_list,
    default=PdDistrict_list if select_all_pd else []
)
# 연도 선택
select_all_year = st.checkbox("전체 연도(year) 선택")
selected_year = st.multiselect(
    "연도(year) 선택", sorted(Year_list),
    default=Year_list if select_all_year else []
)
# 월 선택
select_all_month = st.checkbox("전체 월(month) 선택")
selected_month = st.multiselect(
    "월(month) 선택", sorted(Month_list),
    default=Month_list if select_all_month else []
)
# 일 선택
select_all_day = st.checkbox("전체 일(day) 선택")
selected_day = st.multiselect(
    "일(day) 선택", sorted(Day_list),
    default=Day_list if select_all_day else []
)
# 시간 선택
select_all_hour = st.checkbox("전체 시간(hour) 선택")
selected_hour = st.multiselect(
    "시간(hour) 선택", sorted(Hour_list),
    default=Hour_list if select_all_hour else []
)
# 요일 선택
select_all_weekday = st.checkbox("전체 요일(Weekday) 선택")
selected_weekday = st.multiselect(
    "요일(Weekday) 선택", Weekday_list,
    default=Weekday_list if select_all_weekday else []
)

# 필터 함수 정의
@st.cache_data
def filter_crime_data(df, selected_lcat, selected_cat, selected_pd, selected_year,
                      selected_month, selected_day, selected_hour):
    filtered_df = df.copy()
    selected_columns = ['l_category', 'category', 'pddistrict', 'year', 'month', 'day', 'hour', 'dayofweek']

    if selected_lcat:
        filtered_df = filtered_df[filtered_df['l_category'].isin(selected_lcat)]
    else:
        selected_columns.remove('l_category')

    if selected_cat:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_cat)]
    else:
        selected_columns.remove('category')

    if selected_pd:
        filtered_df = filtered_df[filtered_df['pddistrict'].isin(selected_pd)]
    else:
        selected_columns.remove('pddistrict')

    if selected_year:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_year)]
    else:
        selected_columns.remove('year')

    if selected_month:
        filtered_df = filtered_df[filtered_df['month'].isin(selected_month)]
    else:
        selected_columns.remove('month')

    if selected_day:
        filtered_df = filtered_df[filtered_df['day'].isin(selected_day)]
    else:
        selected_columns.remove('day')

    if selected_hour:
        filtered_df = filtered_df[filtered_df['hour'].isin(selected_hour)]
    else:
        selected_columns.remove('hour')

    if selected_weekday:
        filtered_df = filtered_df[filtered_df['dayofweek'].isin(selected_weekday)]
    else:
        selected_columns.remove('dayofweek')

    return filtered_df, selected_columns

# 위에서 사용자가 선택한 필터 입력값들을 바탕으로
filtered_df, selected_columns = filter_crime_data(
    df,
    selected_lcat, selected_cat, selected_pd,
    selected_year, selected_month, selected_day, selected_hour
)

# try:
#     if len(selected_columns) == 0:
#         st.warning("선택된 필터가 없어 그룹화할 수 없습니다.")
#         df_group = pd.DataFrame()
#     else:
#         df_group = filtered_df.groupby(selected_columns).agg(
#             Counts=('severity_score', 'count'),
#             Severity_sum=('severity_score', 'sum'),
#             Severity_mean=('severity_score', 'mean'),
#             Resolution_sum=('resolution_score', 'sum'),
#             Resolution_mean=('resolution_score', 'mean'),
#             severity_per_resolution_sum=('severity_per_resolution', 'sum'),
#             severity_per_resolution_mean=('severity_per_resolution', 'mean'),
#         ).reset_index()

# except Exception as e:
#     st.error(f"그룹화 중 오류 발생: {e}")
#     df_group = pd.DataFrame()

try:
    if len(selected_columns) == 0:
        st.warning("선택된 필터가 없어 그룹화할 수 없습니다.")
        df_group = pd.DataFrame()
    else:
        # --- 동적 agg_dict 생성 ---
        agg_dict = {}
        if 'severity_score' in filtered_df.columns:
            agg_dict.update({
                'severity_score': ['sum', 'mean']
            })
        if 'resolution_score' in filtered_df.columns:
            agg_dict.update({
                'resolution_score': ['sum', 'mean']
            })
        if 'severity_per_resolution' in filtered_df.columns:
            agg_dict.update({
                'severity_per_resolution': ['sum', 'mean']
            })
        if 'dates' in filtered_df.columns:
            agg_dict.update({
                'dates': ['count']
            })
        if agg_dict:
            # 그룹화
            df_group = filtered_df.groupby(selected_columns).agg(agg_dict)

            # 컬럼 다중인덱스(flatten)
            df_group.columns = [
                f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                for col in df_group.columns
            ]
            df_group = df_group.reset_index()
            rename_mapping = {
                'dates_count': 'Counts',
                'severity_score_sum': 'Severity_sum',
                'severity_score_mean': 'Severity_mean',
                'resolution_score_sum': 'Resolution_sum',
                'resolution_score_mean': 'Resolution_mean',
                'severity_per_resolution_sum': 'Severity_per_resolution_sum',
                'severity_per_resolution_mean': 'Severity_per_resolution_mean'
            }
            df_group = df_group.rename(columns=rename_mapping)

        else:
            st.warning("선택된 컬럼이 없어 그룹화할 수 없습니다.")
            df_group = pd.DataFrame()
except Exception as e:
    st.error(f"그룹화 중 오류 발생: {e}")
    df_group = pd.DataFrame()

# 시각화 설정 옵션 제공
st.subheader("그래프 설정")
st.write("위에서 선택된 필터에 따라 축과 색을 설정 해주세요.")
columns_for_x_and_color = ['없음', 'l_category', 'category', 'pddistrict', 'year', 'month', 'day', 'hour', 'dayofweek']
metrics = ['Counts', 'Severity_sum', 'Severity_mean',
           'Resolution_sum', 'Resolution_mean',
           'Severity_per_resolution_sum', 'Severity_per_resolution_mean']

graph_types = ['Bar']

# 사용자 선택 입력
metric = st.multiselect('그래프를 표시할 값 선택', metrics, default=['Counts'])
x_axis = st.selectbox('X축 선택', columns_for_x_and_color[1:], index=0)
color_axis = st.selectbox('Color 기준 선택', columns_for_x_and_color, index=0)
graph_type = st.selectbox('그래프 유형 선택', graph_types, index=0)

barmode = st.radio(
    "막대그래프 표시 방식 선택",
    ("group", "overlay"),
    index=0,
    key="barmode_selection"  # 선택 상태 기억
)
sort_option = st.radio(
    "X축 정렬 방식 선택",
    ("정렬 없음", "Y값 내림차순 (큰값 우선)", "Y값 오름차순 (작은값 우선)"),
    horizontal=True
)
shared_xaxes_option = st.radio(
    "X축 공유 설정",
    ("공유 (shared)", "개별 (not shared)"),
    index=0
)
shared_xaxes = True if shared_xaxes_option == "공유 (shared)" else False

if st.button('그래프 생성하기'):
    if len(metric) >= 1 and not df_group.empty:
        rows = len(metric)
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=shared_xaxes, vertical_spacing=0.1,
            subplot_titles=metric
        )
        fig.update_xaxes(showticklabels=True)

        for i, m in enumerate(metric):
            df_to_plot = df_group.copy()

            # --- 정렬 적용 ---
            if sort_option != "정렬 없음":
                ascending = True if sort_option == "Y값 오름차순 (작은값 우선)" else False
                df_to_plot = df_to_plot.sort_values(by=m, ascending=ascending)

            if color_axis == '없음':
                # 색상 기준 없음
                fig.add_trace(
                    go.Bar(x=df_to_plot[x_axis], y=df_to_plot[m], name=m),
                    row=i+1, col=1
                )
            else:
                # 색상 기준 있음
                for category in df_to_plot[color_axis].dropna().unique():
                    df_filtered = df_to_plot[df_to_plot[color_axis] == category]
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

        fig.update_layout(height=500 * rows, title_text=f"{x_axis} 기준 지표별 Subplot 비교", barmode=barmode)
        st.plotly_chart(fig)
    else:
        st.warning("1개 이상의 지표를 선택하고, 필터링된 데이터가 있어야 시각화할 수 있습니다.")

# 히트맵 시각화 섹션
st.subheader("피벗 테이블 히트맵")
st.write("x축과 color기준(y축) 설정에서 2개 이상 선택 해주세요.")
st.write("그래프를 표시할 값을 하나만 선택해주세요.")
sort_option = st.radio(
        "X축 정렬 기준 (합계)",
        ("총합 큰 순", "총합 작은 순"),
        index=0
    )
if st.button('히트맵 생성하기'):
    if x_axis != '없음' and color_axis != '없음' and len(metric) == 1:
        pivot_metric = metric[0]

        try:
            pivot_df = df_group.pivot_table(
                index=color_axis,
                columns=x_axis,
                values=pivot_metric,
                aggfunc='sum'
            )
            # 정렬하기 (pivot_df의 columns 기준)
            if sort_option == "총합 큰 순":
                sorted_cols = pivot_df.sum(axis=0).sort_values(ascending=False).index
            else:
                sorted_cols = pivot_df.sum(axis=0).sort_values(ascending=True).index
            pivot_df = pivot_df[sorted_cols]  
        
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
                aggfunc='sum'
            )

            # 행(row)별 정규화 (각 row를 0~1 범위로)
            normalized_pivot_df = pivot_df.copy()
            normalized_pivot_df = normalized_pivot_df.apply(
                lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) if np.nanmax(x) != np.nanmin(x) else 0,
                axis=1
            )

            # 정렬하기 (pivot_df의 columns 기준)
            if sort_option == "총합 큰 순":
                sorted_cols = normalized_pivot_df.sum(axis=0).sort_values(ascending=False).index
            else:
                sorted_cols = normalized_pivot_df.sum(axis=0).sort_values(ascending=True).index
            normalized_pivot_df = normalized_pivot_df[sorted_cols]  

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
                aggfunc='sum'
            )

            # 행(row)별 정규화 (각 row를 0~1 범위로)
            normalized_pivot_df = pivot_df.copy()
            normalized_pivot_df = normalized_pivot_df.apply(
                lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) if np.nanmax(x) != np.nanmin(x) else 0,
                axis=0
            )
            normalized_pivot_df = normalized_pivot_df[sorted_cols]
            
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

@st.cache_data
def load_geojson():
    file_id = "1FX4nVP9GiQY1rJg3jrfd7PwwBgN4baOZ"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    geojson_path = "Current_Police_Districts.geojson"

    if not os.path.exists(geojson_path):
        gdown.download(url, geojson_path, quiet=False)

    gdf = gpd.read_file(geojson_path)
    gdf_json = json.loads(gdf.to_json())

    return gdf, gdf_json

gdf, gdf_json = load_geojson()


# 지도 표시 여부를 Streamlit 세션 상태로 관리
if 'show_map' not in st.session_state:
    st.session_state['show_map'] = False

st.subheader("선택된 범죄 위치 지도")
st.write("위에서 색 기준 설정이 마커 색에도 적용됩니다.")
st.write("적절한 필터 설정을 통해 지도 생성을 가볍게 해주세요.")


# 지도 타입 선택
map_type = st.radio(
    "지도 엔진 선택",
    ("Pydeck 지도", "Folium 지도")
)

# 지도 On/Off 토글 버튼
if 'show_map' not in st.session_state:
    st.session_state['show_map'] = False

if st.button("지도 보기 / 숨기기"):
    st.session_state['show_map'] = not st.session_state['show_map']

# 현재 상태에 따라 지도 표시
if st.session_state['show_map']:
    if not filtered_df.empty:
        center_lat = filtered_df['y'].mean()
        center_lon = filtered_df['x'].mean()
    else:
        center_lat, center_lon = 37.77, -122.42

    if map_type == "Folium 지도":
        ## --- 여기에 folium 코드 (당신이 주석처리했던 부분) ---
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

        folium.GeoJson(
            gdf,
            name="Police Districts",
            style_function=lambda feature: {
                'color': 'gray',         # 경계선 색
                'weight': 2,             # 선 굵기
                'fillColor': 'lightgray', # 내부 채우는 색
                'fillOpacity': 0.1       # 내부 채우기 투명도
            },
            tooltip=folium.GeoJsonTooltip(fields=["district"], aliases=["district:"])
        ).add_to(m)

        heat_data = [[row['y'], row['x']] for idx, row in filtered_df.iterrows() if pd.notnull(row['y']) and pd.notnull(row['x'])]
        heatmap = HeatMap(heat_data, radius=8, blur=15, min_opacity=0.4)

        if color_axis != '없음' and color_axis in filtered_df.columns:
            unique_values = filtered_df[color_axis].dropna().unique()
            colormap = cm.get_cmap('plasma')
            color_mapping = {val: mcolors.to_hex(colormap(i / len(unique_values))) for i, val in enumerate(unique_values)}
        else:
            color_mapping = {}

        marker_cluster = MarkerCluster(name="MarkerCluster")
        for idx, row in filtered_df.iterrows():
            if pd.notnull(row['y']) and pd.notnull(row['x']):
                popup_text = f"category: {row['category']}<br>resolution: {row['resolution']}"
                color = color_mapping.get(row[color_axis], 'blue') if color_axis != '없음' else 'blue'

                folium.CircleMarker(
                    location=[row['y'], row['x']],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=popup_text
                ).add_to(marker_cluster)

        heatmap_layer = folium.FeatureGroup(name='Heatmap')
        heatmap_layer.add_child(heatmap)
        m.add_child(heatmap_layer)
        m.add_child(marker_cluster)

        folium.LayerControl(position='topright', collapsed=False).add_to(m)

        st_folium(m, width=800, height=600)

    elif map_type == "Pydeck 지도":
        weekday_mapping = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4,
            'Saturday': 5,
            'Sunday': 6
        }
        if animation_axis == 'dayofweek':
            # 요일을 숫자로 변환
            filtered_df['animation_value'] = filtered_df['dayofweek'].map(weekday_mapping)
        else:
            # year, month, hour은 그대로 사용
            filtered_df['animation_value'] = filtered_df[animation_axis]
        if color_axis != '없음' and color_axis in filtered_df.columns:
            unique_values = filtered_df[color_axis].dropna().unique()
            cmap = cm.get_cmap('Set1', len(unique_values))  # 'plasma' 대신 원하는 colormap 가능
            color_mapping = {val: [int(r*255), int(g*255), int(b*255)] 
                            for val, (r, g, b, _) in zip(unique_values, cmap(np.linspace(0, 1, len(unique_values))))}
            filtered_df['color'] = filtered_df[color_axis].map(color_mapping)
        else:
            filtered_df['color'] = [[0, 0, 255] for _ in range(len(filtered_df))]
        # --- 슬라이더로 애니메이션 값 선택 ---
        min_val = int(filtered_df['animation_value'].min())
        max_val = int(filtered_df['animation_value'].max())
        selected_value = st.slider(
            f"{animation_axis} 값 선택 (애니메이션 슬라이더)",
            min_value=min_val,
            max_value=max_val,
            value=min_val,
            step=1
        )
        filtered_df_anim = filtered_df[filtered_df['animation_value'] == selected_value]
        if filtered_df_anim.empty:
            st.warning(f"선택한 {animation_axis} = {selected_value} 에 해당하는 데이터가 없습니다.")

        if layer_type == "ScatterplotLayer":
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_df_anim,
                get_position='[x, y]',
                get_fill_color='color',
                get_radius=10,
                radius_min_pixels=2,
                radius_max_pixels=10,
                opacity=1,
                filled=True,
                stroked=True,
                get_line_color=[0, 0, 0],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            )
        elif layer_type == "HexagonLayer":
            layer = pdk.Layer(
                "HexagonLayer",
                data=filtered_df_anim,
                get_position='[x, y]',
                radius=100,
                elevation_scale=50,
                elevation_range=[0, 1000],
                extruded=True,
                coverage=1,
                pickable=True,
                auto_highlight=True,
            )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=0
        )

        # GeoJsonLayer 추가
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            data=gdf_json,
            stroked=True,
            filled=False,
            line_width_min_pixels=2,
            get_line_color=[128, 128, 128],
            pickable=True
        )
        # 지도 스타일 선택
        map_style_option = st.radio(
            "지도 스타일 선택",
            (
                "Dark (어두운 지도)",
                "Light (밝은 지도)",
                "Street (도로 지도)",
                "Satellite (위성 지도)"
            )
        )
        map_style_dict = {
            "Dark (어두운 지도)": "mapbox://styles/mapbox/dark-v9",
            "Light (밝은 지도)": "mapbox://styles/mapbox/light-v9",            
            "Street (도로 지도)": "mapbox://styles/mapbox/streets-v11",
            "Satellite (위성 지도)": "mapbox://styles/mapbox/satellite-v9"
        }
        selected_map_style = map_style_dict[map_style_option]

        r = pdk.Deck(
            layers=[geojson_layer, layer],  # 여러 레이어 추가 가능
            initial_view_state=view_state,
            map_style=selected_map_style,
            tooltip={"text": "category: {category}\nresolution: {resolution}"},
        )

        st.pydeck_chart(r)

        # 범례 추가
        if color_axis != '없음' and color_axis in filtered_df.columns:
            st.markdown("### 색상 범례")
            for val, color in color_mapping.items():
                color_hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                st.markdown(
                    f"<div style='display:flex; align-items:center;'>"
                    f"<div style='width:20px; height:20px; background:{color_hex}; margin-right:10px;'></div>"
                    f"<span style='font-size:16px;'>{val}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
else:
    st.info("지도가 숨겨져 있습니다.")
