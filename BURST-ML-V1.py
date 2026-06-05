import streamlit as st

# 반드시 앱의 가장 상단(다른 st 명령어 전)에 위치해야 합니다.
st.set_page_config(
    page_title="Burst-AI 분석기",
    page_icon="https://your-website.com/logo.png", # 혹은 로컬 파일 경로 'icon.png'
    layout="wide"
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ====================================================================
# 0. Matplotlib 한글 설정 (깨짐 방지)
# ====================================================================
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False 
except:
    pass

# ====================================================================
# 1. 파일 및 변수 설정
# ====================================================================

TARGET_COLUMN = 'Y_Burst'
RANDOM_SEED = 42

# ====================================================================
# 2. 데이터 로드 및 3. 모델 훈련
# ====================================================================

@st.cache_data
def load_data(uploaded_file):
    """업로드된 CSV 파일을 읽어 Dataframe을 반환합니다."""
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_resource(show_spinner="모델 훈련 및 최적 조건 분석 중...")
def train_model(df):
    """데이터를 분리하고 RandomForest 모델을 훈련 및 평가합니다."""
    
    if TARGET_COLUMN not in df.columns:
        st.error(f"🚨 오류: 업로드된 파일에 목표 변수 '{TARGET_COLUMN}' 컬럼이 없습니다.")
        return None, None, None, None, None, None, None
        
    X = df.drop(columns=[TARGET_COLUMN])
    Y = df[TARGET_COLUMN]
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED
    )
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    df_importances = feature_importances.nlargest(len(X.columns)).reset_index()
    df_importances.columns = ['Feature', 'Importance_Score']
    
    Y_pred_all = model.predict(X)
    max_burst_index = np.argmax(Y_pred_all)
    max_predicted_burst = round(Y_pred_all[max_burst_index], 1)
    
    best_condition_series = X.iloc[max_burst_index].round(1)
    
    return X, r2, rmse, mse, df_importances, max_predicted_burst, best_condition_series

# ====================================================================
# 4. Streamlit UI 구성 함수
# ====================================================================

def display_reliability(r2, rmse, mse):
    """AI 모델 신뢰성 지표를 표시합니다."""
    
    r2_evaluation = ""
    if r2 >= 0.8:
        r2_evaluation = "✅ 매우 높음 (패턴 학습 완벽)"
    elif r2 >= 0.5:
        r2_evaluation = "👍 양호함 (일부 패턴 학습)"
    elif r2 >= 0.0:
        r2_evaluation = "⚠️ 낮음 (추가 데이터/튜닝 필요)"
    else:
        r2_evaluation = "🚨 매우 낮음 (모델 개선 시급)"
        
    st.subheader("1. AI 모델 신뢰성 지표")
    
    metrics_data = {
        'Metric': ['Model Type', 'R-squared (결정 계수)', 'RMSE', 'MSE'],
        'Value': ['RandomForestRegressor', f"{r2:.4f}", f"{rmse:.4f}", f"{mse:.4f}"],
        'Explanation': ['사용된 회귀 모델', r2_evaluation, '오차 표준 편차 (낮을수록 좋음)', '오차 제곱 평균 (낮을수록 좋음)']
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    st.table(df_metrics.set_index('Metric'))
    st.info(f"R-squared 평가: **{r2_evaluation}**")


def display_best_condition_bar_chart(max_burst, best_condition_series):
    """최대 Burst 예측값과 최적 조건을 수평 막대 차트로 시인성 있게 표시합니다."""
    
    # 0. 최대 예측값 강조
    st.markdown("### 🔥 최대 BURST 예측 결과")
    # 폰트 크기 조절을 위해 markdown 사용 (Streamlit 기본 metric보다 큼)
    st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #FF4B4B; border-radius: 5px; text-align: center; background-color: #FFF0F0;">
            <p style="font-size: 16px; margin: 0;">최대 예측 {TARGET_COLUMN}</p>
            <h1 style="color: #FF4B4B; margin: 5px 0 0 0; font-size: 40px;">{max_burst:.1f}</h1>
            <p style="font-size: 12px; color: gray; margin: 0;">(최적 조건 적용 시 기대값)</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 1. 최적 조건 섹션 (막대 차트)
    st.subheader(f"2. 최적 사출 공정 파라미터 ({max_burst:.1f} 달성 조건)")
    
    # 데이터를 DataFrame으로 변환
    df_condition = best_condition_series.to_frame(name='Optimal Value')
    
    # Matplotlib 차트 생성 (가로폭을 넓게, 세로 길이를 동적으로)
    fig, ax = plt.subplots(figsize=(12, len(df_condition) * 0.4 + 2)) 
    
    # 수평 막대 그래프 (Optimal Value)
    y_pos = np.arange(len(df_condition))
    values = df_condition['Optimal Value'].values
    params = df_condition.index.values

    # 막대 색상을 눈에 띄는 주황색 계열로 변경
    ax.barh(y_pos, values, color='#FF8C00')
    
    # 레이블 설정 (폰트 크기 키움)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=12) # 파라미터 이름 폰트 크기 증가
    ax.set_xlabel('Optimal Value (최적값)', fontsize=14)
    ax.set_title('최대 BURST 달성을 위한 사출 조건별 최적값', fontsize=16)
    ax.invert_yaxis() 

    # 값 표시 (막대 옆에 숫자를 크게 출력)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1f}", color='black', va='center', fontsize=11, fontweight='bold')

    plt.grid(axis='x', linestyle='--', alpha=0.7) # 그리드 추가
    plt.tight_layout()
    st.pyplot(fig) # Streamlit에 그래프 표시
    
    # 전체 데이터 프레임도 하단에 축소하여 제공
    with st.expander("🔍 모든 최적 조건 파라미터 테이블로 보기"):
        st.dataframe(df_condition.T)
    


def display_importance_chart(df_importances):
    """변수 중요도를 막대 그래프로 표시합니다."""
    
    st.subheader("3. Feature Importance (변수 중요도)")
    
    # 상위 10개 변수만 추출
    df_top10 = df_importances.sort_values(by='Importance_Score', ascending=False).head(10).set_index('Feature')
    
    # 차트 크기를 키워 시인성을 확보
    fig = plt.figure(figsize=(12, len(df_top10) * 0.4 + 3)) # 가로폭 12로 확대
    
    plt.barh(df_top10.index, df_top10['Importance_Score'], color='#007BFF')
    plt.title('Y_Burst 예측을 위한 상위 10개 변수 중요도', fontsize=16)
    plt.xlabel('중요도 점수', fontsize=14)
    
    # Y축 라벨 폰트 크기 증가
    plt.yticks(fontsize=12)
    
    plt.gca().invert_yaxis() 
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)
    st.markdown("---")
    st.caption("AI 모델의 신뢰도가 낮을 경우(R-squared < 0.5), 변수 중요도는 참고용으로만 활용해야 합니다.")
    
    

# ====================================================================
# 5. Streamlit 메인 실행 함수
# ====================================================================

def main_app():
    
    st.set_page_config(layout="wide") 
    st.title("AI 기반 사출 성형 (Y_Burst) 최적화 분석")
    st.markdown("---")

    # 1. 파일 업로드 섹션
    uploaded_file = st.sidebar.file_uploader( # 사이드바에 업로드 영역 배치
        "CSV 데이터 파일을 업로드하세요:", 
        type=['csv'], 
        help="공정 조건(X)과 목표 변수(Y_Burst)를 포함하는 CSV 파일이 필요합니다."
    )
    
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.sidebar.success(f"✅ 파일 로드 성공: 총 {len(df)}개 레코드")
            
            # 2. 분석 실행 및 결과 표시
            if st.sidebar.button("▶️ AI 모델 분석 및 결과 표시 시작", use_container_width=True):
                
                X, r2, rmse, mse, df_importances, max_burst, best_condition_series = train_model(df)
                
                if X is not None:
                    
                    st.success("✅ 분석 완료! 결과를 확인하세요.")
                    
                    # 결과를 2개 컬럼으로 분할
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # 1. 신뢰성 지표 표시
                        display_reliability(r2, rmse, mse)

                    with col2:
                        # 2. 최대 BURST 값과 최적 조건 (막대 차트 형식) 표시
                        display_best_condition_bar_chart(max_burst, best_condition_series)

                    # 3. 변수 중요도 (하단 전체 영역)
                    display_importance_chart(df_importances)
                
        except Exception as e:
            st.error(f"🚨 데이터 처리 중 오류 발생: 파일 형식을 확인하거나 데이터에 문제가 없는지 검토하세요. ({e})")
    else:
        st.info("좌측 사이드바에서 CSV 데이터 파일을 업로드하여 분석을 시작하세요.")

# ====================================================================
# 6. 스크립트 실행
# ====================================================================
if __name__ == '__main__':
    main_app()