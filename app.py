"""
app.py — 구조물 응답 복원 데모 (Streamlit 웹앱)
================================================
패널 QR → 휴대폰 브라우저 접속용.

3가지 입력 모드:
  실시간 복원   : ensemble_pred.npz(720 테스트 시점) 시간순 재생.
  예시 가상데이터: 대표 케이스 선택 → 결과.
  직접 입력      : 부재별 Δ변형률·Δ각도 입력 → 가장 가까운 관측 사례.

출력: 부재 축응력(∝ε) 색분포 + 변형형상 + 앙상블 불확실성.

실행:  streamlit run app.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from ensemble_core import (EnsembleResults, build_branch_from_dstrain_drot,
                           SENSOR_ELEM_IDS, SENSOR_FILE_TAG)
from plotting import draw_response

st.set_page_config(page_title="구조물 응답 복원", layout="wide")

VMAX = 25.0   # 응력 색범위 고정 (±25 MPa)


@st.cache_resource
def get_results():
    return EnsembleResults()


# ──────────────────────────────────────────────────────────────────
# 공통 렌더
# ──────────────────────────────────────────────────────────────────
def metrics_row(u, sigma, u_std=None):
    r1 = st.columns(2)
    r1[0].metric("최대 인장응력", f"{sigma.max():.2f} MPa")
    r1[1].metric("최대 압축응력", f"{sigma.min():.2f} MPa")
    r2 = st.columns(2)
    r2[0].metric("최대 변위", f"{np.linalg.norm(u, axis=1).max():.2f} mm")
    if u_std is not None:
        r2[1].metric("평균 변위 불확실성", f"±{np.linalg.norm(u_std, axis=1).mean():.2f} mm")


def render_case(u, sigma, scale_factor, title, u_std=None):
    metrics_row(u, sigma, u_std)
    st.pyplot(draw_response(u, sigma, scale_factor, vmax=VMAX, title=title, u_std=u_std))


# ──────────────────────────────────────────────────────────────────
# 메인 — 제목 + 입력모드 버튼 + 변형배율 (휴대폰에서도 바로 보이게 메인에 배치)
# ──────────────────────────────────────────────────────────────────
st.title("구조물 응답 복원")
st.caption("센서 4점(변형률·회전) → 구조물 전체 변위·응력 복원 + 불확실성")

R = get_results()

MODES = ["실시간 복원", "예시 가상 데이터", "직접 입력"]
mode = st.segmented_control("입력 모드", MODES, default=MODES[0], key="mode")
if not mode:
    mode = MODES[0]
scale_factor = st.slider("변형 과장 배율", 1, 500, 100, step=1)

# 사이드바: 보조 옵션/정보
show_unc = st.sidebar.toggle("불확실성(±std) 표시", value=True)
st.sidebar.caption(f"센서: {', '.join(f'{t}(부재{e})' for t,e in zip(SENSOR_FILE_TAG, SENSOR_ELEM_IDS))}")
st.sidebar.caption(f"앙상블 시드 {len(R.seeds)}개 · 테스트 시점 {R.T}개")

st.divider()


# ──────────────────────────────────────────────────────────────────
# 직접 입력 (가장 가까운 관측 사례 검색)
# ──────────────────────────────────────────────────────────────────
if mode == "직접 입력":
    st.subheader("직접 입력 — 부재별 Δ변형률·Δ각도")
    st.caption("입력한 변화량과 **가장 가까운 실제 관측 시점**의 복원 결과를 보여줍니다. "
               "(Δε 단위 µε, Δθ 단위 °)")
    cols = st.columns(4)
    dstrain, drot = [], []
    for j, (t, e) in enumerate(zip(SENSOR_FILE_TAG, SENSOR_ELEM_IDS)):
        with cols[j]:
            st.markdown(f"**{t}** (부재 {e})")
            dstrain.append(st.slider(f"Δε_{t} (µε)", -300.0, 300.0, 0.0, 1.0, key=f"ds{j}"))
            drot.append(st.slider(f"Δθ_{t} (°)", -0.5, 0.5, 0.0, 0.005, key=f"dr{j}",
                                  format="%.3f"))
    X = build_branch_from_dstrain_drot(np.array(dstrain), np.array(drot))
    i, dist = R.nearest(X)
    cs = R.case(i)
    st.info(f"🔎 가장 가까운 관측 사례: test idx={cs['idx_te']} (#{i})  ·  정규화 거리={dist:.2f}")
    render_case(cs["u"], cs["sigma"], scale_factor, f"Nearest observed case (#{i})",
                u_std=cs["u_std"] if show_unc else None)
    with st.expander("입력값 vs 매칭된 관측값"):
        import pandas as pd
        matched_eps = cs["eps_meas"] * 1e6        # µε
        matched_th = np.degrees(cs["th_meas"])    # °
        df = pd.DataFrame({
            "센서": [f"{t}(부재{e})" for t, e in zip(SENSOR_FILE_TAG, SENSOR_ELEM_IDS)],
            "입력 Δε(µε)": np.round(dstrain, 1), "매칭 Δε(µε)": np.round(matched_eps, 1),
            "입력 Δθ(°)": np.round(drot, 4), "매칭 Δθ(°)": np.round(matched_th, 4),
        })
        st.dataframe(df, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# 예시 가상 데이터
# ──────────────────────────────────────────────────────────────────
elif mode == "예시 가상 데이터":
    st.subheader("예시 가상 데이터 — 대표 케이스 선택")
    ex = R.example_indices()
    names = list(ex.keys())
    pick = st.selectbox("케이스 선택", names)
    i = ex[pick]
    cs = R.case(i)
    st.caption(f"📌 test idx={cs['idx_te']}  |  |ε|={np.linalg.norm(cs['eps_meas'])*1e6:.1f}µε  "
               f"|θ|={np.degrees(np.linalg.norm(cs['th_meas'])):.3f}°  "
               f"불확실성≈±{np.linalg.norm(cs['u_std'],axis=1).mean():.2f}mm")
    render_case(cs["u"], cs["sigma"], scale_factor, f"Example: {pick} (#{i})",
                u_std=cs["u_std"] if show_unc else None)


# ──────────────────────────────────────────────────────────────────
# 실시간 복원 (앙상블 결과 재생)
# ──────────────────────────────────────────────────────────────────
else:
    st.subheader("실시간 복원")
    SPEED = 5          # 한 틱에 5시점씩 진행 (고정)
    INTERVAL = 0.5     # 틱 간격(초) — 애니메이션 속도

    if "rt_idx" not in st.session_state:
        st.session_state.rt_idx = 0

    # 자동 재생: fragment 영역만 주기적으로 다시 그림 (전체 rerun 안 함 → 깜빡임 없음)
    @st.fragment(run_every=INTERVAL)
    def animate():
        i = int(st.session_state.rt_idx) % R.T
        cs = R.case(i)
        metrics_row(cs["u"], cs["sigma"], cs["u_std"] if show_unc else None)
        fig = draw_response(
            cs["u"], cs["sigma"], scale_factor, vmax=VMAX,
            title=f"Realtime  test idx={cs['idx_te']}  step {i+1}/{R.T}",
            u_std=cs["u_std"] if show_unc else None)
        st.pyplot(fig)
        plt.close(fig)
        # 들어오는 센서 데이터(최근 이력)
        s0 = max(0, i - 120)
        eps_hist = R.eps_meas[s0:i + 1] * 1e6   # µε
        st.line_chart(
            {f"{t} (µε)": eps_hist[:, j] for j, t in enumerate(SENSOR_FILE_TAG)},
            height=160)
        # 다음 프레임으로 진행 (다음 자동 rerun 때 반영)
        st.session_state.rt_idx = (i + SPEED) % R.T

    animate()

    with st.expander("ℹ️ 이 모드 설명"):
        st.markdown(
            "- 테스트 시점을 시간순으로 **자동 재생**합니다(항상 재생, 속도 5).\n"
            "- **들어오는 센서 데이터** → **복원한 전역 변위/응력** + "
            "**불확실성(±std, 주황 점)**.")
