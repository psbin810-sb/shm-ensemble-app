"""
app.py — 5시드 앙상블 PI-DeepONet 구조 전역응답 복원 데모 (Streamlit 웹앱)
========================================================================
패널 QR → 휴대폰 브라우저 접속용.

3가지 입력 모드:
  ① 실시간 복원   : ensemble_pred.npz(720 테스트 시점) 시간순 재생.
                    센서 실측이 들어오면 앙상블 복원 전역응답 + 불확실성 표시.
  ② 예시 가상데이터: 대표 케이스 선택 → 결과.
  ③ 직접 입력      : 부재별 Δ변형률·Δ각도 입력 → 5시드 라이브 앙상블 추론(평균±std).

출력: 부재 축응력(∝ε) 색분포 + 변형형상 + 앙상블 불확실성.

실행:  streamlit run app.py
"""
import numpy as np
import streamlit as st

from ensemble_core import (EnsembleResults, build_branch_from_dstrain_drot,
                           SENSOR_ELEM_IDS, SENSOR_FILE_TAG)
from plotting import draw_response, bar_wind_coefs

st.set_page_config(page_title="앙상블 구조 전역응답 복원", page_icon="🏗️", layout="wide")
ASSET = "assets"


@st.cache_resource
def get_results():
    return EnsembleResults()


# ──────────────────────────────────────────────────────────────────
# 공통 렌더
# ──────────────────────────────────────────────────────────────────
def metrics_row(u, sigma, c_mean, c_std, u_std=None):
    cols = st.columns(5)
    cols[0].metric("최대 인장응력", f"{sigma.max():.2f} MPa")
    cols[1].metric("최대 압축응력", f"{sigma.min():.2f} MPa")
    cols[2].metric("최대 변위", f"{np.linalg.norm(u, axis=1).max():.2f} mm")
    if u_std is not None:
        cols[3].metric("평균 변위 불확실성", f"±{np.linalg.norm(u_std, axis=1).mean():.2f} mm")
    cols[4].metric("풍하중 |c|", f"{np.linalg.norm(c_mean):.2f}")


def render_case(u, sigma, c_mean, c_std, scale_factor, vmax, title, u_std=None):
    metrics_row(u, sigma, c_mean, c_std, u_std)
    left, right = st.columns([3, 1])
    with left:
        st.pyplot(draw_response(u, sigma, scale_factor, vmax=vmax, title=title, u_std=u_std))
    with right:
        st.pyplot(bar_wind_coefs(c_mean, c_std))


# ──────────────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────────────
st.title("🏗️ 구조 건전성 모니터링 — 5시드 앙상블 전역응답 복원")
st.caption("PI-DeepONet ×5 시드 앙상블: 센서 4점(변형률·회전) → 전체 변위·응력 복원 + 불확실성")

mode = st.sidebar.radio("입력 모드",
                        ["① 실시간 복원 (앙상블 결과 재생)",
                         "② 예시 가상 데이터",
                         "③ 직접 입력 (Δ변형률·Δ각도)"])
scale_factor = st.sidebar.slider("변형 과장 배율", 1, 500, 100, step=1)
fix_scale = st.sidebar.checkbox("응력 색범위 고정", value=False)
fixed_vmax = st.sidebar.number_input("고정 색범위 vmax (MPa)", 1.0, 200.0, 30.0,
                                     disabled=not fix_scale)
vmax = fixed_vmax if fix_scale else None
show_unc = st.sidebar.checkbox("불확실성(±std) 표시", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"센서: {', '.join(f'{t}(부재{e})' for t,e in zip(SENSOR_FILE_TAG, SENSOR_ELEM_IDS))}")

R = get_results()
st.sidebar.caption(f"앙상블 시드: {list(R.seeds)} ({len(R.seeds)}개)")
st.sidebar.caption(f"테스트 시점: {R.T}개")


# ──────────────────────────────────────────────────────────────────
# 모드 ③ — 직접 입력 (가장 가까운 관측 사례 검색)
# ──────────────────────────────────────────────────────────────────
if mode.startswith("③"):
    st.subheader("③ 직접 입력 — 부재별 Δ변형률·Δ각도")
    st.caption("입력한 변화량과 **가장 가까운 실제 관측 시점**의 앙상블 복원 결과를 보여줍니다. "
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
    render_case(cs["u"], cs["sigma"], cs["c_mean"], cs["c_std"],
                scale_factor, vmax, f"Nearest observed case (#{i})",
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
        st.caption("※ 이 모드는 학습 스케일러(scalers.pkl)가 없어 라이브 추론 대신 "
                   "최근접 관측 사례를 보여줍니다. scalers.pkl 확보 시 진짜 5시드 추론으로 교체 가능.")


# ──────────────────────────────────────────────────────────────────
# 모드 ② — 예시 가상 데이터
# ──────────────────────────────────────────────────────────────────
elif mode.startswith("②"):
    st.subheader("② 예시 가상 데이터 — 대표 케이스 선택")
    ex = R.example_indices()
    names = list(ex.keys())
    pick = st.selectbox("케이스 선택", names)
    i = ex[pick]
    cs = R.case(i)
    st.caption(f"📌 test idx={cs['idx_te']}  |  |ε|={np.linalg.norm(cs['eps_meas'])*1e6:.1f}µε  "
               f"|θ|={np.degrees(np.linalg.norm(cs['th_meas'])):.3f}°  "
               f"불확실성≈±{np.linalg.norm(cs['u_std'],axis=1).mean():.2f}mm")
    render_case(cs["u"], cs["sigma"], cs["c_mean"], cs["c_std"],
                scale_factor, vmax, f"Example: {pick} (#{i})",
                u_std=cs["u_std"] if show_unc else None)


# ──────────────────────────────────────────────────────────────────
# 모드 ① — 실시간 복원 (앙상블 결과 재생)
# ──────────────────────────────────────────────────────────────────
else:
    st.subheader("① 실시간 복원 — 앙상블 결과 재생")
    cc = st.columns([1, 1, 2])
    playing = cc[0].toggle("▶ 재생", value=False, key="rt_play")
    speed = cc[1].selectbox("재생 속도", [1, 2, 5, 10], index=1,
                            help="한 번에 건너뛸 시점 수")
    if "rt_idx" not in st.session_state:
        st.session_state.rt_idx = 0
    idx = st.slider("재생 위치", 0, R.T - 1, st.session_state.rt_idx, key="rt_slider")
    st.session_state.rt_idx = idx

    ph_metrics = st.container()
    ph_fig = st.empty()
    ph_hist = st.empty()

    def render(i):
        cs = R.case(i)
        with ph_metrics:
            metrics_row(cs["u"], cs["sigma"], cs["c_mean"], cs["c_std"], cs["u_std"])
        left, right = ph_fig.columns([3, 1])
        with left:
            st.pyplot(draw_response(
                cs["u"], cs["sigma"], scale_factor, vmax=vmax,
                title=f"Realtime  test idx={cs['idx_te']}  step {i+1}/{R.T}",
                u_std=cs["u_std"] if show_unc else None))
        with right:
            st.pyplot(bar_wind_coefs(cs["c_mean"], cs["c_std"]))
        # 들어오는 센서 데이터(최근 이력)
        s0 = max(0, i - 120)
        eps_hist = R.eps_meas[s0:i + 1] * 1e6   # µε
        ph_hist.line_chart(
            {f"{t} (µε)": eps_hist[:, j] for j, t in enumerate(SENSOR_FILE_TAG)},
            height=160)

    render(idx)

    @st.fragment(run_every=0.6 if playing else None)
    def autoplay():
        if not st.session_state.get("rt_play"):
            return
        nxt = st.session_state.rt_idx + int(speed)
        if nxt >= R.T:
            nxt = 0
        st.session_state.rt_idx = nxt
        st.rerun()

    autoplay()

    with st.expander("ℹ️ 이 모드 설명"):
        st.markdown(
            "- `ensemble_pred.npz` 의 720개 테스트 시점을 시간순 재생합니다.\n"
            "- **들어오는 센서 데이터**(eps_meas) → **5시드 앙상블이 복원한 전역 변위/응력** + "
            "**불확실성(±std, 주황 구름)**.\n"
            "- 풍하중계수는 평균±앙상블std 막대로 표시.")
