"""
ensemble_core.py
================
5개 시드 앙상블 PI-DeepONet 기반 구조 전역응답 복원 코어.

데이터 소스 (런타임):
  · assets/ensemble_pred.npz       — 720 테스트 시점의 앙상블 예측 결과(평균±std).
    → 모드 ①(재생)·②(예시)·③(최근접 관측 사례) 모두 이 파일만 사용. numpy 만 필요.

  (선택) assets/seeds/seed_*/best.pt + assets/scalers.pkl 는 live_inference.py 의
   5시드 라이브 추론용. 현재 app.py 는 import 하지 않음 → torch 불필요(클라우드 경량 배포).

입력 규약 (base 코드 = '웹 개발 기반 코드.py' 와 동일):
  branch 입력 8 = [strain×4, rotation×4], 부재순 (1,2,200,201) ← (1V,1D,4D,4V).
    strain  = npz eps_meas 와 동일 공간 (상대 변형률, 무차원)
    rotation= npz th_meas 와 동일 공간 (drot_fb_deg × π/180, rad)
"""
import os
import numpy as np

from truss_geometry import ELEMENTS_1BASED, NODE_COORDS

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
NPZ_PATH    = os.path.join(ASSET_DIR, "ensemble_pred.npz")
SEEDS_DIR   = os.path.join(ASSET_DIR, "seeds")
SCALER_PATH = os.path.join(ASSET_DIR, "scalers.pkl")

# 모델 하이퍼파라미터 (base 코드 기본값)
P_LATENT, HIDDEN_BRANCH, HIDDEN_TRUNK = 16, 64, 32

SENSOR_ELEM_IDS = (1, 2, 200, 201)
SENSOR_FILE_TAG = ("1V", "1D", "4D", "4V")
SENSOR_ELEM_TO_PRIMARY_NODE = {1: 2, 2: 4, 200: 100, 201: 101}
SENSOR_NODE_IDS_1B = np.array([SENSOR_ELEM_TO_PRIMARY_NODE[e] for e in SENSOR_ELEM_IDS], dtype=int)

STRAIN_RAW_TO_PHYS = 1.0e-6
E_STEEL_MPA = 200_000.0


# ──────────────────────────────────────────────────────────────────
# 부재 기하 + 응력
# ──────────────────────────────────────────────────────────────────
def _member_geometry():
    M = len(ELEMENTS_1BASED)
    ni = np.zeros(M, np.int64); nj = np.zeros(M, np.int64)
    L0 = np.zeros(M); cos = np.zeros(M); sin = np.zeros(M)
    for m, (_eid, a, b, _sec) in enumerate(ELEMENTS_1BASED):
        i0, j0 = a - 1, b - 1
        xi, yi = NODE_COORDS[i0]; xj, yj = NODE_COORDS[j0]
        L = float(np.hypot(xj - xi, yj - yi))
        ni[m], nj[m] = i0, j0
        L0[m] = L; cos[m] = (xj - xi) / L; sin[m] = (yj - yi) / L
    return {'ni': ni, 'nj': nj, 'L0': L0, 'cos': cos, 'sin': sin}


MEMBER_GEOM = _member_geometry()
ELEM_SEGMENTS_UNDEF = np.array(
    [[NODE_COORDS[a - 1], NODE_COORDS[b - 1]] for _e, a, b, _s in ELEMENTS_1BASED])


def member_strain(u_field):
    """u (N,102,2) 또는 (102,2) → 부재 축변형률 (N,201) 또는 (201,)."""
    single = (u_field.ndim == 2)
    u = u_field[None] if single else u_field
    g = MEMBER_GEOM
    dux = u[:, g['nj'], 0] - u[:, g['ni'], 0]
    duy = u[:, g['nj'], 1] - u[:, g['ni'], 1]
    eps = (g['cos'] * dux + g['sin'] * duy) / g['L0']
    return eps[0] if single else eps


def member_stress(u_field):
    """축응력 σ = E·ε  (MPa). 인장 +, 압축 −."""
    return E_STEEL_MPA * member_strain(u_field)


# ──────────────────────────────────────────────────────────────────
# 모드①② — 앙상블 예측 결과(npz) 로더
# ──────────────────────────────────────────────────────────────────
class EnsembleResults:
    def __init__(self, npz_path=NPZ_PATH):
        d = np.load(npz_path)
        self.u_mean = d["u_pred_mean"]          # (T,102,2)
        self.u_std  = d["u_pred_std"]           # (T,102,2)
        self.eps_meas = d["eps_meas"]           # (T,4)
        self.th_meas  = d["th_meas"]            # (T,4)
        self.c_mean = d["c_pred_mean"]          # (T,4)
        self.c_std  = d["c_pred_std"]           # (T,4)
        self.idx_te = d["idx_te"]               # (T,)
        self.seeds  = d["seeds"]
        self.T = self.u_mean.shape[0]
        # 부재 응력 (평균장 기준) — 미리 계산
        self.sigma_mean = member_stress(self.u_mean)        # (T,201)
        # 모드③(최근접) 용 입력 공간 = branch 입력 = concat(eps_meas, th_meas)
        self.X_in = np.concatenate([self.eps_meas, self.th_meas], axis=1)   # (T,8)
        self._in_std = self.X_in.std(axis=0)
        self._in_std[self._in_std < 1e-12] = 1.0

    def nearest(self, X_phys):
        """입력 X_phys(8) 에 가장 가까운 관측 시점 인덱스 + 정규화 거리.

        채널 스케일(strain~1e-5 vs rot~1e-3)이 달라 채널별 std 로 정규화 후 유클리드.
        """
        X_phys = np.asarray(X_phys, dtype=np.float64).reshape(-1)
        z = (self.X_in - X_phys) / self._in_std            # (T,8)
        d = np.linalg.norm(z, axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def case(self, i):
        """시점 i 의 결과 묶음."""
        return {
            "u": self.u_mean[i], "u_std": self.u_std[i],
            "sigma": self.sigma_mean[i],
            "c_mean": self.c_mean[i], "c_std": self.c_std[i],
            "eps_meas": self.eps_meas[i], "th_meas": self.th_meas[i],
            "idx_te": int(self.idx_te[i]),
        }

    def example_indices(self):
        """대표 케이스 인덱스 (런타임 선정)."""
        eps = self.eps_meas; th = self.th_meas
        mag_e = np.linalg.norm(eps, axis=1)
        mag_t = np.linalg.norm(th, axis=1)
        unc = np.linalg.norm(self.u_std.reshape(self.T, -1), axis=1)
        return {
            "정적 (변형 최소)":      int(np.argmin(mag_e + mag_t)),
            "최대 변형률 응답":      int(np.argmax(mag_e)),
            "최대 회전 응답":        int(np.argmax(mag_t)),
            "좌측 우세 변형":        int(np.argmax(eps[:, 0] - eps[:, 3])),
            "우측 우세 변형":        int(np.argmax(eps[:, 3] - eps[:, 0])),
            "불확실성 최대":         int(np.argmax(unc)),
        }


# ──────────────────────────────────────────────────────────────────
# 입력 헬퍼 (모드③ 최근접 검색)
# ──────────────────────────────────────────────────────────────────
def build_branch_from_dstrain_drot(dstrain_ue, drot_deg):
    """Δε(µε)·Δθ(°) → X_phys(8) = [Δε·1e-6 ×4, Δθ·π/180 ×4]."""
    eps = np.asarray(dstrain_ue, dtype=np.float64) * STRAIN_RAW_TO_PHYS
    rot = np.asarray(drot_deg,   dtype=np.float64) * (np.pi / 180.0)
    return np.concatenate([eps, rot])


def scaler_available():
    return os.path.exists(SCALER_PATH)
