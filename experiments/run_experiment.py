"""역전파 연구실 실험: 2012년 워싱턴 DC 자전거 대여(일별, 366일)를 기온 하나로 예측.

신경망: 입력 1 → 은닉 4 (ReLU) → 은닉 4 (ReLU) → 출력 1  (가중치·편향 33개)
세 가지 학습 방법을 같은 시작점·같은 '정방향 계산 예산'에서 비교한다.
  A. 손잡이 하나씩 (유한 차분)   B. 무작위로 돌려 보기   C. 역전파
데이터: UCI Bike Sharing Dataset (Fanaee-T & Gama, 2013), day.csv
"""
import json, sys, time
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "day.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results.json"

# ---------- 데이터 ----------
raw = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
d12 = raw[raw["yr"] == 1]                       # 2012년만
temp_c = d12["temp"] * 41.0                     # 원자료는 °C/41 로 정규화돼 있음
rides = d12["cnt"].astype(float)
X = (temp_c / 41.0).reshape(-1, 1)              # 입력: 0~1
Y = (rides / 1000.0).reshape(-1, 1)             # 출력: 천 건 단위
N = len(Y)

# ---------- 신경망 ----------
SIZES = [(1, 4), (4, 4), (4, 1)]

def init(seed=0):
    rng = np.random.default_rng(seed)
    p = []
    for i, o in SIZES:
        p.append(rng.normal(0, np.sqrt(2.0 / i), (i, o)))   # He 초기화
        p.append(np.zeros(o) + 0.1)
    return p

def flat(p):   return np.concatenate([a.ravel() for a in p])
def unflat(v):
    out, k = [], 0
    for i, o in SIZES:
        out.append(v[k:k + i * o].reshape(i, o)); k += i * o
        out.append(v[k:k + o].copy()); k += o
    return out

def forward(p, x):
    W1, b1, W2, b2, W3, b3 = p
    z1 = x @ W1 + b1; h1 = np.maximum(z1, 0)
    z2 = h1 @ W2 + b2; h2 = np.maximum(z2, 0)
    yhat = h2 @ W3 + b3
    return yhat, (x, z1, h1, z2, h2)

def loss(p):  # 평균제곱오차 (천 건 단위)
    yhat, _ = forward(p, X)
    return float(np.mean((yhat - Y) ** 2))

def mae_rides(p):
    yhat, _ = forward(p, X)
    return float(np.mean(np.abs(yhat - Y)) * 1000)

def backprop(p):
    W1, b1, W2, b2, W3, b3 = p
    yhat, (x, z1, h1, z2, h2) = forward(p, X)
    g = 2 * (yhat - Y) / N                       # dL/dyhat
    gW3 = h2.T @ g;  gb3 = g.sum(0)
    g2 = (g @ W3.T) * (z2 > 0)                   # ReLU 기울기: 켜져 있으면 1, 꺼져 있으면 0
    gW2 = h1.T @ g2; gb2 = g2.sum(0)
    g1 = (g2 @ W2.T) * (z1 > 0)
    gW1 = x.T @ g1;  gb1 = g1.sum(0)
    return [gW1, gb1, gW2, gb2, gW3, gb3]

def numgrad(p, eps=1e-4):                        # 손잡이 하나씩: 기준 1번 + 손잡이당 1번
    v = flat(p); base = loss(p); g = np.zeros_like(v)
    for j in range(len(v)):
        w = v.copy(); w[j] += eps
        g[j] = (loss(unflat(w)) - base) / eps
    return unflat(g), len(v) + 1

P0 = init(seed=7)
NPARAM = len(flat(P0))
LR = 0.05
ROUNDS = 400

def predict_at(p, c):
    return float(forward(p, np.array([[c / 41.0]]))[0][0, 0] * 1000)

# ---------- 한 번의 역전파를 손으로 따라가기 (30 °C 하루) ----------
day30 = int(np.argmin(np.abs(temp_c - 30.0)))
trace = {
    "date": str(d12["dteday"][day30]), "temp_c": round(float(temp_c[day30]), 1),
    "real": int(rides[day30]), "guess_at_start": round(predict_at(P0, float(temp_c[day30]))),
}

# ---------- C. 역전파 ----------
def train_backprop(rounds):
    p = [a.copy() for a in P0]; hist = [mae_rides(p)]; t = time.perf_counter()
    for _ in range(rounds):
        g = backprop(p); p = [a - LR * b for a, b in zip(p, g)]; hist.append(mae_rides(p))
    return p, hist, time.perf_counter() - t

# ---------- A. 손잡이 하나씩 ----------
def train_fd(rounds):
    p = [a.copy() for a in P0]; hist = [mae_rides(p)]; t = time.perf_counter(); fw = 0
    for _ in range(rounds):
        g, n = numgrad(p); fw += n; p = [a - LR * b for a, b in zip(p, g)]; hist.append(mae_rides(p))
    return p, hist, time.perf_counter() - t, fw

# ---------- B. 무작위로 돌려 보기 (같은 정방향 예산) ----------
def train_random(budget, step=0.05, seed=1):
    rng = np.random.default_rng(seed)
    v = flat(P0); best = loss(P0); hist = [mae_rides(P0)]; used = 1
    while used < budget:
        w = v + rng.normal(0, step, v.shape); l = loss(unflat(w)); used += 1
        if l < best: v, best = w, l
        if used % (NPARAM + 1) == 0: hist.append(mae_rides(unflat(v)))
    return unflat(v), hist

pC, hC, tC = train_backprop(ROUNDS)
pA, hA, tA, fwA = train_fd(ROUNDS)
budget_C = ROUNDS * 2                     # 역전파: 라운드당 정방향 1 + 역방향 1
pB_same, hB_same = train_random(budget_C)          # 역전파와 같은 예산
pB_big, hB_big = train_random(fwA)                 # 손잡이 하나씩과 같은 예산

# 첫 라운드 기울기: 역전파 vs 손잡이 하나씩 이 같은 답을 내는지
gb = flat(backprop(P0)); gn = flat(numgrad(P0)[0])

# 체인룰 한 경로: 출력 ← 은닉2 j ← 은닉1 i (가장 큰 경로)
W1, b1, W2, b2, W3, b3 = P0
x0 = np.array([[temp_c[day30] / 41.0]])
_, (_, z1, h1, z2, h2) = forward(P0, x0)
best_path = None
for i in range(4):
    for j in range(4):
        if z1[0, i] > 0 and z2[0, j] > 0:
            s = abs(W2[i, j] * W3[j, 0])
            if best_path is None or s > best_path[0]:
                best_path = (s, i, j)
_, i, j = best_path
chain = {"hidden1": i + 1, "hidden2": j + 1,
         "w_h1_to_h2": round(float(W2[i, j]), 3), "w_h2_to_out": round(float(W3[j, 0]), 3),
         "product": round(float(W2[i, j] * W3[j, 0]), 3)}

sandy = int(np.argmin(rides))
res = {
    "data": {"days": N, "year": 2012, "mean_rides": round(float(rides.mean())),
             "min_day": {"date": str(d12["dteday"][sandy]), "rides": int(rides[sandy]),
                         "temp_c": round(float(temp_c[sandy]), 1)},
             "max_day": {"date": str(d12["dteday"][int(np.argmax(rides))]), "rides": int(rides.max())}},
    "network": {"layers": "1-4-4-1 ReLU", "params": NPARAM, "lr": LR, "rounds": ROUNDS},
    "trace_30c": trace, "chain_path": chain,
    "grad_check_max_abs_diff": float(np.max(np.abs(gb - gn))),
    "backprop": {"mae_start": round(hC[0]), "mae_end": round(hC[-1]), "forward_equiv": budget_C,
                 "seconds": round(tC, 3), "curve": [round(v) for v in hC[::10]],
                 "guess_30c_end": round(predict_at(pC, trace["temp_c"]))},
    "one_knob_at_a_time": {"mae_end": round(hA[-1]), "forward_passes": fwA,
                           "seconds": round(tA, 3), "curve": [round(v) for v in hA[::10]]},
    "random_same_budget_as_backprop": {"mae_end": round(hB_same[-1]), "forward_passes": budget_C},
    "random_same_budget_as_one_knob": {"mae_end": round(hB_big[-1]), "forward_passes": fwA},
    "fit_curve": [{"temp_c": c, "pred": round(predict_at(pC, c))} for c in range(0, 37, 2)],
    "scatter": [{"temp_c": round(float(t), 1), "rides": int(r)} for t, r in zip(temp_c, rides)],
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
summary = {k: v for k, v in res.items() if k not in ("scatter", "fit_curve")}
for k in ("backprop", "one_knob_at_a_time"):
    summary[k] = {kk: vv for kk, vv in summary[k].items() if kk != "curve"}
print(json.dumps(summary, ensure_ascii=False, indent=1))
