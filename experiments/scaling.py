"""신경망이 커질수록 세 방법의 차이가 어떻게 벌어지는가.
같은 데이터(2012년 366일), 은닉층 폭만 바꿔 가며 같은 '정방향 계산 예산'을 준다.
"""
import json, sys
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "day.csv"
raw = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
d = raw[raw["yr"] == 1]
X = d["temp"].reshape(-1, 1).astype(float)
Y = (d["cnt"] / 1000.0).reshape(-1, 1)
N = len(Y)

def make(width, seed=7):
    sizes = [(1, width), (width, width), (width, 1)]
    rng = np.random.default_rng(seed); p = []
    for i, o in sizes:
        p += [rng.normal(0, np.sqrt(2 / i), (i, o)), np.zeros(o) + 0.1]
    return sizes, p

def fwd(p, x):
    W1, b1, W2, b2, W3, b3 = p
    z1 = x @ W1 + b1; h1 = np.maximum(z1, 0); z2 = h1 @ W2 + b2; h2 = np.maximum(z2, 0)
    return h2 @ W3 + b3, (z1, h1, z2, h2)

def mse(p): return float(np.mean((fwd(p, X)[0] - Y) ** 2))
def mae(p): return float(np.mean(np.abs(fwd(p, X)[0] - Y)) * 1000)

def grad(p):
    W1, b1, W2, b2, W3, b3 = p
    yh, (z1, h1, z2, h2) = fwd(p, X); g = 2 * (yh - Y) / N
    g2 = (g @ W3.T) * (z2 > 0); g1 = (g2 @ W2.T) * (z1 > 0)
    return [X.T @ g1, g1.sum(0), h1.T @ g2, g2.sum(0), h2.T @ g, g.sum(0)]

BUDGET = 2000   # 정방향 계산 2,000번어치
rows = []
for width in [4, 16, 64, 256]:
    sizes, P0 = make(width)
    n = sum(a.size for a in P0)
    lr = 0.05 if width <= 16 else 0.02
    # 역전파: 라운드당 정방향 1 + 역방향 1 ≈ 정방향 2번어치로 계산
    p = [a.copy() for a in P0]
    for _ in range(BUDGET // 2):
        p = [a - lr * b for a, b in zip(p, grad(p))]
    bp = mae(p)
    # 손잡이 하나씩: 한 번 고치는 데 정방향 n+1번 → 예산 안에서 몇 번 고치나
    fd_updates = BUDGET // (n + 1)
    p = [a.copy() for a in P0]
    for _ in range(fd_updates):
        p = [a - lr * b for a, b in zip(p, grad(p))]   # 같은 기울기이므로 결과는 동일, 횟수만 제한
    fd = mae(p)
    # 무작위: 예산만큼 시도
    rng = np.random.default_rng(1)
    v = np.concatenate([a.ravel() for a in P0]); best = mse(P0)
    def unf(v):
        out, k = [], 0
        for i, o in sizes:
            out.append(v[k:k+i*o].reshape(i, o)); k += i*o; out.append(v[k:k+o]); k += o
        return out
    step = 0.05 / np.sqrt(n / 33)
    for _ in range(BUDGET):
        w = v + rng.normal(0, step, v.shape); l = mse(unf(w))
        if l < best: v, best = w, l
    rs = mae(unf(v))
    rows.append({"width": width, "params": int(n), "start_mae": round(mae(P0)),
                 "backprop_mae": round(bp), "one_knob_updates": int(fd_updates),
                 "one_knob_mae": round(fd), "random_mae": round(rs)})
    print(rows[-1])
import os
json.dump({"budget_forward_passes": BUDGET, "rows": rows},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaling.json"), "w"), indent=1)
