"""results.json · scaling.json → ../index.html (실험 노트 페이지) 를 만든다."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
S = json.load(open(os.path.join(HERE, "scaling.json"), encoding="utf-8"))

def fmt(n):
    return f"{n:,}"

bp, fd = R["backprop"], R["one_knob_at_a_time"]
rs_same, rs_big = R["random_same_budget_as_backprop"], R["random_same_budget_as_one_knob"]
t30, ch, net = R["trace_30c"], R["chain_path"], R["network"]

def knob_cell(r):
    if not r["one_knob_updates"]:
        return "한 번도 못 고침"
    return f"{fmt(r['one_knob_mae'])} <small>({r['one_knob_updates']}번 고침)</small>"

scale_rows = "\n".join(
    f"<tr><td>{fmt(r['params'])}</td><td><b>{fmt(r['backprop_mae'])}</b></td>"
    f"<td>{knob_cell(r)}</td><td>{fmt(r['random_mae'])}</td></tr>"
    for r in S["rows"])

data_js = json.dumps({
    "scatter": [[p["temp_c"], p["rides"]] for p in R["scatter"]],
    "fit": [[p["temp_c"], p["pred"]] for p in R["fit_curve"]],
    "bp": bp["curve"], "fd": fd["curve"], "params": net["params"],
}, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>역전파 연구실 · 실험 노트 — 손잡이 33개를 한 번에 돌리는 법</title>
<meta name="description" content="2012년 워싱턴 DC 자전거 대여 366일을 기온 하나로 예측하는 작은 신경망으로, 역전파·손잡이 하나씩·무작위 세 가지 학습 방법을 같은 조건에서 비교한 실험 노트.">
<style>
:root{{--bg:#fbfaf7;--ink:#1d2327;--sub:#5b6670;--line:#e3e0d8;--acc:#d9822b;--acc2:#2f6f8f;--bad:#b0413e;--card:#fff}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 -apple-system,"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif}}
main{{max-width:760px;margin:0 auto;padding:28px 18px 60px}}
.kicker{{color:var(--acc);font-weight:700;letter-spacing:.04em;font-size:.85rem}}
h1{{font-size:1.75rem;line-height:1.3;margin:.2em 0 .3em}}
h2{{font-size:1.25rem;margin:2.2em 0 .5em;padding-top:.6em;border-top:1px solid var(--line)}}
.by{{color:var(--sub);font-size:.9rem}}
.box{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}}
.three{{border-left:4px solid var(--acc)}}
.three ol{{margin:.2em 0;padding-left:1.2em}}
.note{{background:#fff7ec;border:1px solid #f0d9b8}}
table{{width:100%;border-collapse:collapse;font-size:.92rem;margin:10px 0}}
th,td{{border-bottom:1px solid var(--line);padding:7px 6px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{color:var(--sub);font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
.m{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}
.m b{{display:block;font-size:1.05rem}} .m span{{color:var(--sub);font-size:.85rem}}
.m.win{{border-color:var(--acc)}}
svg{{width:100%;height:auto;display:block}}
.cap{{color:var(--sub);font-size:.85rem;margin-top:4px}}
.eq{{font-family:ui-monospace,Menlo,monospace;background:#f3f1ec;border-radius:8px;padding:10px 12px;overflow-x:auto}}
code{{background:#f3f1ec;border-radius:4px;padding:0 4px}}
small{{color:var(--sub)}}
a{{color:var(--acc2)}}
@media (max-width:560px){{.grid{{grid-template-columns:1fr}} h1{{font-size:1.45rem}}}}
</style>
</head>
<body>
<main>
<div class="kicker">역전파 연구실 · 실험 노트 00</div>
<h1>손잡이 33개를 한 번에 돌리는 법<br><small>— 자전거 대여 366일로 비교한 역전파와 그렇지 않은 방법</small></h1>
<p class="by">글 · 안상선 ((주)M-Robo 대표) · 2026-11 · 해설편(연구실 식구 진행)은 이어서 공개합니다</p>

<div class="box three">
<b>세 줄로 먼저</b>
<ol>
<li>신경망의 가중치는 <b>손잡이</b>입니다. 학습은 예측이 덜 틀리도록 손잡이를 돌리는 일입니다.</li>
<li>손잡이를 하나씩 돌려 보면 손잡이 수만큼 계산을 다시 해야 합니다. <b>역전파</b>는 앞으로 한 번, 뒤로 한 번 계산해 모든 손잡이의 방향과 세기를 한꺼번에 알려 줍니다.</li>
<li>실제 데이터로 돌려 보니 손잡이가 적을 때는 어떤 방법이든 되지만, 손잡이가 늘면 <b>역전파만 남습니다</b>.</li>
</ol>
</div>

<h2>1. 문제: 기온만 보고 오늘 자전거가 몇 대 나갈까</h2>
<p>워싱턴 DC 공공자전거의 <b>2012년 {R['data']['days']}일</b> 기록입니다. 하루 평균 {fmt(R['data']['mean_rides'])}대가 나갔고, 가장 많은 날은 {R['data']['max_day']['date']}({fmt(R['data']['max_day']['rides'])}대)였습니다. 점 하나가 하루입니다.</p>
<div class="box"><svg id="sc" viewBox="0 0 640 330"></svg>
<div class="cap">주황 선은 역전파로 {net['rounds']}번 학습한 신경망의 예측. 바닥에 홀로 떨어진 점은 {R['data']['min_day']['date']}, 허리케인 샌디가 온 날({R['data']['min_day']['rides']}대)입니다.</div></div>
<p>점들을 보면 기온이 오를수록 많이 타다가, 25 °C를 넘으면 너무 더워 조금 줄어듭니다. 직선 하나로는 이런 꺾임을 못 그립니다. 그래서 꺾을 수 있는 작은 신경망을 씁니다. 다만 {net['rounds']}번 학습한 이 선은 아직 더운 날의 내리막까지는 따라가지 못했습니다. 더 오래 학습하면 잡아 갑니다.</p>

<h2>2. 신경망: 입력 1 → 4 → 4 → 출력 1</h2>
<p>기온 하나가 들어가 은닉층 두 개(각 4칸)를 지나 예측 하나가 나옵니다. 은닉층은 ReLU를 씁니다(0보다 작으면 0, 크면 그대로). 돌릴 손잡이(가중치와 편향)는 모두 <b>{net['params']}개</b>입니다.</p>
<div class="eq">8개 (입력→은닉1) + 20개 (은닉1→은닉2) + 5개 (은닉2→출력) = {net['params']}개</div>

<h2>3. 손잡이를 돌리는 세 가지 방법</h2>
<div class="grid">
<div class="m"><b>A. 하나씩</b><span>손잡이 하나를 살짝 돌리고 오차를 다시 잰다. 이걸 {net['params']}번. 한 번 고치는 데 계산 {net['params']+1}번.</span></div>
<div class="m"><b>B. 무작위</b><span>모든 손잡이를 아무렇게나 살짝 돌려 보고, 나아졌으면 남긴다. 운에 맡긴다.</span></div>
<div class="m win"><b>C. 역전파</b><span>앞으로 1번(예측), 뒤로 1번(책임 나누기). 한 번 고치는 데 계산 약 2번.</span></div>
</div>

<h2>4. 30 °C 하루를 따라가 보기</h2>
<p>{t30['date']}, 기온 {t30['temp_c']} °C, 실제로는 <b>{fmt(t30['real'])}대</b>가 나갔습니다. 학습 전 신경망은 손잡이가 아무렇게나 놓여 있어 <b>{fmt(t30['guess_at_start'])}대</b>라고 답했습니다. 한참 틀렸습니다.</p>
<p><b>앞으로(순전파)</b>: 기온이 층을 지나며 곱해지고 더해져 예측이 됩니다.<br>
<b>뒤로(역전파)</b>: "예측이 {fmt(t30['real'] - t30['guess_at_start'])}대 모자란다"는 소식이 출력에서 거꾸로 흘러가며, 각 손잡이가 얼마나 책임이 있는지 나눕니다. 길을 따라 만나는 효과를 <b>곱하는 것</b>, 이것이 연쇄법칙(체인룰)입니다.</p>
<div class="eq">은닉1의 {ch['hidden1']}번 칸 → 은닉2의 {ch['hidden2']}번 칸 → 출력<br>
{ch['w_h1_to_h2']} × ({ch['w_h2_to_out']}) = {ch['product']}</div>
<p>곱이 음수라는 것은 "이 길로 신호를 키우면 예측이 오히려 줄어든다"는 뜻입니다. 그래서 이 길의 손잡이는 반대 방향으로 돌려야 합니다. 역전파는 이런 계산을 {net['params']}개 손잡이 모두에 대해 한 번에 해 줍니다.</p>
<div class="box note">검산: 역전파로 구한 기울기와 손잡이를 하나씩 돌려 구한 기울기를 비교했더니 {net['params']}개 모두 같았습니다(차이 최대 {R['grad_check_max_abs_diff']:.0e}). 역전파는 <b>다른 답</b>을 내는 방법이 아니라 <b>같은 답을 싸게</b> 내는 방법입니다.</div>

<h2>5. 결과: 같은 답, 다른 비용</h2>
<div class="box"><svg id="lc" viewBox="0 0 640 300"></svg>
<div class="cap">가로축은 쓴 계산량(순전파 횟수), 세로축은 하루 평균 오차(대). 두 방법은 한 걸음마다 똑같이 고치지만, 하나씩 방법은 한 걸음에 계산을 {net['params']+1}번 씁니다.</div></div>
<table>
<tr><th>{net['rounds']}번 고쳤을 때</th><th>평균 오차</th><th>쓴 계산</th><th>걸린 시간</th></tr>
<tr><td>C. 역전파</td><td>{fmt(bp['mae_start'])} → <b>{fmt(bp['mae_end'])}</b></td><td>약 {fmt(bp['forward_equiv'])}번</td><td>{bp['seconds']}초</td></tr>
<tr><td>A. 하나씩</td><td>{fmt(bp['mae_start'])} → {fmt(fd['mae_end'])}</td><td>{fmt(fd['forward_passes'])}번</td><td>{fd['seconds']}초</td></tr>
</table>
<p>학습이 끝난 신경망은 30 °C 그날을 <b>{fmt(bp['guess_30c_end'])}대</b>로 예측합니다(실제 {fmt(t30['real'])}대).</p>

<h2>6. 손잡이가 늘어나면</h2>
<p>은닉층 폭만 키우고 세 방법에 <b>똑같이 계산 {fmt(S['budget_forward_passes'])}번어치</b>를 줬습니다. 숫자는 하루 평균 오차(대)입니다.</p>
<table>
<tr><th>손잡이 수</th><th>C. 역전파</th><th>A. 하나씩</th><th>B. 무작위</th></tr>
{scale_rows}
</table>
<div class="box note"><b>솔직한 발견</b> — 손잡이 {net['params']}개짜리 작은 문제에서는 무작위도 꽤 잘합니다(같은 예산에서 {fmt(rs_same['mae_end'])}대). 기온 하나로 맞히는 쉬운 문제라 운으로도 닿기 때문입니다. 그러나 손잡이가 늘수록 무작위는 뒤처지고, 하나씩 방법은 한 번도 고치지 못합니다. GPT-3의 손잡이는 1,750억 개입니다. 하나씩 방법이라면 한 걸음에 계산이 1,750억 번 필요하지만, 역전파는 여전히 앞으로 한 번, 뒤로 한 번입니다.</div>

<h2>7. 원본 영상과 맞춰 보기</h2>
<table>
<tr><th>영상의 내용</th><th>이 실험</th></tr>
<tr><td style="text-align:left">366일, 워싱턴 DC, 2012년</td><td>같은 데이터(UCI Bike Sharing)</td></tr>
<tr><td style="text-align:left">30 °C 하루, 실제 7,458대</td><td>{t30['date']}, {fmt(t30['real'])}대 — 일치</td></tr>
<tr><td style="text-align:left">신경망 1-4-4-1</td><td>같은 구조, ReLU</td></tr>
<tr><td style="text-align:left">평균 오차 5,169 → 839</td><td>{fmt(bp['mae_start'])} → {fmt(bp['mae_end'])} ({net['rounds']}번). 시작값은 처음 손잡이 위치에 따라 다름. 이 데이터의 바닥은 830대 안팎</td></tr>
<tr><td style="text-align:left">체인룰 예: 0.83 × 0.36</td><td>그 영상 신경망의 값. 이 실험에서는 {ch['w_h1_to_h2']} × ({ch['w_h2_to_out']})</td></tr>
</table>

<h2>8. 직접 돌려 보기</h2>
<p>코드는 이 저장소의 <code>experiments/</code>에 있습니다. 파이썬과 numpy만 있으면 됩니다.</p>
<div class="eq">python3 experiments/run_experiment.py experiments/day.csv experiments/results.json<br>python3 experiments/scaling.py experiments/day.csv</div>

<h2>출처</h2>
<p><small>데이터: Fanaee-T, H. &amp; Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. <i>Progress in Artificial Intelligence</i>, 2, 113–127. UCI Machine Learning Repository, Bike Sharing Dataset (CC BY 4.0).<br>
착안: Datasciencebrain (@datasciencebrain), "Backpropagation, explained simply" 영상(2026). 영상의 숫자는 이 저장소의 코드로 다시 계산해 확인했습니다.<br>
역전파: Rumelhart, D. E., Hinton, G. E. &amp; Williams, R. J. (1986). Learning representations by back-propagating errors. <i>Nature</i>, 323, 533–536.</small></p>
<p class="by">글 · 안상선 ((주)M-Robo 대표) · <a href="https://github.com/adminmrobo/backprop_lab">github.com/adminmrobo/backprop_lab</a></p>
</main>
<script>
const D={data_js};
const NS="http://www.w3.org/2000/svg";
function el(t,a,p){{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);p.appendChild(e);return e}}
function axes(s,W,H,m,xr,yr,xt,yt,xl,yl){{
 const X=v=>m.l+(v-xr[0])/(xr[1]-xr[0])*(W-m.l-m.r), Y=v=>H-m.b-(v-yr[0])/(yr[1]-yr[0])*(H-m.t-m.b);
 el('line',{{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#bbb'}},s);el('line',{{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#bbb'}},s);
 xt.forEach(v=>{{const t=el('text',{{x:X(v),y:H-m.b+18,'text-anchor':'middle','font-size':12,fill:'#5b6670'}},s);t.textContent=xl(v)}});
 yt.forEach(v=>{{el('line',{{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:'#eee'}},s);const t=el('text',{{x:m.l-6,y:Y(v)+4,'text-anchor':'end','font-size':12,fill:'#5b6670'}},s);t.textContent=yl(v)}});
 return [X,Y]}}
(function(){{const s=document.getElementById('sc'),m={{l:46,r:12,t:12,b:30}};
 const [X,Y]=axes(s,640,330,m,[0,37],[0,9000],[5,10,15,20,25,30,35],[0,2000,4000,6000,8000],v=>v+' °C',v=>v?(v/1000)+'k':'0');
 D.scatter.forEach(p=>el('circle',{{cx:X(p[0]),cy:Y(p[1]),r:2.6,fill:'#2f6f8f','fill-opacity':.45}},s));
 el('polyline',{{points:D.fit.map(p=>X(p[0])+','+Y(Math.max(p[1],0))).join(' '),fill:'none',stroke:'#d9822b','stroke-width':3}},s);}})();
(function(){{const s=document.getElementById('lc'),m={{l:50,r:12,t:12,b:30}},n=D.params+1;
 const maxX=D.bp.length*10*2*1.0, bpX=D.bp.map((v,i)=>[i*10*2,v]), fdX=D.fd.map((v,i)=>[i*10*n,v]).filter(p=>p[0]<=maxX*4);
 const xm=Math.max(...fdX.map(p=>p[0]));
 const [X,Y]=axes(s,640,300,m,[0,xm],[0,6000],[0,Math.round(xm/4),Math.round(xm/2),Math.round(3*xm/4),xm],[0,2000,4000,6000],v=>v.toLocaleString(),v=>v.toLocaleString());
 el('polyline',{{points:fdX.map(p=>X(p[0])+','+Y(p[1])).join(' '),fill:'none',stroke:'#b0413e','stroke-width':2.5}},s);
 el('polyline',{{points:bpX.map(p=>X(p[0])+','+Y(p[1])).join(' '),fill:'none',stroke:'#d9822b','stroke-width':3}},s);
 const a=el('text',{{x:X(bpX[bpX.length-1][0])+6,y:Y(bpX[bpX.length-1][1])-8,'font-size':13,fill:'#d9822b','font-weight':700}},s);a.textContent='역전파';
 const b=el('text',{{x:X(xm*0.55),y:Y(fdX[Math.floor(fdX.length*0.55)][1])-10,'font-size':13,fill:'#b0413e','font-weight':700}},s);b.textContent='하나씩';}})();
</script>
</body>
</html>
"""
open(os.path.join(HERE, "..", "index.html"), "w", encoding="utf-8").write(html)
print("index.html", len(html))
