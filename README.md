# 역전파 연구실 (backprop_lab)

신경망이 **어떻게 스스로 고쳐지는지**를 초보자도 따라올 수 있게 보여 주는 해설 저장소입니다.
읽는 순서: 텐서 연구실(`what_is_tensor`) → 활성화 함수(`neuron_reaction`) → **역전파 연구실**

사이트: https://adminmrobo.github.io/backprop_lab/

## 지금 있는 것

| 편 | 내용 |
|---|---|
| 실험 노트 00 · `index.html` | 2012년 워싱턴 DC 자전거 대여 366일을 기온 하나로 예측하는 신경망(1-4-4-1, 손잡이 33개)으로, **손잡이 하나씩 · 무작위 · 역전파** 세 학습 방법을 같은 조건에서 비교 |
| 해설편 | 연구실 식구가 진행하는 해설 페이지 (준비 중) |

## 실험에서 확인한 것

- 원본 영상(Datasciencebrain, "Backpropagation, explained simply")의 숫자는 이 데이터와 일치합니다. 30 °C 하루 = 2012-06-23, 실제 7,458대.
- 역전파로 구한 기울기와 하나씩 돌려 구한 기울기는 33개 모두 같습니다. 역전파는 같은 답을 싸게 냅니다.
- 400번 학습: 평균 오차 5,638 → 944대. 역전파는 계산 약 800번, 하나씩 방법은 13,600번.
- 손잡이가 적으면 무작위도 통하지만, 손잡이가 늘수록 역전파만 남습니다(`experiments/scaling.json`).

## 직접 돌려 보기

```bash
pip install numpy
python3 experiments/run_experiment.py experiments/day.csv experiments/results.json
python3 experiments/scaling.py experiments/day.csv
python3 experiments/build_page.py      # results.json·scaling.json → index.html
```

## 출처

- 데이터: Fanaee-T, H. & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 2, 113–127. UCI Machine Learning Repository, Bike Sharing Dataset (CC BY 4.0). `experiments/day.csv`는 원본 그대로입니다.
- 착안: Datasciencebrain (@datasciencebrain), "Backpropagation, explained simply" (2026).
- 역전파: Rumelhart, Hinton & Williams (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.

글 · 안상선 ((주)M-Robo 대표)
