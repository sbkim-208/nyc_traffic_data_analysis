# NYC Traffic Data Analysis

![NYC 교통: PM 첨두시 혼잡도 및 센서 오류 발생지역](outputs/congestion_sensor_map.png)

*PM 첨두시(15~19시) 혼잡도(색상, 원활→혼잡)와 센서 오류 발생지역(×, 오류율 50% 이상)을 125개 구간 전체 네트워크에 표시. 실행: `python congestion_error_map.py`*

## 1. 센서 신뢰도

데이터: NYC DOT Traffic Speeds NBE (Socrata `i4gi-tjb9`), 2024-04-01 ~ 2024-08-01 (4개월), 4,233,169행

센서 신뢰도 = (전체 행 수 − `status == -101` 행 수) / 전체 행 수

### 전체 신뢰도

| 전체 데이터 수 | status=-101 개수 | 신뢰도          |
| -------------- | ---------------- | --------------- |
| 4,233,169      | 1,056,062        | 0.7505 (75.05%) |

### Borough별 신뢰도

| borough       | total     | error_count | error_rate | 신뢰도 |
| ------------- | --------- | ----------- | ---------- | ------ |
| Manhattan     | 904,947   | 395,470     | 0.4370     | 56.3%  |
| Brooklyn      | 370,312   | 140,285     | 0.3788     | 62.1%  |
| Staten Island | 871,438   | 304,459     | 0.3494     | 65.1%  |
| Bronx         | 798,345   | 130,361     | 0.1633     | 83.7%  |
| Queens        | 1,288,127 | 85,487      | 0.0664     | 93.4%  |

Queens가 가장 안정적이고 Manhattan이 가장 불안정 — 6.6배 차이. 시각화: `borough_reliability.png`

### Borough별 문제 구간

표본 100건 미만(노이즈성 구간)은 제외하고 borough별 오류율 top 5만 집계.

-   Manhattan, Brooklyn, Staten Island, Bronx는 top 구간 대부분이 **error_rate = 1.0** (4개월 내내 정상 응답 0건 — 완전 고장)인 반면, **Queens는 top 5 전부 100% 미만(19~33%)** — 완전 고장 센서가 없고 부분적 불안정만 있음. Queens의 전체 신뢰도가 가장 높은 이유와 일치.
-   완전 고장 구간 예: `VNB` (Verrazzano-Narrows Bridge), `MDE`(Major Deegan Expwy), `WSE`(West Shore Expwy), `SIE`(Staten Island Expwy), `Westside Hwy`, `Lincoln Tunnel` 등

시각화: `borough_problem_roads.png` (borough별 small multiples)

실행: `python sensor_analysis.py`

### 혼잡이 센서를 고장나게 하는가?

가설: GWB 접근로, Lincoln Tunnel, West Side Highway(9A) 같은 유명 혼잡 회랑의 구간일수록 통행량 부하로 센서가 더 많이 고장난다.

부분 고장 상태(일부 유효 데이터가 남아있는) Manhattan 구간 두 곳으로 검증 — Lincoln Tunnel W center tube(오류율 94.11%), 12th Ave S 57th–45th(오류율 93.9%).

**Lincoln Tunnel W center tube (id 329)** — 전체 34,614건 중 valid 2,039건(5.89%):

-   valid 구간 속도: 평균 17.19 mph, 중앙값 16.15 mph(범위 1.2~39.8 mph) — 실제로 낮음, 남아있는 데이터만 보면 혼잡이 맞음
-   하지만 **시간대별 오류율은 러시아워(8~10시 72~84%, 17~18시 86~92%)에 가장 낮고**, 새벽·심야(1~5시, 20~23시)는 거의 100% — 혼잡 가설과 정반대 패턴. 혼잡이 센서를 마모시킨다면 오류율이 혼잡 시간에 같이 올라가야 하는데 오히려 떨어짐일

**12th Ave S 57th–45th (id 106)** — valid 평균 속도 26.68 mph(덜 심각), 주간(89~92%) vs 야간(97~99%) 오류율 차이도 미미하고 뚜렷한 러시아워 패턴 없음

**결론: 가설 기각.** "혼잡이 센서를 고장낸다"는 데이터로 뒷받침되지 않음. 대신 확인된 건 **구간별로 유효 데이터가 존재하는 시간대가 다르다**는 것 — Lincoln Tunnel의 경우 유효 데이터가 러시아워에 몰려있는데, 이는 오히려 우리가 예측하려는 시간대(혼잡 시간)와 겹쳐서 유리함. 구간마다 사용 가능한 시간대를 개별 확인해야지, 균일하다고 가정하면 안 됨.

## 2. 전처리 (Data Processing)

`soobin/cleaning/preprocess.py`의 함수들.

1. **`drop_duplicate_ids`** — `link_id`, `transcom_id`, `encoded_poly_line`, `encoded_poly_line_lvls` 드롭 (미사용/중복 컬럼, 위 피처 검토 참고).
2. **`remove_dead_segments`** — 구간 전체(`id` 기준)가 `status == -101`인, 단 한 번도 정상 응답이 없던 완전 고장 구간 제거. 이런 구간은 보간(interpolate)이 불가능함 — 앵커로 삼을 실제 관측치가 시리즈 어디에도 없기 때문. 결과: **16개 구간, 487,811행 제거**.
3. **`fill_missing_speed` / `fill_missing_travel_time`** — 남은 구간에 대해 `status == -101` 행을 NaN 처리 후 구간별로 `interpolate(method="time")`, 보간이 안 닿는 양 끝은 `ffill`/`bfill` 폴백.

|              | speed   | travel_time |
| ------------ | ------- | ----------- |
| 보정 전 결측 | 396,112 | 396,077     |
| 보정 후 결측 | 0       | 0           |

최종 행 수: 4,233,169 → 완전 고장 구간 제거 후 **3,745,358**.

**주의**: 보간은 `speed`/`travel_time`만 채우고 `status`는 건드리지 않음. 그래서 1번 섹션의 신뢰도 수치는 항상 원본(raw) 데이터 기준으로 계산해야 하고, 이 전처리 결과로 재계산하면 안 됨 — `remove_dead_segments`로 최악의 구간을 이미 분모에서 뺐기 때문에 신뢰도가 실제로 좋아진 게 아니라 숫자만 높아 보이게 됨.

## 3. 시계열 분석

`data_as_of`를 datetime으로 변환해 정렬된 index로 설정한 뒤, resample(시간별/일별)과 rolling window(7일 이동평균, 14일 이동표준편차, 전주 대비 변화량)로 확인. 실행: `python time_series_analysis.py`

### Borough별 Peak Hour (평일 vs 주말)

혼잡 peak = AM(6~9시)/PM(15~19시) 구간 내에서 평균 속도가 가장 낮은 시각.

| Borough       | 구분 | AM peak       | PM peak            |
| ------------- | ---- | ------------- | ------------------ |
| Bronx         | 평일 | 9시 (31.8mph) | **16시** (25.1mph) |
| Brooklyn      | 평일 | 8시 (29.6mph) | **16시** (24.5mph) |
| Manhattan     | 평일 | 9시 (18.7mph) | **16시** (15.3mph) |
| Queens        | 평일 | 8시 (32.8mph) | **16시** (28.2mph) |
| Staten Island | 평일 | 8시 (46.4mph) | **17시** (40.8mph) |

Bronx·Brooklyn·Manhattan·Queens는 평일 PM peak가 전부 16시로 동일하고, Staten Island만 17시로 다름. 속도 절대 수준도 SI가 압도적으로 높음(AM peak 46.4mph vs Manhattan 18.7mph). SI는 지하철이 없어 통근이 Verrazzano 브릿지·페리 중심이라, 지하철 기반인 나머지 4개 borough와 혼잡 타이밍·강도가 구조적으로 다른 것으로 보임. 시각화: `weekday_weekend_comparison.png`

### Peak vs. Free-flow: Manhattan이 정말 가장 혼잡한가?

Free-flow 기준선 = 각 구간의 새벽 2~4시 평균 속도(`add_tti(method="night")`), SI(Speed Index) = 속도 / free-flow.

네트워크 전체: free-flow 평균 **44.22 mph**; AM peak(9시) 34.72 mph(free-flow 대비 78.5%, SI=0.793); PM peak(16시) 28.04 mph(63.4%, SI=0.636).

| Borough       | Free-flow(2~4시), mph | PM peak(16시), mph | 절대 하락폭 |
| ------------- | --------------------- | ------------------ | ----------- |
| Manhattan     | 25.90                 | 15.25              | **10.65**   |
| Staten Island | 55.78                 | 41.54              | 14.24       |
| Queens        | 47.24                 | 28.24              | 19.00       |
| Brooklyn      | 47.15                 | 24.51              | 22.64       |
| Bronx         | 48.03                 | 25.08              | 22.95       |

Manhattan은 하루 중 어느 시간대든 절대 속도가 가장 낮지만, 자기 자신의 free-flow 기준선 대비 하락폭은 오히려 **가장 작음** — 기준선 자체가 이미 낮기 때문(~26 mph, 고속도로가 아니라 저속 제한 도심 격자 도로). 즉 Manhattan은 "peak 시간에 제일 심한" 게 아니라 **하루 종일 균일하게 느린** 것 — 러시아워 현상이 아니라 도로망 자체의 구조적 특성. 반대로 Brooklyn/Bronx/Queens는 야간에는 고속도로급으로 빠르다가(~47~48 mph) peak 시간에 가장 크게 무너짐(19~23 mph 하락) — 실제 네트워크의 peak-hour 혼잡 문제를 짊어지고 있는 쪽은 이들임.

## 4. ML/DL 피처 엔지니어링 (Lag Features)

`soobin/analysis/ml_forecast.py::build_ml_feature_df()`.

원본 데이터는 구간(`id`)별로 대략 1분에 한 번씩 들어오지만 타임스탬프가 불규칙해서, 단순 `groupby("id").shift(n)`은 실제 시간 간격이 아니라 "n행" 단위로만 lag를 만듦. `build_ml_feature_df`는 먼저 각 구간을 규칙적인 시간 그리드(`freq`, 예: `"5min"`)로 재색인 — `interp_limit_min`까지의 공백은 시간 기반으로 보간하고, 그보다 긴 공백은 `NaN`으로 남김 — 이렇게 하면 `shift(n)`이 정확한 경과 시간에 대응하게 됨. 이후 추가하는 피처:

-   캘린더: `hour`, `dow`, `month`, `is_weekend`
-   Cyclical 인코딩: `hour_sin/cos`, `dow_sin/cos` (23시→0시 불연속 문제 제거)
-   Lag 피처: 분 단위(`min_lags`)와 일 단위(`day_lags`) lag를 설정 가능, 예: 5/10/30분, 1/30일
-   `roll_1d_mean/std`: 과거 24시간 rolling mean/std, `shift(1)` 이후에 계산해서 타깃 자기 자신의 값이 자신의 rolling 통계에 leak되지 않도록 함

모든 lag/rolling 피처는 `id`별로(`groupby("id")`) 계산해서 구간끼리 섞이지 않음.

### Lag Correlation

`corr(speed, lag_X)`, 전체 125개 구간, 4개월치, `freq="5min"`:

| Lag  | Raw correlation | Stuck-run 제외 |
| ---- | --------------- | -------------- |
| 5분  | 0.9687          | 0.9635         |
| 10분 | 0.9437          | 0.9343         |
| 30분 | 0.8964          | 0.8791         |
| 1일  | 0.7781          | 0.7427         |
| 30일 | 0.6710          | 0.6237         |

**Stuck-run 진단**: `_mask_stuck_runs()`는 동일한 값이 12개 이상 연속되는 구간(5분 그리드 기준 1시간)을 센서가 "멈췄다(stuck)"고 판단 — 실제로 변화하는 교통 신호라기보다는, `ffill`/`bfill` 폴백이 얼어붙은 고장 센서일 가능성이 높음. 전체 행의 5.92%(164,039 / 2,770,956)가 여기 해당. 구간 1, 3, 4, 106번은 4개월 내내 **100% stuck** 상태 — `remove_dead_segments`(모든 행이 `status == -101`인 구간만 제거)는 이런 구간을 걸러내지 못함, 정상 status 행이 일부 있긴 하지만 속도 값 자체가 전혀 안 바뀌는 사실상 고장 센서이기 때문. 이 마스크는 구간(segment) 단위가 아니라 run(연속 구간) 단위로 동작해서, 완전히 고장난 구간은 전부 제외되고 일시적으로만 멈췄던 구간은 앞뒤의 정상 데이터를 살릴 수 있음.

**시사점**: 단기(5~30분) 상관관계가 이미 충분히 높아서, "값이 안 바뀐다고 가정하는" naive persistence baseline을 이기기가 쉽지 않음 — ML/DL 모델은 단순 평균이 아니라 이 baseline을 기준으로 검증해야 함. 반면 장기(1일, 30일)로 갈수록 상관관계가 눈에 띄게 떨어져서, 캘린더/cyclical 피처가 persistence 대비 실질적인 가치를 더할 여지가 여기에 있음.
