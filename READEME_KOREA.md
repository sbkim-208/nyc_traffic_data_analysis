# NYC Traffic Data Analysis

## 1. 센서 신뢰도

데이터: NYC DOT Traffic Speeds NBE (Socrata `i4gi-tjb9`), 2024-04-01 ~ 2024-08-01 (4개월), 4,233,169행

센서 신뢰도 = (전체 행 수 − `status == -101` 행 수) / 전체 행 수

### 전체 신뢰도

| 전체 데이터 수 | status=-101 개수 | 신뢰도 |
|---|---|---|
| 4,233,169 | 1,056,062 | 0.7505 (75.05%) |

### Borough별 신뢰도

| borough | total | error_count | error_rate | 신뢰도 |
|---|---|---|---|---|
| Manhattan | 904,947 | 395,470 | 0.4370 | 56.3% |
| Brooklyn | 370,312 | 140,285 | 0.3788 | 62.1% |
| Staten Island | 871,438 | 304,459 | 0.3494 | 65.1% |
| Bronx | 798,345 | 130,361 | 0.1633 | 83.7% |
| Queens | 1,288,127 | 85,487 | 0.0664 | 93.4% |

Queens가 가장 안정적이고 Manhattan이 가장 불안정 — 6.6배 차이. 시각화: `borough_reliability.png`

### Borough별 문제 구간

표본 100건 미만(노이즈성 구간)은 제외하고 borough별 오류율 top 5만 집계.

- Manhattan, Brooklyn, Staten Island, Bronx는 top 구간 대부분이 **error_rate = 1.0** (4개월 내내 정상 응답 0건 — 완전 고장)인 반면, **Queens는 top 5 전부 100% 미만(19~33%)** — 완전 고장 센서가 없고 부분적 불안정만 있음. Queens의 전체 신뢰도가 가장 높은 이유와 일치.
- 완전 고장 구간 예: `VNB` (Verrazzano-Narrows Bridge), `MDE`(Major Deegan Expwy), `WSE`(West Shore Expwy), `SIE`(Staten Island Expwy), `Westside Hwy`, `Lincoln Tunnel` 등

시각화: `borough_problem_roads.png` (borough별 small multiples)

실행: `python sensor_analysis.py`

### 혼잡이 센서를 고장나게 하는가?

가설: GWB 접근로, Lincoln Tunnel, West Side Highway(9A) 같은 유명 혼잡 회랑의 구간일수록 통행량 부하로 센서가 더 많이 고장난다.

부분 고장 상태(일부 유효 데이터가 남아있는) Manhattan 구간 두 곳으로 검증 — Lincoln Tunnel W center tube(오류율 94.11%), 12th Ave S 57th–45th(오류율 93.9%).

**Lincoln Tunnel W center tube (id 329)** — 전체 34,614건 중 valid 2,039건(5.89%):
- valid 구간 속도: 평균 17.19 mph, 중앙값 16.15 mph(범위 1.2~39.8 mph) — 실제로 낮음, 남아있는 데이터만 보면 혼잡이 맞음
- 하지만 **시간대별 오류율은 러시아워(8~10시 72~84%, 17~18시 86~92%)에 가장 낮고**, 새벽·심야(1~5시, 20~23시)는 거의 100% — 혼잡 가설과 정반대 패턴. 혼잡이 센서를 마모시킨다면 오류율이 혼잡 시간에 같이 올라가야 하는데 오히려 떨어짐일

**12th Ave S 57th–45th (id 106)** — valid 평균 속도 26.68 mph(덜 심각), 주간(89~92%) vs 야간(97~99%) 오류율 차이도 미미하고 뚜렷한 러시아워 패턴 없음

**결론: 가설 기각.** "혼잡이 센서를 고장낸다"는 데이터로 뒷받침되지 않음. 대신 확인된 건 **구간별로 유효 데이터가 존재하는 시간대가 다르다**는 것 — Lincoln Tunnel의 경우 유효 데이터가 러시아워에 몰려있는데, 이는 오히려 우리가 예측하려는 시간대(혼잡 시간)와 겹쳐서 유리함. 구간마다 사용 가능한 시간대를 개별 확인해야지, 균일하다고 가정하면 안 됨.

## 2. 전처리 (Data Processing)

`soobin/cleaning/preprocess.py`의 함수들.

1. **`drop_duplicate_ids`** — `link_id`, `transcom_id`, `encoded_poly_line`, `encoded_poly_line_lvls` 드롭 (미사용/중복 컬럼, 위 피처 검토 참고).
2. **`remove_dead_segments`** — 구간 전체(`id` 기준)가 `status == -101`인, 단 한 번도 정상 응답이 없던 완전 고장 구간 제거. 이런 구간은 보간(interpolate)이 불가능함 — 앵커로 삼을 실제 관측치가 시리즈 어디에도 없기 때문. 결과: **16개 구간, 487,811행 제거**.
3. **`fill_missing_speed` / `fill_missing_travel_time`** — 남은 구간에 대해 `status == -101` 행을 NaN 처리 후 구간별로 `interpolate(method="time")`, 보간이 안 닿는 양 끝은 `ffill`/`bfill` 폴백.

| | speed | travel_time |
|---|---|---|
| 보정 전 결측 | 396,112 | 396,077 |
| 보정 후 결측 | 0 | 0 |

최종 행 수: 4,233,169 → 완전 고장 구간 제거 후 **3,745,358**.

**주의**: 보간은 `speed`/`travel_time`만 채우고 `status`는 건드리지 않음. 그래서 1번 섹션의 신뢰도 수치는 항상 원본(raw) 데이터 기준으로 계산해야 하고, 이 전처리 결과로 재계산하면 안 됨 — `remove_dead_segments`로 최악의 구간을 이미 분모에서 뺐기 때문에 신뢰도가 실제로 좋아진 게 아니라 숫자만 높아 보이게 됨.


## 3. 시계열 분석

`data_as_of`를 datetime으로 변환해 정렬된 index로 설정한 뒤, resample(시간별/일별)과 rolling window(7일 이동평균, 14일 이동표준편차, 전주 대비 변화량)로 확인. 실행: `python time_series_analysis.py`

### Borough별 Peak Hour (평일 vs 주말)

혼잡 peak = AM(6~9시)/PM(15~19시) 구간 내에서 평균 속도가 가장 낮은 시각.

| Borough | 구분 | AM peak | PM peak |
|---|---|---|---|
| Bronx | 평일 | 9시 (31.8mph) | **16시** (25.1mph) |
| Brooklyn | 평일 | 8시 (29.6mph) | **16시** (24.5mph) |
| Manhattan | 평일 | 9시 (18.7mph) | **16시** (15.3mph) |
| Queens | 평일 | 8시 (32.8mph) | **16시** (28.2mph) |
| Staten Island | 평일 | 8시 (46.4mph) | **17시** (40.8mph) |

Bronx·Brooklyn·Manhattan·Queens는 평일 PM peak가 전부 16시로 동일하고, Staten Island만 17시로 다름. 속도 절대 수준도 SI가 압도적으로 높음(AM peak 46.4mph vs Manhattan 18.7mph). SI는 지하철이 없어 통근이 Verrazzano 브릿지·페리 중심이라, 지하철 기반인 나머지 4개 borough와 혼잡 타이밍·강도가 구조적으로 다른 것으로 보임. 시각화: `weekday_weekend_comparison.png`


