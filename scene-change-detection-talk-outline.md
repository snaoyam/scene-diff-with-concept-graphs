# ConceptGraph-based Scene Change Detection — 발표 개요

## 제목: ConceptGraph-based Scene Change Detection

## Problem

- Scene Change Detection: 동일한 scene을 서로 다른 시점에 촬영한 두 영상이 주어졌을 때, scene 내 물체의 추가(added) / 제거(removed) / 위치 이동(moved)을 검출하는 문제
- 최근 SceneDiff가 이 task를 위한 첫 multiview 벤치마크(SceneDiff Benchmark)와 training-free 방법(SceneDiff algorithm)을 제안
- 서로 다른 카메라 궤적으로 촬영된 350개 video pair (SD-V 200 / SD-K 150), dense object instance-level annotation
- 평가지표 3종: px/im IoU(픽셀 단위) → obj/im AP(프레임 내 물체 단위) → obj/sc AP(씬 전체에 걸친 물체 단위, 가장 중요한 지표)

## Motivation

- 기존 접근(SceneDiff 포함)은 before/after 프레임을 pixel/region 단위로 직접 비교하는 방식 - geometry reprojection과 appearance feature 차이로 "여기가 달라졌다"를 탐지
- 장점: 결과가 해석 가능 - 어떤 물체가, 왜 변했다고 판단됐는지 그래프 노드 단위로 추적 가능
- 반면 pixel/region 직접 비교는 "무엇이 바뀌었는가"를 물체 단위 개념 없이 판단하므로, 동일 물체의 여러 관측을 하나의 정체성으로 묶어 추론하기 어렵고, 결과를 물체 단위 설명으로 되짚기 어려움

본 연구는 다른 관점을 취함: 각 시점의 scene을 먼저 object-level 3D representation인 ConceptGraph로 표현한 뒤, 두 그래프를 비교해서 변화를 검출

## Approach

ConceptGraph를 활용한 Scene Change Detection의 파이프라인

Before / After 영상 각각에 대해 독립적으로 ConceptGraph 구축

scene vocabulary discovery → detection/segmentation → geometry 기반 object fusion → recognition confidence 계산 → (Before/After) graph comparison → SceneDiff 벤치마크 형식 변환 및 채점

파이프라인은 3단계로 자동화되어 있음: (1) ConceptGraph 생성, (2) ConceptGraph 비교 및 SceneDiff 형식 변환, (3) SceneDiff 공식 채점 스크립트(evaluate_multiview.py)로 채점

## Original ConceptGraph의 한계

Original ConceptGraph는 일반적인 3D scene understanding을 목적으로 만들어졌기 때문에, scene change detection에 그대로 쓰기엔 세 가지 구조적 한계가 있음

- 고정 vocabulary 의존: object vocabulary가 고정된 class 목록(ScanNet200)에 의존 → 목록에 없는 물체는 애초에 인식이 안 되고, 그 물체가 변했을 경우 손실이 그대로 발생
- object fusion의 오류: semantic similarity와 geometry를 함께 쓰는 방식은, 서로 다른 위치의 비슷한 물체를 잘못 합칠 가능성이 있음
- detection이 불확실한 object도 최종 그래프에 그대로 포함되면 Graph Comparison 단계에 영향을 줌

따라서 Scene Change Detection에 적합한 ConceptGraph를 만들기 위해 representation 생성 과정 자체를 수정할 필요가 있다.

## Scene-specific Vocabulary Discovery

- 기존 YOLO는 ScanNet200 고정 200개 class에 의존 → 실제 scene에는 이 목록에 없는 물체가 존재
- 해결: detection 이전 단계로, depth+pose 기반 voxel 역투영과 greedy set-cover로 씬 전체를 커버하는 최소 대표 프레임 집합을 선정 → 이 대표 프레임 전체를 VLM에 질의해 씬에 어떤 물체들이 있는지 먼저 탐색 → SAM으로 프레임을 분할하고 각 조각을 VLM에 개별 질문 → 두 결과를 합쳐 discovered vocabulary 구성
- Before/After 간 vocabulary 목록을 공유하여 같은 물체가 같은 label로 인식되도록 함 → 이후 graph comparison에서 label 불일치로 인한 매칭 실패를 사전에 방지
- 결과: 고정 vocabulary에 제한되지 않고, 각 scene에 실제로 존재하는 물체를 중심으로 ConceptGraph를 구성

## Geometry-only-based Object Fusion

- Original ConceptGraph: semantic similarity가 높은 물체들이 geometry적으로 충분히 구분되지 않으면 잘못 merge될 위험 (동일/유사 물체가 많을수록 악화)
- 따라서 두 detection이 공간적으로 실제 같은 물체에 해당할 때만 병합하도록 기준을 geometry-only로 전면 교체
- semantic similarity에 의존하지 않고, 기존 object의 누적 3D point cloud를 현재 카메라 view로 projection → morphological closing으로 메워 카메라 프레임 상 이 물체가 차지하는 영역 생성 → 새 detection mask와 비교해 max(교집합/footprint, 교집합/mask) ≥ 0.7이고 교집합 영역 point들의 normal vector 방향 편차가 30% 이내인 비율이 0.7 이상일 때만 merge

## Object Confidence와 Visibility를 이용한 신뢰성 있는 Change Detection

- 물체의 일부만 인식되거나, viewpoint 차이로 인식 신뢰도가 낮을 수 있음. 반대로 YOLO의 오분류/부분 인식으로 실제로 없는 물체가 인식될 수도 있음 → 이런 물체를 반대편 그래프와 직접 비교하면 false positive 발생
- 물체가 카메라 시야에 들어온 프레임 대비 실제로 검출된 프레임의 비율로 계산. 인식 프레임 수 1개 이하이거나 confidence 0.5 미만이면 change detection 비교 대상에서 제외 (단, 매칭 자체는 허용 — 다른 물체의 판정에 참고 정보로는 남겨둠)
- Visibility check: 반대편 scene의 실제 camera trajectory로 봤을 때 해당 object가 관측 가능했는지 확인. 관측 자체가 불가능했다면 detection failure와 실제 removal을 구분하지 못하므로 unchanged로 보수적으로 처리

## Before / After Graph Comparison

두 ConceptGraph를 실제로 어떻게 비교하는지 설명

1. 1단계 (기하): before/after object pair 중 point cloud가 실제로 거의 겹치는 경우(trimmed 평균 거리 ≤ 1.5cm) 바로 1:1로 확정. 물체가 아예 안 움직인 명백한 케이스를 먼저 걸러내 이후 단계의 판단 부담을 줄임
2. 2단계 (외형): 1단계에서 안 잡힌 나머지에 대해, CLIP+DINO feature의 평균 cosine similarity(0.5/0.5 가중)로 매칭. 임계값 0.62(한 씬에서 실측한 클래스 내/클래스 간 유사도 갭의 중간값). 1:n, n:1 매칭 허용
3. 판정: 매칭된 쌍 중 중심 거리가 0.3m 이내면 unchanged, 넘으면 moved (여러 매칭이 있으면 하나라도 0.3m 이내면 unchanged)
4. Visibility check + confidence filter를 최종 관문으로 적용해 added/removed/moved 판정은 신뢰도 높은 물체에서만 발생하도록 제한

## Experimental Setup

- Dataset: SceneDiff Benchmark val split — SD-V(varied) 50 pairs / SD-K(kitchen) 55 pairs, 총 105 scene pairs
- Metric: px/im IoU, obj/im AP, obj/sc AP (모두 SceneDiff 공식 evaluate_multiview.py로 채점, IoU threshold 0.5)
- 비교 대상: (1) Original ConceptGraph 기반 파이프라인, (2) 본 연구의 개선된 ConceptGraph 파이프라인, (3) SceneDiff algorithm (training-free baseline)
- **[TODO] Original ConceptGraph baseline과 SceneDiff algorithm의 정확한 수치는 별도 재현 실행 결과로 채워 넣기** — 발표 전 outputs-history의 초기 실행 결과(예: outputs-0~outputs-9) 중 개선 이전 코드에 해당하는 run이 있는지 확인하거나, baseline을 다시 돌려서 확보

## Experimental Results

- 2026-08-20 기준 val split 105 scene pairs 중 64개 완료 (진행 중)
- 완료된 scene 기준 평균 점수

| Split | # scenes | px/im IoU | obj/im AP | obj/sc AP |
|---|---|---|---|---|
| 전체 | 64 | 0.190 | 0.204 | 0.199 |
| SD-V (varied) | 35 | 0.215 | 0.217 | 0.215 |
| SD-K (kitchen) | 29 | 0.161 | 0.188 | 0.181 |

- SD-K(kitchen)가 SD-V(varied)보다 전 지표에서 낮음 → 주방 씬은 그릇/컵/조리도구 등 같은 class의 물체가 여러 개 밀집해 있어 detection·fusion·매칭 난이도가 전반적으로 더 높음
- **[TODO] Original ConceptGraph baseline, SceneDiff algorithm 점수를 같은 표에 나란히 넣어 "개선됐지만 SceneDiff algorithm에는 아직 못 미침"을 수치로 보여주기**
- 나머지 41개 scene pair는 발표 전까지 run.sh 완주해서 105개 전체 기준으로 갱신 필요

## Ablation Study

세 가지 개선 요소(vocabulary discovery / geometry-only fusion / confidence·visibility filter) 각각이 최종 점수에 얼마나 기여하는지 분리해서 보여주는 절이 필요

- **[TODO]** 아래 조합으로 재실행해서 obj/sc AP 비교
  - 전체 개선안 (현재)
  - vocabulary discovery만 제거 (ScanNet200 고정 목록으로 되돌림)
  - geometry-only fusion만 제거 (semantic+geometry 방식으로 되돌림)
  - confidence/visibility filter만 제거 (모든 object를 비교에 사용)
- 목적: "세 가지 중 어떤 것이 가장 크게 기여했는가", "어떤 것이 오히려 recall을 깎았는가"를 분리해서 보여줌 → Error Analysis에서 병목이 representation 단계에 있다는 주장을 뒷받침

## Error Analysis

Dense Scene에서 성능이 저하되는 이유 분석

- 완료된 64개 scene을 물체 수(before 그래프 기준) 기준 3등분(tercile)해서 obj/sc AP 비교

| 물체 수 구간 | # scenes | 평균 obj/sc AP |
|---|---|---|
| 적음 (2–17개) | 21 | 0.219 |
| 중간 (17–31개) | 21 | 0.241 |
| 많음 (31–128개) | 23 | 0.139 |

  - 물체 수와 obj/sc AP의 상관계수 ≈ -0.19로 뚜렷한 선형 관계는 아니지만, 물체 수가 가장 많은 상위 1/3 구간에서 평균 점수가 눈에 띄게 낮음 (0.14 vs 나머지 구간 0.22~0.24)
  - 예: store_11_store_12 (128개 물체) obj/sc AP 0.0 — 가장 물체가 많은 scene 중 하나에서 완전히 실패

- 원인별 분해
  - **detection failure**: scene vocabulary discovery로도 못 찾거나, discovery는 됐지만 detector가 실제 프레임에서 놓친 물체 — 애초에 그래프 노드로 존재하지 않으면 change 판정 자체가 불가능
  - **partial segmentation**: SAM mask가 물체의 일부만 잡거나, 여러 물체를 하나의 mask로 묶음 → footprint 기반 geometry fusion 기준(IoU 0.7)을 충족하지 못해 같은 물체가 여러 개의 별도 노드로 쪼개짐, 혹은 반대로 다른 물체와 뒤섞임
  - **incorrect object fusion**: 물체가 밀집해 있을수록 footprint가 서로 겹치는 경우가 많아져 geometry-only 기준으로도 오병합/과소병합이 발생할 여지가 늘어남
- 세 오류는 독립적이지 않고 누적됨: detection이 불안정한 물체는 confidence가 낮아져 비교 대상에서 제외되고, 이는 recall 손실로 직결 → 물체 수가 많은 씬일수록 이 누적 효과가 커짐
- **결론**: 현재 성능의 주요 bottleneck은 graph comparison 자체(매칭/판정 로직)보다는, 두 시점의 Scene을 얼마나 정확하게 object-level representation으로 만들어내는가(=upstream detection/segmentation/fusion 품질)에 있음

## Failure Cases

구체적인 Failure 사례 — scenegraph_viz / z_frame_grid_viz 등 기존 시각화 결과에서 대표 scene을 골라 before/after 그래프와 실제 GT를 나란히 보여주는 형식 제안

- **[TODO] 아래 후보 중 2~3개를 골라 케이스 스터디로 구체화**
  - `store_11_store_12` (물체 128개, obj/sc AP 0.0) — dense scene에서의 전면적 실패 사례. detection/fusion 오류가 어떻게 누적되는지 보여주기 좋음
  - `gas_station_3_gas_station_4` (물체 4개인데도 obj/sc AP 0.0) — 물체 수가 적어도 실패하는 경우 → dense scene 문제만으로 설명 안 되는 별도 실패 모드가 있음을 보여줌 (예: vocabulary discovery가 애초에 놓친 물체, 혹은 visibility check가 과도하게 보수적으로 unchanged 처리)
  - `hallway_1_hallway_2` / `P09-20240621-093545_0047_...0048` (obj/sc AP 1.0) — 성공 사례도 하나 넣어서 무엇이 잘 맞았을 때와 안 맞았을 때의 차이를 대비시키기

## Discussion: SceneDiff algorithm과의 관점 비교

- SceneDiff algorithm(pixel/region 직접 비교)과 본 연구(object-graph 비교)의 장단점을 정리하는 절
  - 해석가능성: object-graph 방식은 "어느 노드가 왜 바뀌었다고 판단됐는지" 추적 가능 (confidence, visibility, 매칭 threshold까지 역추적 가능) — pixel 방식보다 디버깅과 설명이 쉬움
  - 정확도: 현재는 upstream representation(detection/segmentation/fusion)의 오차가 그대로 누적되어 SceneDiff algorithm보다 낮음. pixel 방식은 물체 단위 추상화를 거치지 않아 이런 누적 오차에 상대적으로 덜 취약
  - 확장성: object-graph는 일단 만들어지면 change detection 외에도 재사용 가능(예: 물체 검색, scene graph 기반 reasoning) — pixel 방식은 change detection 전용
- 이 절은 "우리 방식이 왜 아직 뒤처지는가"에 대한 변명이 아니라, "이 방식이 어떤 상황에서 더 유리한가"를 규명하는 절로 구성

## Conclusion

- Original ConceptGraph를 scene change detection에 맞게 세 방향(scene-specific vocabulary discovery, geometry-only object fusion, confidence/visibility 기반 필터링)으로 개선
- object-level 3D representation을 먼저 만들고 이를 비교하는 방식이, 해석 가능한 change detection을 가능하게 함을 보임
- 다만 현재 SceneDiff algorithm 대비 점수는 아직 낮으며, 원인은 graph comparison 로직이 아니라 upstream representation(특히 dense scene에서의 detection/segmentation/fusion) 품질에 있음을 error analysis로 확인
- **[TODO]** Original ConceptGraph 대비, 그리고 SceneDiff algorithm 대비 최종 수치 비교를 한 문장으로 요약

## Future Work

- Upstream representation 품질 개선이 최우선 과제
  - 더 강한 open-vocabulary detector/segmenter 도입, 또는 프레임 간 tracking을 활용해 partial segmentation을 프레임 단위가 아닌 track 단위로 보정
  - SAM mask 품질 자체를 개선하거나(SAM2 등), mask 품질에 대한 자체 confidence를 fusion 단계에 반영
- Fusion 기준의 고정 threshold(0.7 IoU, 30% normal deviation, 0.62 CLIP+DINO 등)를 씬 밀도에 따라 적응적으로 조정하거나, threshold 대신 학습 기반 판정으로 교체
- Graph comparison에 노드 속성뿐 아니라 edge(물체 간 관계)까지 활용 — 예: "이 물체 주변 물체들이 안 변했다면 이 물체도 안 변했을 가능성이 높다"는 relational consistency를 판정에 반영
- Ablation study를 통해 세 가지 개선 요소 각각의 기여도를 분리하고, 가장 남은 병목에 자원을 집중
- Dense scene 전용 평가 subset을 구성해, scene 밀도별 성능 곡선을 별도로 추적
