# Whisper E2E Latency 실험 결과

실험 원시 데이터는 `result/raw/`에 저장됩니다.

- `results.csv`: 엔진·RQ·표본·청크 조건별 측정 결과. 측정이 끝날 때마다 한 행씩 기록합니다.
- `samples.csv`: 고정 시드로 선택한 원본 talk과 구간, STM 원문 및 Whisper 토큰 수입니다.
- `results.csv`에는 청크별 모델 전사도 포함해 측정값과 출력을 함께 보존합니다.
- `runs/<config_id>.json`: 데이터 경로, 모델·디코딩 설정, talk 길이 등 실행 메타데이터입니다.
- `result/readme/summary_<config_id>.csv`: 완료된 측정의 평균 및 중앙 E2EL 요약입니다.
- `result/readme/figures/`: RQ1, RQ2, RQ3 그래프입니다. 실험 중 주기적으로 갱신됩니다.

## 실행

TEDLIUM Release 3 legacy test의 `sph/`와 `stm/` 디렉터리를 포함하는 경로를 지정합니다.

```bash
uv sync
uv run python script/run_benchmark.py --data-dir /data/TEDLIUM_release-3/legacy/test
```

또는 실행 스크립트를 사용할 수 있습니다:

```bash
script/run_benchmark.sh --data-dir /data/TEDLIUM_release-3/legacy/test
```

전체 실험은 `faster-whisper`와 `openai-whisper`를 순서대로 실행합니다. GPU 서버에서는 기본 CUDA FP16을 사용합니다. CPU에서 실행할 때는 `--device cpu`를 지정하며, 해당 결과는 GPU FP16 조건과 직접 비교하지 않습니다.

실행 중에는 엔진별·RQ별 완료 건수와 비율을 표본마다 출력합니다. 기본 전체 실행에서 RQ1은 엔진당 100건, RQ2는 엔진당 300건(50개 입력 × 2개 길이 × 3개 청크 조건)입니다. 재실행할 때는 기존 성공 건수를 먼저 표시합니다.

짧은 실행 및 일부 조건 실행 예시:

```bash
uv run python script/run_benchmark.py --data-dir /data/TEDLIUM_release-3/legacy/test --limit 1
uv run python script/run_benchmark.py --data-dir /data/TEDLIUM_release-3/legacy/test --engine faster-whisper --rq rq2
```

실행이 중단되면 같은 인자와 결과 경로로 다시 실행합니다. 정상 완료된 표본은 건너뛰고 미완료 표본부터 진행합니다.

## 측정 정의

모델 로드·예열은 E2EL에 포함하지 않습니다. 각 측정은 SPH 오디오 구간 읽기, 모노/16 kHz 변환, 청크 분할, Whisper 추론, 청크별 전사 결합 시간을 포함합니다. 각 표본은 기본 1회 측정하며 RTF는 E2EL을 오디오 길이로 나눈 값입니다.

RQ1은 5~600초 길이를 1분 bin별 10개씩 구성합니다. RQ2는 60초 및 600초 원본 구간을 각각 50개 만들고, 동일 구간을 5·15·30초의 독립 추론 청크로 처리합니다. TEDLIUM test talk이 제한되어 있어 표본 구간은 서로 겹칠 수 있으며, 원본 talk ID와 시작·끝 시각을 함께 보존합니다. RQ3의 token density는 STM 발화 구간을 Whisper tokenizer로 토큰화한 뒤 오디오 길이로 나눕니다.

## 그래프

- RQ1: 오디오 길이와 E2EL 산점도
- RQ2: 청크 크기별 평균 E2EL 막대 그래프
- RQ3: STM 기반 발화 밀도와 E2EL 산점도

파일명은 `figures/<rq>_<config_id>.png` 형식입니다.
