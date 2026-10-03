# 검증 기록과 수정 범위

검토일: 2026-10-02. 대상: 드라이브의 `viton code` 32개 원본 파일과 제공 논문.

## 확인한 범위

| 점검 | 결과 |
| --- | --- |
| 원본 보존 | 32개 파일의 SHA-256을 `archive/original/`과 비교 |
| Python 문법 | 현재 실행 코드와 비교 모델, 도구, 원본 archive 전체 검사 |
| 로컬 import | 핵심 실행 경로의 파일 대응 검사 |
| 데이터 사전 점검 | 7개 테스트로 마스크, 정렬 의상 이름, 가중치 누락, 지원 해상도, JSON 및 pair list 검사 |
| 실제 데이터 로더 | 2개 테스트로 GMM / TOM의 256 x 192와 512 x 384 이미지와 마스크, pose 좌표, grid 크기를 확인하고 크기가 다른 파일의 오류 처리 검사 |
| GMM 전체 forward | 두 해상도에서 특징 추출 → 상관관계 → 회귀 → TPS → coarse warping → Attention U-Net 실행. 출력 크기와 유한값 확인 |
| 회귀 계층과 TPS 역전파 | 두 해상도에서 입력 상관관계와 회귀 가중치에 유한한 gradient가 생성됨을 확인 |
| TPS | 두 해상도에서 offset 0의 grid가 항등 변환과 일치. buffer의 dtype 이동 확인 |
| checkpoint | 두 해상도의 회귀 가중치 저장과 로딩 후 동일 값 확인. 해상도가 다른 가중치는 변경 전에 거부 |
| 256 원본 호환성 | archive의 원본 회귀 클래스를 직접 구성. parameter key와 크기가 같고 동일 가중치 / 입력에서 출력이 정확히 일치 |
| 입력 오류 | 모델 입력의 채널 또는 크기가 선택한 해상도와 다르면 특징 추출 전에 오류 처리 |
| 문서 | 상대 링크, 이미지 경로, 논문 표 수치와 실행 옵션 대응 검사 |
| 제공 TOM checkpoint | 세 파일의 key, tensor 크기와 유한값 검사. 모두 현재 전체 TOM과 불일치. 자세한 결과는 `checkpoint-compatibility.md` 참고 |
| TOM 전체 forward / GPU 학습과 추론 | 미수행. CUDA GPU가 없고 현재 전체 TOM과 호환되는 최종 가중치도 확보하지 못함 |
| 논문 성능 재현 | 미수행. 최종 가중치, 실험 설정, split과 원시 로그가 필요 |

총 15개 테스트를 통과했습니다. 실행 환경은 Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu입니다. 저장소의 Python 3.10 / torch 2.1.2 의존성 후보와 별도의 CPU 검증 환경이며, 원본 학습 환경을 복원한 것은 아닙니다.

모델 테스트는 무작위 입력과 초기 가중치를 사용합니다. 데이터 로더 테스트의 이미지는 합성 fixture입니다. 실제 가상 착용 결과의 품질이나 논문 수치를 검증한 결과가 아닙니다. `grid_sample`의 `align_corners`는 원본처럼 명시하지 않았고, 검증 환경의 기본값인 `False`로 실행됐습니다.

## 해상도 지원 범위

크기 표기는 높이 x 너비입니다. 실행 옵션과 실제 전처리된 이미지, 마스크의 크기가 같아야 합니다.

| 설정 | GMM correlation 채널 | GMM linear 입력 수 | 확인한 실행 |
| --- | ---: | ---: | --- |
| 256 x 192 | 192 | 768 | GMM 전체 forward, 회귀와 TPS 역전파, 데이터 로더 |
| 512 x 384 | 768 | 3072 | GMM 전체 forward, 회귀와 TPS 역전파, 데이터 로더 |

correlation 채널은 `(H // 16) * (W // 16)`, linear 입력 수는 `64 * (H // 64) * (W // 64)`로 계산합니다. 256 고정 크기를 512 입력에 적용하던 GMM 오류를 수정했습니다. 두 크기만 지원하며 임의 해상도나 전체 TOM의 두 해상도 실행까지 확인한 것은 아닙니다.

기본 실행 옵션은 256 x 192입니다. 512 x 384에는 `--fine_height 512 --fine_width 384`를 지정하고 같은 해상도의 데이터와 GMM checkpoint를 사용해야 합니다. 256 GMM 가중치의 회귀 계층은 512 GMM과 크기가 달라 직접 불러올 수 없습니다.

제공 TOM 가중치의 encoder, fusion과 attention head 차이는 위 GMM 해상도 문제와 구분해야 합니다. 입력 크기 옵션만 바꿔도 현재 전체 TOM과 호환되지는 않습니다.

## 수정 범위

| 수정 | 이유와 영향 |
| --- | --- |
| GMM 회귀 계층의 채널과 선형 계층 크기를 해상도에서 계산 | 512 입력에 256 고정값을 적용하던 오류 해결. 256과 512 설정 지원 |
| 실행 진입점과 데이터 사전 점검의 해상도 검사 통일 | 기존 256 전용 차단을 제거하고 두 해상도를 같은 계약으로 검사 |
| 모델 입력과 데이터 파일의 크기 검사 | 이미지, 마스크, 실행 옵션이 섞였을 때 원인과 파일을 안내 |
| 흑백 shape / pose에 1채널 정규화 적용 | RGB 정규화의 broadcast 오류 해결. 원본과 같은 [-1, 1] 범위 유지 |
| TPS tensor를 비영구 buffer로 등록 | 모델의 device / dtype 이동을 따르며 기존 checkpoint key 유지 |
| checkpoint 오류 안내 | GMM은 해상도와 grid_size, TOM은 모델 구조와 attention head를 확인하도록 안내 |
| checkpoint 크기 검사와 device 유지 | 해상도가 다른 회귀 가중치를 설명과 함께 거부. 저장 / 로딩이 live model을 다른 device로 옮기지 않도록 수정 |
| 데이터 로더의 GMM `cloth_mask` 반환 복구 | 기존 추론과 학습이 접근하는 키가 주석 처리되어 있었음 |
| 학습용 `gt_cltoh_warp_mask` 로딩과 반환 복구 | 학습 loop에서 사용하는 key가 빠져 있었음. 없으면 명확히 실패 |
| agnostic / skeleton / TOM cloth의 RGB 변환 | 인물 표현 6채널, 의상 3채널의 입력 계약을 명시 |
| `grid.png`의 모듈 기준 경로와 크기 처리 | 실행 위치와 출력 크기의 불일치 방지 |
| TOM 정렬 의상 폴더와 이름 기준을 옵션으로 제공 | 기존 데이터와 GMM 결과를 명시적으로 연결 |
| GMM 추론 출력을 인물 이름으로 저장 | 같은 의상을 여러 인물에게 적용할 때 덮어쓰기 방지 |
| 가중치 경로와 CUDA 오류 처리 | 추론 가중치 누락과 GPU 환경 문제를 안내 |
| shuffle이 꺼졌을 때 순서 고정 | 원본 loader는 옵션이 꺼져도 shuffle이 활성화되어 있었음 |
| 평가 utility와 paired 대응 검사 정리 | 불필요한 diffusion 의존성 제거. 누락된 대응, 중복 ID와 batch metric 누적 방지 |
| 전체 TOM 중심으로 구성하고 원본 archive 보존 | 실행 경로는 전체 모델. 비교 후보는 원본 archive에 보존 |

학습 가능한 parameter 이름, 256 회귀 계층의 크기, forward의 detach, 학습 손실의 수식과 가중치는 유지했습니다. 512에서는 회귀 계층의 입력 가중치 크기가 해상도에 맞게 바뀝니다. 학습 loss와 scheduler 관련 미확인 항목은 `reproduction.md`에 기록했습니다. 새로 학습하거나 논문 성능을 개선한 버전은 아닙니다.

## 검증 명령

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

모델 테스트에는 PyTorch가 필요하고 데이터 로더 테스트에는 torchvision과 Pillow도 필요합니다. 의존성이 없으면 해당 테스트를 skip하므로 전체 테스트 수뿐 아니라 skip 여부를 확인하세요. 다음 검증은 실제 데이터와 최종 가중치로 GPU에서 한 batch의 forward와 추론을 수행하는 것입니다.
