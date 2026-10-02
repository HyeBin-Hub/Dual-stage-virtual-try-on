# 모델 구조와 코드의 대응

## 실제 진입점

`train.py`와 `test.py`는 `--stage GMM`에서 `GMM(opt)`, `--stage TOM`에서 `TOM_Fusion_GT_U_Transformer.FusionUNet(9, 4, 64)`를 사용합니다. 현재 학습 모델 선택을 파일명에 근거해 추정하지 않고 이 import를 기준으로 정리했습니다.

| 논문 명칭 | 실제 클래스와 파일 | 입력과 출력 |
| --- | --- | --- |
| GGAM / Coarse warping | `GMM_networks.GMM`, `TpsGridGen` | 인물 표현 6채널과 의상 3채널 → TPS grid, theta, coarse RGB |
| Attention U-Net refinement | `GMM_network_Unet.AttentionUNet` | 인물 표현 + coarse RGB, 총 9채널 → RGB 3채널 + 마스크 1채널 |
| GBFM / Trans-Fusion U-Net | `TOM_Fusion_GT_U_Transformer.FusionUNet` | 인물 표현 + 정렬 의상, 총 9채널 → 인물 RGB 3채널 + 마스크 1채널 |
| Group-wise encoder | `TOM_network_GT_U_Net.BotBlock`, `MHSA` | 4 x 4 단위의 그룹 처리와 self-attention |
| Multi-scale fusion | `TOM_Fusion_GT_U_Transformer.FuseModule` | encoder의 네 해상도 특징을 양방향으로 융합 |
| Cross-attention decoder | `TOM_network_U_Net_Transformer.TransformerUp` | 깊은 특징과 skip 특징 사이의 관계를 반영해 복원 |

인물 표현은 `agnostic-v3.2`의 RGB 3채널과 `skeletons`의 RGB 3채널을 연결한 결과입니다. 현재 parsing의 의상 label은 `15`, 머리 영역은 `>20`이며, 이 매핑의 생성 코드는 제공되지 않았습니다.

GMM의 보정 의상은 `coarse_cloth * sigmoid(mask) + tanh(rendered_cloth) * (1 - sigmoid(mask))`입니다. TOM의 최종 합성은 같은 방식으로 정렬 의상과 rendered person을 결합합니다.

## 해상도 설정

GMM은 `--fine_height`와 `--fine_width`에서 특징 맵, 상관관계 채널, 선형 계층 크기를 계산합니다. 특징 추출기는 입력을 16배 축소하고, 회귀 계층은 그 특징 맵을 추가로 4배 축소합니다. 원본의 256 설정과 주석의 512 설정을 같은 계산으로 구성했습니다.

| 입력 높이 x 너비 | 특징 맵 | 상관관계 채널 | 선형 계층 입력 |
| --- | --- | --- | --- |
| 256 x 192 | 16 x 12 | 192 | 64 x 4 x 3 = 768 |
| 512 x 384 | 32 x 24 | 768 | 64 x 8 x 6 = 3072 |

기본값은 256 x 192입니다. 512 실행에는 `--fine_height 512 --fine_width 384`를 지정합니다. 두 해상도 모두 GMM 전체 forward를 CPU 텐서로 확인했습니다. TOM 데이터 로더도 두 크기를 확인했으며, TOM 전체 forward와 GPU 학습은 미확인입니다.

256과 512 GMM은 회귀 계층 가중치의 크기가 다릅니다. 각 해상도로 학습한 가중치를 사용해야 하며, 다른 해상도의 가중치를 읽으면 계층별 크기 차이를 출력합니다. 256 학습 가중치의 단순 리사이즈로 512 모델의 성능을 보장할 수 없습니다. 원본의 256 회귀 계층과 parameter key는 유지했으며, 동일 가중치와 입력에서 출력이 정확히 일치함을 확인했습니다.

## 최종 TOM에 필요한 공통 코드

TOM의 전체 모델 진입점은 `TOM_Fusion_GT_U_Transformer.py`입니다. 아래 파일은 이 모델이 직접 불러오는 공통 계층이므로 함께 유지합니다.

| 파일 | 역할 |
| --- | --- |
| `TOM_network_GT_U_Net.py` | Group-wise self-attention encoder의 `BotBlock` |
| `TOM_network_U_Net_Transformer.py` | `TransformerUp` cross-attention decoder와 출력 계층 |
| `TOM_network_Fusion_u_net_utils.py` | Convolution, 정규화와 fusion 공통 계층 |

현재 학습과 추론은 전체 TOM을 사용합니다. 비교 실험 후보와 사용하지 않는 구현은 `archive/original/`에 원본 그대로 보존했습니다. 논문 Table 2의 각 제거 실험과 최종 코드의 정확한 대응에는 당시 설정과 로그가 필요합니다.
