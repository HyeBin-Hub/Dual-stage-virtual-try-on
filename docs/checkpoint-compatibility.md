# TOM checkpoint 호환성 확인

제공된 세 TOM 파일을 PyTorch 2.5.1+cpu에서 `weights_only=True`, `map_location='cpu'`로 읽고 현재 실행 경로의 `TOM_Fusion_GT_U_Transformer.FusionUNet(9, 4, 64)`와 비교했습니다. 현재 모델의 state_dict는 573개 항목입니다.

## 결론

**세 파일 모두 현재 전체 TOM과 구조가 달라 직접 불러올 수 없습니다.** 해상도 옵션만 바꿔서 해결되는 차이가 아닙니다. 현재 모델과 일치하는 checkpoint 또는 각 가중치를 학습한 TOM 구현과 공통 코드, 실험 설정이 필요합니다.

| 파일 | state_dict 항목 | 현재 모델에 필요한 key 누락 | 모델이 사용하지 않는 key | 공통 key의 크기 불일치 | 유한값 검사 |
| --- | ---: | ---: | ---: | ---: | --- |
| `tom_final.pth` | 498 | 124 | 49 | 5 | NaN / Inf 없음 |
| `tom_final(1).pth` | 357 | 220 | 4 | 197 | NaN / Inf 없음 |
| `tom_final(2).pth` | 390 | 251 | 68 | 192 | BatchNorm 통계 256개가 Inf |

위 key 수에는 parameter와 buffer가 모두 포함됩니다. `strict=False`는 크기가 다른 tensor를 호환되게 만들지 않으며, 누락된 계층을 학습된 상태로 복구하지도 않습니다.

## 파일별 차이

### tom_final.pth

| 항목 | 제공된 파일 | 현재 TOM |
| --- | --- | --- |
| 초기 encoder | `inc.conv`, `down1.nConvs` 등 일반 convolution 계층 | `inc.mhsa`, `down1.mhsa` 등 group-wise attention 계층 |
| 마지막 encoder attention head | 2 | 4 |
| `down4.mhsa.q_proj.w` 크기 | `[256, 2, 128]` | `[256, 4, 64]` |
| `down4.mhsa.self_attention.rel_emb_w` 크기 | `[7, 128]` | `[7, 64]` |

fusion과 cross-attention decoder의 key와 크기는 현재 모델과 일치합니다. 일반 convolution 초기 encoder와 마지막 단계의 2-head attention을 조합한 진단용 parameter 구성에는 498개 항목과 크기가 모두 일치했고 strict 로딩도 통과했습니다. 이는 메모리에서 수행한 parameter 구조 비교이며, 학습 당시의 forward 구현을 확보하거나 복원한 결과는 아닙니다. 저장소의 최종 모델도 이 구성으로 변경하지 않았습니다.

### tom_final(1).pth

- encoder의 group-wise attention은 2-head 구성이며 현재 모델은 4-head입니다.
- 현재 모델에 있는 `fuse.*` 가중치가 없습니다.
- decoder는 `up1`이 가장 깊은 단계이고 `up4`가 가장 얕은 단계인 이름 구성을 사용합니다. 현재 전체 TOM은 반대 순서의 이름을 사용합니다.
- 현재 모델에 없는 최상위 `MHSA.*` 항목 4개가 있습니다.

fusion을 포함하지 않는 다른 TOM 구성의 특징이 보이지만, 파일명과 parameter만으로 논문 Table 2의 특정 제거 실험에 대응한다고 확정할 수 없습니다.

### tom_final(2).pth

- `vgg_enc.*`, `middle_1.*`, `middle_2.*` 계층이 있고 현재 전체 TOM의 `down4.*`, `fuse.*`는 없습니다.
- 해당 key는 texture transfer를 포함한 다른 모델 구성을 시사합니다. 현재 전체 TOM과의 직접 호환성은 없습니다.
- `up2.MHCA.Yconv2.3.running_var`의 256개 요소가 모두 양의 무한값(`Inf`)입니다. NaN은 발견되지 않았습니다.

이 파일은 구조 차이에 더해 BatchNorm 통계에도 문제가 있습니다. 통계를 임의로 0 또는 1로 바꾸지 않았으며, 정상적인 최종 추론용 가중치로 검증된 상태가 아닙니다.

## 해상도 수정과의 관계

GMM은 입력의 높이와 너비에서 회귀 계층의 크기를 계산하도록 수정했으며, 256 x 192와 512 x 384의 전체 forward를 CPU에서 확인했습니다. 이 수정은 제공된 TOM 가중치의 encoder, fusion, attention head 차이를 해결하지 않습니다.

TOM 가중치의 파일명이나 state_dict만으로 학습 해상도, 최종 실험 버전 또는 성능을 확정할 수 없습니다. 실제 데이터의 TOM 전체 forward, GPU 추론과 논문 성능 재현도 수행하지 않았습니다.

공통 checkpoint 로더의 크기 오류 안내는 GMM과 다른 모델을 구분하도록 수정했습니다. GMM은 해상도와 grid_size를 안내하고, TOM은 모델 구조와 attention head 설정 확인을 안내합니다. 기존 GMM 저장과 로딩 테스트 및 첫 TOM 파일의 실제 거부 동작을 확인했습니다. 가중치를 무시하거나 일부만 불러오는 방식으로 최종 모델을 변경하지 않았습니다.

## 파일 식별

가중치 자체는 Git에 포함하지 않았습니다.

| 파일 | 크기 (bytes) | SHA-256 |
| --- | ---: | --- |
| `tom_final.pth` | 184,480,525 | `aafaa7eb9493a2c388e0ac57ffd9d8dc100415001b1f31f95d25888e660a0f38` |
| `tom_final(1).pth` | 158,195,621 | `e3b9ac06845c9175b9ce5c4952e6070234fa1d38239692f08c8d76b1cb68fb66` |
| `tom_final(2).pth` | 271,610,143 | `55a56bf88b8d6c88073840d36f405bdedd84824918999381ab8bcd704f09f9bc` |
