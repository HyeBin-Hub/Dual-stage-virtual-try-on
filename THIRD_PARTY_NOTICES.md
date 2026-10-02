# 참고 구현과 출처

원본 코드의 `LICENSE`는 **Copyright (c) 2018 Bochao Wang / MIT License**입니다. 이 고지를 삭제하거나 이혜빈 단독 저작으로 변경하지 않았습니다.

| 자료 또는 구성 요소 | 확인한 출처 | 비고 |
| --- | --- | --- |
| CP-VTON 기반 GMM, 학습과 데이터 처리 기반 | [sergeywong/cp-vton](https://github.com/sergeywong/cp-vton) | 제공 README와 LICENSE가 해당 기반 구현을 가리킴 |
| Group-wise encoder 아이디어 | [GT U-Net 논문](https://arxiv.org/abs/2109.14813), [저자 코드](https://github.com/Kent0n-Li/GT-U-Net) | 첨부 논문의 참고 문헌. 현재 코드와 외부 코드의 줄 단위 출처를 모두 검증한 것은 아님 |
| Multi-scale fusion 아이디어 | [FusionU-Net 논문](https://proceedings.mlr.press/v222/li24a.html), [저자 코드](https://github.com/Zongyi-Lee/FusionU-Net) | 첨부 논문의 참고 문헌 |
| 일부 U-Net utility | [UCTransNet](https://github.com/McGregorWwww/UCTransNet) | `TOM_network_Fusion_u_net_utils.py`에 출처 주석이 있음 |
| U-Net padding 처리 | [Pytorch-UNet](https://github.com/xiaopeng-liao/Pytorch-UNet) | 여러 원본 파일의 commit 출처 주석을 보존 |
| U-Net Transformer / cross-attention | 첨부 논문의 참고 문헌 [32] | 외부 원본 코드의 정확한 출처와 라이선스 추가 확인 필요 |
| README의 모델 구조 및 비교 이미지 | 이 연구의 첨부 논문 Figure 2 / Figure 5 | 논문에서 추출. 새로 실행한 결과가 아님 |

이 표는 확인한 출처 기록입니다. 저장소 전체의 모든 구성 요소에 동일한 라이선스가 적용된다는 새 선언은 아닙니다. 원본 고지와 각 upstream의 조건을 함께 확인해야 합니다. 미확인 provenance를 임의로 작성하지 않았습니다.
