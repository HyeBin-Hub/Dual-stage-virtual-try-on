# 추가로 필요한 자료

검토 대상은 드라이브의 `viton code` 32개 파일과 논문입니다. 저장소 공개에는 아래 자료가 모두 필요한 것은 아닙니다. 다음 목록은 실제 추론, 재학습과 논문 결과 재현에 필요한 자료입니다. 아래에서 '미포함'은 이번 코드 묶음에 없다는 뜻입니다. 별도 드라이브 폴더가 있더라도 파일 내용을 확인하지 못한 항목은 확보한 것으로 표시하지 않았습니다.

| 우선순위 | 필요한 파일 또는 자료 | 확인 상태 | 필요한 이유 |
| --- | --- | --- | --- |
| 1 | GGAM / GMM 최종 `.pth`와 대응 설정 | 코드 묶음에 미포함. 드라이브에 256 / 512 checkpoint 폴더는 있으나 연결을 통해 실제 가중치 파일을 확인하지 못함 | 정렬 추론과 기존 결과 검증 |
| 1 | GBFM / TOM 최종 `.pth`와 대응 설정 | 코드 묶음에 미포함. 256 / 512-2 checkpoint 폴더만 확인 | 합성 추론, 모델 조합 확인 |
| 1 | 논문 실험 당시의 512 x 384 설정과 실행 명령 | 현재 코드는 두 해상도를 지원하도록 수정하고 텐서 실행을 확인. 당시의 최종 실험 설정은 미포함 | 정확한 가중치, grid_size, 손실과 학습 조건의 대응 확인 |
| 1 | 수정한 pair list 세 파일 | 코드 묶음에 미포함. 별도 Dresscode-512 root에서 `train_pairs_modified.txt`, `test_pairs_paired_modified.txt`, `test_pairs_unpaired_modified.txt`는 발견 | 실제 학습과 paired / unpaired 대상 확정 |
| 1 | 학습용 `cloth_gt` 정답 마스크와 생성 방식 | 코드 묶음에 미포함. 별도 Dresscode-512/train에서 폴더 존재 확인 | 학습 loop의 mask supervision, 파일명 대응 확인 |
| 2 | `agnostic-v3.2`, `image_parse`, `skeletons`, `keypoints`, `cloth_deformation`, `cloth_mask_deformation`, `images` | 별도 Dresscode-512/train에서 폴더 존재 확인. 개별 데이터와 완전성 미검증 | 현재 로더의 입력 계약 충족 |
| 2 | VITON-HD 최종 데이터 로더와 label / pose 매핑 | 현재 활성 로더는 DressCode 형식. VITON 관련 과거 코드는 주석으로 남음 | 두 번째 벤치마크의 정확한 재현 |
| 2 | 원본 학습 환경의 버전 기록과 실행 명령 | 이전 CP-VTON 환경 파일, 별도 평가 의존성만 포함 | 학습 당시 PyTorch / CUDA와 실험 조건 복원 |
| 2 | ablation별 최종 코드, 설정, 로그, 가중치 | 원본 후보와 변형은 archive에 보존. Table 2 전체 매핑 미확인 | 실제 제거 대상과 논문 표의 대응 확인 |
| 3 | 최종 비교 이미지, 평가 원시 로그, test subset 정보 | 논문 이미지와 표는 확인. 원시 실험 결과 묶음 미포함 | 결과 검증과 추가 시각적 사례 구성 |

핵심 모델이 import하는 로컬 Python 파일은 모두 포함되어 있습니다. `grid.png`, 원본 `LICENSE`, 시각화 코드도 포함되어 있습니다. 따라서 현재 막히는 지점은 모델 Python 파일 전체의 부재보다, **최종 실험 버전과 가중치, 전처리 결과의 대응**입니다.

원본의 비활성 `Transfer_block.py`는 `vgg19-dcbb9e9d.pth`를 로컬 파일로 읽지만 해당 파일은 제공되지 않았습니다. 현재 GMM / TOM 진입점에서는 사용하지 않습니다. 현재 학습의 VGG19 perceptual loss는 torchvision pretrained weight를 사용하므로 최초 실행 시 다운로드 또는 사전 cache가 필요합니다.

논문의 데이터 개수 설명에는 합계가 맞지 않는 부분이 있고, Table 1과 Table 2의 전체 모델 FID / KID도 다릅니다. 이 저장소는 숫자를 임의로 보정하지 않았습니다. 실제 split 목록과 원시 평가 로그로 확정해야 합니다.

데이터와 가중치는 코드 저장소에 넣지 않습니다. 자료를 추가로 확보하면 대응되는 모델 버전과 해상도, 학습 조건을 먼저 확인한 후 별도 배포 여부를 정할 수 있습니다.
