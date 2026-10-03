# 실행 안내

이 문서는 **DressCode 형식, 256 x 192 또는 512 x 384**의 실행 절차입니다. GMM 회귀 계층은 선택한 해상도에 맞게 구성됩니다. 아래 예시는 논문 해상도인 512 x 384이며, 논문 결과를 재현했다고 보장하는 절차는 아닙니다. 학습 가중치, 전처리 결과와 정확한 실험 환경이 필요합니다.

256 x 192를 사용할 때는 데이터 root를 해당 전처리 폴더로 바꾸고 `--fine_height 256 --fine_width 192`를 지정합니다. 옵션을 생략하면 256 x 192가 기본값입니다. GMM의 checkpoint는 선택한 해상도로 학습한 파일이어야 합니다. 256 GMM checkpoint를 512 GMM에 직접 불러올 수 없으므로 512 가중치를 확보하거나 512 설정으로 새로 학습해야 합니다.

TOM의 해상도 옵션을 전달하는 경로는 포함되어 있지만, TOM 전체 forward와 GPU 추론은 아직 검증하지 않았습니다. 추가로 제공된 `tom_final.pth`, `tom_final(1).pth`, `tom_final(2).pth`는 모두 현재 전체 TOM과 구조가 달라 아래 TOM 추론 명령에 그대로 사용할 수 없습니다. [가중치 호환성 확인](checkpoint-compatibility.md)을 참고하고 현재 모델과 일치하는 checkpoint를 준비하세요. 아래의 `checkpoints/tom/tom_final.pth`는 호환되는 가중치를 놓을 예시 경로입니다.

## 데이터 계약

pair list는 데이터 root 아래에 두고 각 줄을 `person_filename garment_filename`의 두 열로 구성합니다. 현재 로더는 JPG 인물 파일명을 기준으로 parsing과 JSON 파일명을 만듭니다. 모든 이미지와 마스크의 실제 크기는 실행 옵션과 일치해야 합니다. 로더는 크기가 다른 파일의 경로와 기대 크기를 오류로 안내합니다. 자동 리사이즈는 하지 않으므로 parsing label과 pose 좌표에 맞춰 전처리한 데이터를 사용하세요.

| `<dataroot>/<datamode>/` 아래 경로 | 파일명 | 용도 |
| --- | --- | --- |
| `images/` | 인물 이름 | 인물 RGB |
| `image_parse/` | 인물 이름의 `.jpg`를 `.png`로 치환 | label ID를 가진 parsing |
| `agnostic-v3.2/` | 인물 이름 | 의상을 제거한 인물 표현, RGB |
| `skeletons/` | 인물 stem + `.jpg` | pose 시각화, RGB |
| `keypoints/` | 인물 이름의 `.jpg`를 `.json`으로 치환 | `keypoints` 필드, 점마다 숫자 4개 |
| `cloth_deformation/` | 의상 이름 | GMM 의상 입력 |
| `cloth_mask_deformation/` | 의상 이름 | GMM 의상 이진 마스크 |
| `cloth_gt/` | 의상 이름 | 학습용 정답 의상 마스크, `--gt_mask_dir`로 변경 가능 |
| `paired_warp_cloth-2/` | 기본: 인물 이름 | TOM의 정렬 의상 입력, `--warp_cloth_dir`로 변경 가능 |

keypoints 좌표는 기존 코드가 가정하는 384 x 512 기준입니다. 코드에서 실행 너비와 높이로 축소합니다. 이미 축소한 좌표를 전달하면 다시 축소될 수 있습니다. `image_parse`의 label 15가 의상 영역이며, 컬러 parsing 이미지를 단순 grayscale로 바꿔 넣으면 안 됩니다.

학습용 정답 마스크는 원본 코드의 명명대로 **의상 이름**으로 찾습니다. paired list에서 인물과 의상의 파일명이 다르다면 해당 정답 마스크의 대응을 먼저 확인하세요. 전처리 자료 없이 정답을 임의로 생성하지 않습니다.

## 환경 설정

Python 3.10과 `requirements.txt`를 출발점으로 사용하세요. torch 2.1.2 / torchvision 0.16.2는 제공된 평가 의존성 파일에서 확인한 버전입니다. CUDA와 GPU에 맞는 PyTorch 빌드가 필요합니다. 이 의존성 목록은 원본 학습 환경의 완전한 복원이 아닙니다.

`archive/original/environment.yml`은 Python 3.5 / PyTorch 0.4.1 기반 CP-VTON 환경이고, 현재 연구 코드의 사용 API와 일치하지 않습니다. 현재 프로젝트 설치용으로 그대로 사용하지 마세요. 원본의 평가 의존성 목록도 실제 사용하는 `clean-fid`, `torchmetrics`, `prettytable`이 빠져 있어 별도로 정리했습니다.

## 1. 실행 전 데이터 확인

```bash
python tools/check_assets.py \
  --dataroot data/Dresscode-512 --datamode test \
  --fine_height 512 --fine_width 384 \
  --data_list test_pairs_paired_modified.txt --stage GMM \
  --checkpoint checkpoints/gmm/gmm_final.pth --check-env
```

학습을 점검할 때는 `--datamode train --data_list train_pairs_modified.txt --training`을 사용합니다. 사전 점검은 모델을 실행하지 않습니다.

## 2. GMM 학습 또는 추론

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --name gmm --stage GMM --dataroot data/Dresscode-512 \
  --fine_height 512 --fine_width 384 \
  --datamode train --data_list train_pairs_modified.txt \
  --checkpoint_dir checkpoints --gt_mask_dir cloth_gt --shuffle

CUDA_VISIBLE_DEVICES=0 python test.py \
  --name gmm_test --stage GMM --dataroot data/Dresscode-512 \
  --fine_height 512 --fine_width 384 \
  --datamode test --data_list test_pairs_paired_modified.txt \
  --checkpoint checkpoints/gmm/gmm_final.pth \
  --result_dir results --BASE paired
```

두 명령은 각각 새 학습과 기존 가중치 추론입니다. 가중치가 있으면 추론만 실행할 수 있습니다. 정렬된 이미지는 `results/paired/test/c_result/`에 **인물 이름**으로 저장됩니다. 서로 다른 인물이 같은 의상을 사용하는 unpaired 목록에서도 파일이 덮어써지지 않도록 정리했습니다.

학습 loop의 손실 계산과 스케줄러 적용은 원본대로 남겼습니다. 현재 loop는 생성한 scheduler의 `step()`을 호출하지 않고, 보정 L1 항에도 coarse RGB를 다시 사용하는 부분이 있습니다. 최종 실험 코드와 대조가 필요한 항목이며 임의로 학습 방식을 바꾸지 않았습니다.

## 3. 정렬 이미지 전달과 TOM 추론

GMM 결과를 데이터 split 아래의 별도 폴더로 전달합니다. 아래 예시는 결과 폴더에 연결하는 symlink 방식입니다.

```bash
ln -s "$(pwd)/results/paired/test/c_result" \
  data/Dresscode-512/test/paired_warp_cloth-2

CUDA_VISIBLE_DEVICES=0 python test.py \
  --name tom_test --stage TOM --dataroot data/Dresscode-512 \
  --fine_height 512 --fine_width 384 \
  --datamode test --data_list test_pairs_paired_modified.txt \
  --warp_cloth_dir paired_warp_cloth-2 --warp_cloth_key person \
  --checkpoint checkpoints/tom/tom_final.pth \
  --result_dir results --BASE paired
```

최종 결과는 `results/paired/test/try-on/`에 인물 이름으로 저장됩니다. 대상 폴더가 이미 있다면 다른 새 폴더명을 정하고 `--warp_cloth_dir`에도 같은 이름을 전달하세요. 서로 다른 paired / unpaired 결과는 별도 폴더와 `--BASE` 이름을 사용합니다.

기존에 저장한 정렬 이미지가 의상 이름 기준이라면 `--warp_cloth_key garment`로 읽을 수 있습니다. 단, 동일 의상의 여러 인물 결과가 이미 덮어써졌다면 올바른 unpaired 평가를 위해 GMM 결과를 다시 생성해야 합니다.

TOM 학습은 GMM 추론을 `--datamode train --data_list train_pairs_modified.txt`로 먼저 수행해 training split의 정렬 의상을 준비한 후 다음과 같이 진행합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --name tom --stage TOM --dataroot data/Dresscode-512 \
  --fine_height 512 --fine_width 384 \
  --datamode train --data_list train_pairs_modified.txt \
  --warp_cloth_dir paired_warp_cloth-2 --warp_cloth_key person \
  --gt_mask_dir cloth_gt --checkpoint_dir checkpoints --shuffle
```

## 4. 평가

```bash
python eval.py --gt_folder data/Dresscode-512/test/images \
  --pred_folder results/paired/test/try-on --paired
```

paired 평가에서 GT와 예측은 확장자를 제외한 **인물 파일명 집합이 정확히 일치**해야 합니다. subset만 생성했다면 해당 subset의 GT 폴더를 따로 준비하세요. 파일을 누락한 상태에서 부분 점수를 논문 전체 test 결과처럼 사용하면 안 됩니다.

unpaired 평가는 `--paired`를 빼고 FID와 KID만 계산합니다. KID 출력은 `raw KID * 1000`입니다. GT 폴더와 예측 폴더는 같은 평가 대상과 해상도를 사용하세요. 코드가 GT를 리사이즈하면 그 경로를 별도로 출력합니다.

논문 Table 1과 Table 2, 기존 평가 코드, 정리한 평가 코드의 수치는 구분해 보관해야 합니다. 정리한 평가 코드는 누락된 대응 이미지와 중복 ID를 오류로 처리하고 batch 사이의 metric 상태를 reset하도록 수정했습니다. 논문 수치를 새 코드로 재검증한 것은 아닙니다.
