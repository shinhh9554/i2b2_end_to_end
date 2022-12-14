# End-to-End BioMedical Field Entity and Relation Extraction using Various Pre-trained Language Models

사전학습 언어모델을 활용한 End-to-End 기반의 임상분야 개체 및 관계 추출 모델

## 1. 개요

본 연구는 개체명 인식과 관계 추출을 동시에 추출하기 위해 종단적 추출 모델 중 멀티 헤드 레이블링 방식을 사용하였다. 
종단적 추출 모델은 개체명 인식과 관계 추출 과정을 한번에 수행하여 파이프라인 형식의 모델에서 나타나는 오류곱 현상을 해결할 수 있는 모델이다.
멀티 헤드 레이블링 방식은 개체명 인식과 관계 추출 정보를 하나의 레이블에 포함시키는 방식으로, 연속적 레이블링 문제로 종단적 추출 문제에 접근하는 것이다.
이는 개체명 인식 모델과 유사한 방식으로 쉽게 실험이 가능하지만 예측해야 하는 라벨의 수가 늘어나서 모델의 성능이 하락할 수 있으며, 부가적인 후처리 모듈이 필요하다는 단점이 존재한다.
하지만 연속적 레이블링 문제에 효율적인 모델을 구성하여 종단적 추출 과제에 접근할 수 있기 때문에 구현에 들어가는 소모 비용이 적다는 단점이 존재한다.


## 2. 모델 구조

멀티 헤드 레이블링 방식의 종단적 추출 모델을 구성하기 위해 Bert 사전학습 언어모델에 Bidirectional GRU와 CRF를 추가하였다.
모델의 구성도는 아래 그림과 같다.

<img width="716" alt="시스템 구조도" src="https://user-images.githubusercontent.com/57481142/207298375-1697e1e4-cdb0-4af6-a6b4-79ff18c099bc.png">

## 3. 학습 및 실험 데이터

본 연구는 종단적 추출 모델을 학습하기 위해 I2B2 2010 데이터를 활용하였다. 
I2B2(Informatics for Intergrating Biology & the Bedside) 2010 데이터는 보건 및 의료학 분야에 대한 진료 정보 및 임상 자료를 대상으로 구축되어 있으며,
질병, 검사, 처치 정보 및 연관성 추출을 목표로 한다. 데이터는 총 326개 보고서로 이루어져 있으며, 3개의 개체와 개체들 간의 관계를 나타내는 8개의 관계명으로 구성된다.
그에 따른 데이터 통계는 아래 표와 같다.

* 데이터 별 보고서, 문장, 토큰 통계
<img width="1222" alt="스크린샷 2022-12-13 오후 8 36 50" src="https://user-images.githubusercontent.com/57481142/207307937-3056db2a-e4aa-4707-b473-fb3e84f9223f.png">


* 데이터 별 Entity 통계 
<img width="1214" alt="스크린샷 2022-12-14 오후 2 53 48" src="https://user-images.githubusercontent.com/57481142/207517472-c4a9c69e-349d-43e6-b7c5-9a23295df546.png">


* 데이터 별 Relation 통계
<img width="1221" alt="스크린샷 2022-12-14 오후 2 55 01" src="https://user-images.githubusercontent.com/57481142/207517642-252ede2f-0fe2-4c36-80c8-5e8a87b7c14d.png">


## 4. 코드 실행 방법
### 4-1. 데이터 전처리 방법

👉 i2b2_end2end 디렉토리로 이동
```
cd i2b2_end2end
```

👉 toJoin.py 파일을 실행하여 멀티 헤드 레이블링 데이터 생성
```
python toJoint.py \
  --ner_file_path ner/train.txt \
  --re_file_path re/train.txt \
  --output_file_path data/train.txt \
```

### 4-2. 모델 학습 방법

👉 특정 GPU를 활용하여 모델 학습을 수행
```
CUDA_VISIBLE_DEVICES=0,2 python main.py \
  --model_type bigru-crf \
  --model_name_or_path scibert \
  --max_seq_length 128 \
  --output_dir output \
  --num_train_epochs 5.0 \
```

👉 GPU 전체를 활용하여 모델 학습을 수행
```
python main.py \
  --model_type bigru-crf \
  --model_name_or_path scibert \
  --max_seq_length 128 \
  --output_dir output \
  --num_train_epochs 5.0 \
```

## 5. 실험 결과

* End to End 실험 결과

|                          |acc   |f1    |recall|precision|
|--------------------------|------|------|------|---------|
| bert-base                |0.8357|0.5696|0.613 |0.532    |
| bert-crf                 |0.8363|0.5632|0.612 |0.5216   |
| bert-bilstm              |0.6137|0.0315|0.0216|0.0586   |
| bert-bilstm-crf          |0.6369|0.0778|0.0647|0.0976   |
| bert-bigru               |0.7776|0.5372|0.5399|0.5345   |
| bert-bigru-crf           |0.7806|0.5145|0.5219|0.5072   |
| scibert-base             |0.869 |0.6377|0.6596|0.6173   |
| scibert-crf              |0.8797|0.6589|0.6867|0.6334   |
| scibert-bilstm           |0.6452|0.0041|0.0022|0.0357   |
| scibert-bilstm-crf       |0.6557|0.0203|0.0308|0.0151   |
| scibert-bigru            |0.8099|0.5939|0.5766|0.6123   |
| scibert-bigru-crf        |0.8098|0.5628|0.5639|0.5616   |
| biobert-base             |0.8667|0.6357|0.6682|0.6061   |
| biobert-crf              |0.8589|0.6115|0.6478|0.5791   |
| biobert-bilstm           |0.6252|0.04  |0.027 |0.0776   |
| biobert-bilstm-crf       |0.6446|0.0709|0.0557|0.0976   |
| biobert-bigru            |0.7886|0.6011|0.5911|0.6115   |
| biobert-bigru-crf        |0.7973|0.5783|0.5705|0.5864   |
| clinical_bert            |0.8672|0.6281|0.6718|0.5897   |
| clinical_bert-crf        |0.8569|0.6239|0.6593|0.5921   |
| clinical_bert-bilstm     |0.6092|0.0072|0.004 |0.0396   |
| clinical_bert-bilstm-crf |0.638 |0.056 |0.042 |0.0842   |
| clinical_bert-bigru      |0.8015|0.5983|0.5917|0.605    |
| clinical_bert-bigru-crf  |0.8045|0.5764|0.5806|0.5722   |


## Reference
* 김선우, 유석종, 이민호, and 최성필. 2017. 생의학 분야 학술 문헌에서의 이벤트 추출을 위한 심층 학습 모델 구조 비교 분석 연구. 한국문헌정보학회지, 51(4), pp.77-97, https://doi.org/10.4275/KSLIS.2017.51.4.077
* 김선우. "개체명 인식 및 관계 추출 통합 모델 연구." 국내석사학위논문 경기대학교 대학원, 2019. 경기도
* 이명훈, 신현호, 전홍우, 이재민, 하태현 and 최성필. 2021, "사전 학습된 신경망 언어모델 기반 다중 임베딩 조합을 통한 소재 및 화학분야 개체명 인식 성능 비교 연구", 정보과학회논문지, vol.48, no.6 pp.696-706. Available from: doi:https://doi.org/10.5626/JOK.2021.48.6.696


