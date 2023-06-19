<div><img src="https://capsule-render.vercel.app/api?type=waving&animation=fadeIn&color=auto&height=300&section=header&text=Extension&fontSize=90" /></div>

## 프로젝트 소개
<!--Wirte one paragraph of project description -->  
### 혐오발언 인지 언어모델 개선 프로젝트

## 👋 팀원 소개
|이름|학번|학과|GitHub Page|
|------|---|---|---|
|이상돈|2019112491|산업시스템공학과|<https://github.com/leeideal>|
|김건형|2018112016|컴퓨터공학과|<https://github.com/devCharlieP>|
|이종혁|2019112035|컴퓨터공학과|<https://github.com/purple8cloud>|
|황유경|2020111403|경영학과|<https://github.com/yookyung0825>|
|민준영|2019113290|컴퓨터공학과|<https://github.com/Junyoung190198>|


## 빠른 링크

  - [개요](#개요)
  - [설명](#설명)
    - [Simecse](#Simecse)
    - [BackGround Method](#BackGround-Method)
  - [요구사항](#요구사항)
  - [환경 및 언어](#환경-및-언어)
  - [활용 데이터 셋](#활용-데이터-셋)
  - [참고 프로젝트](#참고-프로젝트)
  - [모델 학습 시작하기](#모델-학습-시작하기)
  - [모델 성능](#모델-성능)
  - [차후 계획](#차후-계획)

## 개요
- 학습 데이터 셋 추가 및 데이터 라벨링 개선
- 지도학습 → 자기지도학습(Contrastive Learning) 적용
- 모델 변경(1D CNN → BERT)
- OSR 방법론 변경 (OpenMax → Background Method 적용)

## 설명
Simcse 및 BackGround Method 적용을 통한 혐오발언 분류 모델의 성능개선이 목적이다
### Simecse
Unsupervised SimCSE는 입력 문장을 받아 대조 학습 프레임워크에서 표준 드롭아웃만 노이즈로 사용하여 스스로 예측한다

Supervised SimCSE는 NLI 데이터 세트의 주석이 달린 쌍을 대조 학습에 통합하여 한 쌍을 긍정으로, 한 쌍을 부정으로 사용한다

다음 그림은 이 모델을 보여주는 그림이다

![image](https://github.com/CSID-DGU/2023-1-OSSP1-Extension-9/assets/22547157/d322d154-ae7b-420c-9c9f-35d35752b72d)

### BackGround Method
작성중



## 요구사항
* PyTorch
* Tensorflow
* Transformers
* sklearn
* sentencepiece

## 환경 및 언어
* Colab
* Jupyter Notebook
* Python

## 활용 데이터 셋
기존

korean-hate-speech 데이터셋 : <https://github.com/kocohub/korean-hate-speech>

Curse-detection-data 데이터셋 : <https://github.com/2runo/Curse-detection-data>

추가

korean_unsmile_dataset 데이터셋 : <https://github.com/smilegate-ai/korean_unsmile_dataset>

## 참고 프로젝트
기존

<https://github.com/CSID-DGU/2021-1-OSSP1-FloweryPath-8>

Simcse - Supervised

<https://github.com/BM-K/KoSimCSE-SKT>

Simcse - Unsupervised

<https://github.com/bhuvanakundumani/SimCSE_unsupervised>

BackGround Method

<https://github.com/Vastlab/Reducing-Network-Agnostophobia>

## 모델 학습 시작하기
1.참고 프로젝트의 Simcse코드를 사용해 KoBert모델을 사전학습하여 모델을 획득
  
  사용방법은 해당 페이지에 기록되어 있으므로 여기서 사용방법을 설명하지는 않겠다

2.사전학습이 완료된 모델에 `UnSup_KoBERT_V1`등의 폴더의 코드를 사용해 전이학습하여 분류 모델을 획득
  
3.`OpenMax`폴더에서 분류 모델에 맞는 코드를 사용해 Fit과정 진행 후 Predict를 사용해 문장 분류

## [모델 성능](https://github.com/CSID-DGU/2023-1-OSSP1-Extension-9/blob/main/new_model/README.md)



## 차후 계획
인터넷 용어에 맞는 Adversarial Sample 생성 후 Adversarial Training 적용을 통해 모델의 보안성 향상



## 라이선스

```
MIT License
Copyright (c) 2020 always0ne
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
