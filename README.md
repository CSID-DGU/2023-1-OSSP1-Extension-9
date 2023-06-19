<div><img src="https://capsule-render.vercel.app/api?type=waving&animation=fadeIn&color=auto&height=300&section=header&text=Extension&fontSize=90" /></div>

## 프로젝트 소개
<!--Wirte one paragraph of project description -->  
### 혐오발언 인지 언어모델 개선 프로젝트

## 👋 팀원 소개
|이름|학번|학과|
|------|---|---|
|이상돈|2019112491|산업시스템공학과
|김건형|2018112016|컴퓨터공학과|
|이종혁|2019112035|컴퓨터공학과|
|황유경|2020111403|경영학과|
|민준영|2019113290|컴퓨터공학과|

## Overview
- 학습 데이터 셋 추가 및 데이터 라벨링 개선
- 지도학습 → 자기지도학습(Contrastive Learning) 적용
- 모델 변경(1D CNN → BERT)
- OSR 방법론 변경 (OpenMax → Background Method 적용)

## Description
Simcse 및 BackGround Method 적용을 통한 혐오발언 분류 모델의 성능개선

## Requirement
PyTorch, Tensorflow, Transformers, sklearn, sentencepiece

## 환경 및 언어
Colab, Jupyter Notebook
Python

## 활용 데이터 셋
기존
혐오성, 성차별 문장 데이터셋 : <https://github.com/kocohub/korean-hate-speech>
일베 문장 데이터셋 : <https://github.com/2runo/Curse-detection-data>

추가
<https://github.com/smilegate-ai/korean_unsmile_dataset>

## 참고 프로젝트
기존
<https://github.com/CSID-DGU/2021-1-OSSP1-FloweryPath-8>

Simcse - Supervised
<https://github.com/BM-K/KoSimCSE-SKT>

Simcse - Unsupervised
<https://github.com/bhuvanakundumani/SimCSE_unsupervised>

## Getting Started
1.참고 프로젝트의 Simcse코드를 사용해 KoBert모델을 사전학습
2.`UnSup_KoBERT_V1`등의 폴더의 코드를 사용해 전이학습
3.`OpenMax`폴더에서 해당 모델에 맞는 코드를 사용해 Fit and Calculate


## 차후 계획
Adversarial Training 적용


## License

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
