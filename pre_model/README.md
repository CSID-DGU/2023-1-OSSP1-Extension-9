OSR을 적용한 문장 분류 시스템   
============================   
### 2021-1-OSSP1-FloweryPath-8

꽃길팀   
팀원: 김규열 유천일 조건형 조성운 진하빈 최용진   

* 혐오성 문장, 일베 문장, 성차별 문장과 Unknown 문장으로 4가지 클래스 문장 분류   
* k-fold 교차 검증 적용   
* Unknown 처리를 위한 OpenMax 구현   

Requirement   
-----------   
* tensorflow   
* keras   
* Tokenizer   

구현 환경 및 언어   
-----------------
환경: Jupyter Notebook   
언어: Python   

활용 데이터셋   
-------------   
혐오성, 성차별 문장 데이터셋 : https://github.com/kocohub/korean-hate-speech   
일베 문장 데이터셋 : https://github.com/2runo/Curse-detection-data

주요 기술   
---------

### Open Set Recognition   
기계학습을 시킬 때 주어진 모델로는 구분할 수 없는 데이터들을 Unknown이라는 새로운 클래스를 도입하여 정의함으로써 더 다양하고 정확한 결과를 도출할 수 있도록 하는 학습 방법론
이 프로젝트에서는 OpenMax를 이용하여 구현   

### 합성곱 신경망(CNN)   
자연어 처리를 위해 1D CNN 적용   
![image](https://user-images.githubusercontent.com/80958412/122702135-b2f9cd00-d289-11eb-8eef-c49ac7fc3100.png)   
[사진출처: https://wikidocs.net]   

실행 결과
---------
### Unknown 문장   
입력 :   
![image](https://user-images.githubusercontent.com/80958412/122686262-203a3d80-d24b-11eb-9ca5-d05c319f80de.png)   
출력 :   
![image](https://user-images.githubusercontent.com/80958412/122686324-54156300-d24b-11eb-8fec-db6238875637.png)   

### 혐오성 문장
입력 :   
![image](https://user-images.githubusercontent.com/80958412/122686363-81faa780-d24b-11eb-9f84-b15b4fa866ec.png)   
출력 :   
![image](https://user-images.githubusercontent.com/80958412/122686368-8fb02d00-d24b-11eb-9d7e-a79f010d8400.png)   

업데이트 계획
------------
추가적인 웹 학습을 진행하여 웹을 통한 입/출력부를 구현할 예정
