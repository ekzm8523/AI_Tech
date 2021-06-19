# 문장 내 개체간 관계 추출

- 개요
    - 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

<img width="702" alt="_2021-06-19_16 11 51" src="https://user-images.githubusercontent.com/67869514/122636038-15d05480-d122-11eb-91b3-fc3186b5f8c8.png">


    위 그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

    이번 대회에서는 문장, 엔티티, 관계에 대한 정보를 통해 ,문장과 엔티티 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 엔티티들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 model이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.

    - input: sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.
<img width="863" alt="_2021-06-19_16 12 26" src="https://user-images.githubusercontent.com/67869514/122636053-254f9d80-d122-11eb-8523-8886d7fdb968.png">


    - output: relation 42개 classes 중 1개의 class를 예측한 값입니다.
    - 위 예시문에서 단체:별칭의 label은 6번(아래 label_type.pkl 참고)이며, 즉 모델이 sentence, entity 1과 entity 2의 정보를 사용해 label 6을 맞추는 분류 문제입니다.

---

- 데이터

    전체 데이터에 대한 통계는 다음과 같습니다. 학습에 사용될 수 있는 데이터는 train.tsv 한 가지 입니다. 주어진 데이터의 범위 내 혹은 사용할 수 있는 외부 데이터를 적극적으로 활용하세요!

    train.tsv: 총 9000개

    test.tsv: 총 1000개 (정답 라벨 blind)

    answer: 정답 라벨 (비공개)

    학습을 위한 데이터는 총 9000개 이며, 1000개의 test 데이터를 통해 리더보드 순위를 갱신합니다. private 리더보드는 운영하지 않는 점 참고해 주시기바랍니다.

    label_type.pkl: 총 42개 classes (class는 아래와 같이 정의 되어 있며, 평가를 위해 일치 시켜주시길 바랍니다.) pickle로 load하게 되면, 딕셔너리 형태의 정보를 얻을 수 있습니다.

<img width="860" alt="_2021-06-19_16 13 09" src="https://user-images.githubusercontent.com/67869514/122636062-2b457e80-d122-11eb-83e7-ea686327f317.png">


<img width="877" alt="_2021-06-19_16 13 20" src="https://user-images.githubusercontent.com/67869514/122636068-313b5f80-d122-11eb-8f39-6640309bd66e.png">


    - column 1: 데이터가 수집된 정보.
    - column 2: sentence.
    - column 3: entity 1
    - column 4: entity 1의 시작 지점.
    - column 5: entity 1의 끝 지점.
    - column 6: entity 2
    - column 7: entity 2의 시작 지점.
    - column 8: entity 2의 끝 지점.
    - column 9: entity 1과 entity 2의 관계를 나타내며, 총 42개의 classes가 존재함.
    - class에 대한 정보는 위 label_type.pkl를 따라 주시기 바랍니다.
