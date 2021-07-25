# DQN_Keras_Cartpole

OpenAI의 gym라이브러리를 사용하여 Cartpole(움직이는 카트위의 막대를 넘어지지 않게 하는 환경)에서의 막대가 넘어지지 않도록 카트를 이동시키는 강화학습

<img src="https://user-images.githubusercontent.com/87750521/126890888-03bae56e-11b3-40b4-aaba-24bd0667e789.png" width="500" height="300">

### 구조
1) gym라이브러리의 CartPole-v1의 객체를 사용하여 학습에 사용할 DQNAgent 클래스
2) 학습에 사용할 keras 모델
3) 학습, 테스트 진행

### 학습
 - 500스텝으로 1회 Episode, 총 3000회의 Episode로 학습 진행
 - 막대가 넘어지지 않도록 카트를 조정하는 방식으로 학습 진행
 - 처음에는 랜덤으로 카트를 움직여보면서 데이터를 쌓고 학습을 진행 하고, 점점 랜덤이 아닌 학습된 데이터로 카트를 움직이고 이에 따른 reward(보상값)를 통해 다시 학습
 - 277번째 Episode에서 처음으로 Episode가 끝날때까지 막대를 넘어뜨리지 않음
 - 이후 다시 한동안 Episode가 끝날때까지 안넘어지게 하지 못하다가 564번째 Episode에서 다시 끝까지 막대가 안넘어짐

### 결과
<img src="https://user-images.githubusercontent.com/87750521/126891375-9fceca86-b3c7-4828-994b-5b7b71fd038a.png" width="300" height="200">
