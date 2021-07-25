import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]  # 4
        self.action_size = self.env.action_space.n  # 2
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        # 메모리가 넘치지 않고 과거 데이터는 밀려나고 새로운 데이터가 삽입된다.
        # deque는 컬렉션 오브젝트, 2000개의 기억을 저장하여 나중에 사용
        # LIFO, stack이 LIFO자료구조의 대표적인 구조
        # FIFO, 먼저 들어간게 먼저 나오는 자료구조, Queue(deque가 이 특성을 갖는다.)
        # 데이터의 입출구가 바뀔 수 있음

        #### 하이퍼 파라미터
        # 벨만 방정식에 감마가 나옴
        self.gamma = 0.95  # discount rate, 미래보상에 곱해줄 숫자로 현재에 비해 미래보상치를 낮출때 사용
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001  # 미니멈을 정해줌으로써 탐험을 멈추지 않도록 한다.
        self.epsilon_decay = 0.9999
        self.batch_size = 64  # 배치사이즈가 성능에 큰 영향을 미치지는 않는다.
        self.train_start = 1000

        # 강화학습에는 하이퍼 파라미터와 파라미터가 존재
        # 하이퍼 파라미터는 개발자가 직접 세팅이 가능한것
        # 파라미터는 컴퓨터가 학습에 있어서만 조종가능한 파라미터, 계수(개발자가 세팅 불가(ex, 가중치는 직접 세팅 불가))
        # create main model
        # self.stae_size,에 콤마가 사용되어 튜플 형태임을 잊지말자
        self.model = build_model(input_shape=(self.state_size,), action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        # deque인 memory에 정보 기억
        # 5개의 정보를 기억해둠, 1000번이 쌓이면 리플레이하면서 학습하게됨
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # 상태정보만 있는것이 아닌 reward 등이 있음
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    # file 불러오기
    def load(self, name):
        self.model = load_model(name)

    # file 저장
    def save(self, name):
        self.model.save(name)

    # 자바로 치면 main격의 메소드
    def run(self):
        limit = 0

        for e in range(3000):
            state = self.env.reset()  # env 초기화 (cartpole-v1)
            state = np.reshape(state, [1, self.state_size])  # 상태 정보를 2차원 배열로 만들어라
            done = False  # 에피소드가 시작될 때 done을 초기화
            i = 0
            while not done:  # 한개의 에피소드, for문에 의해 1000번 반복
                # 에피소드가 돌아간다는것이 학습을 진행한다는 것
                # 게임의 데이터를 받아서 신경망을 학습 시키기 위한 것

                self.env.render() # 게임을 모니터링하는 구문
                action = self.act(state)  # 현재 상태에 대한 action을 구하는 것
                # 무작위(random) or 모델을 사용한 액션 추정(predict)
                next_state, reward, done, _ = self.env.step(action)  # 액션후의 다음 상태, reward, done을 추출
                # 액션 이후에 done이 트루가 되면 큰 실수를 한것(마이너스 점수)
                #                 print(reward)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:  # 액션을 통해 중간종료가 되지 않고 500번째가 아니라면
                    reward = reward  # 큰의미가 없는 코드
                elif done and i != self.env._max_episode_steps - 1:  # 행동을 통해 중간에 종료되게 되면 벌점을 크게 받음
                    reward = -100

                # 스텝당 정보를 기억한다, 학습을 목적으로 기억시키는 것(1000개 넘도록 모아서 신경망 학습이 목적)
                self.remember(state, action, reward, next_state, done)
                state = next_state  # 다음 루프에서 사용 될 수 있도록 state에 넣어줌
                i += 1  # 시각적으로 보기 위해 점수로 관리하는것, 시스템내부에서 사용되는 것이 아님
                # 500스텝에 도달할 때까지 게임이 종료되지 않았다는것과 같은 의미로 해석할 수 있기 때문에 점수 대용으로 사용 가능
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("***************************************************")
                        limit += 1
                        if limit == 10:
                            print("Saving trained model as cartpole-dqn_3.h5")
                            self.save("cartpole-dqn.h5")
                            return
                # 메모리에 저장된 게임 정보가 1000개이상이 되어야 실행됨
                self.replay()

    def test(self):
        self.load("cartpole-dqn_3.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break


def build_model(input_shape, action_space):
    # 모델의 입력부, 인풋 객체 생성
    X_input = Input(input_shape)

    X = Dense(16, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    # kernel_initializer를 써주면 빨라짐, 성능 향상에 도움, 제거해도 무방(가중치 초기화, 초기값 부여)
    X = Dense(16, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(16, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPoleDQNmodel')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    # loss=mse에러를 제곱해서 평균낸것 rho =0.95, epsilon을 줄여나가는 비, 0.95를 곱함 epsilon 최저 0.01까지 감소, 최소치

    model.summary()
    return model




# 학습 후 보상이 500에 도달하거나 1000회의 에피소드 학습 진행이 완료되면 모델의 가중치를 모델에 저장
# 학습
agent = DQNAgent()
agent.run()

agent.test()



