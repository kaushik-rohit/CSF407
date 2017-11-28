import gym
import gym_ple
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D , MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
import random
import gym
import numpy as np
from collections import deque
import time as t
import os
import cv2

EPISODES = 10000

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.000001
        self.model = self.make_model()
        
        if os.path.isfile("./save/model.h5"):
            print "loading model"
            self.load("./save/model.h5")
    
    
    def preprocess(self, state):
        ret = state.copy()
        
        cv2.normalize(ret, ret, 0, 255, cv2.NORM_MINMAX)
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
        ret = cv2.adaptiveThreshold(ret, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,11,2)
        resz_img = cv2.resize(ret, (80,80)) #resize it to 25*25 image
        resz_img = resz_img/255
        return resz_img
        
    def make_model(self):
            print("Now we build the model")
            model = Sequential()
            model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(80,80,4)))  #80*80*4
            model.add(Activation('relu'))
            model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
            model.add(Activation('relu'))
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dense(2))
           
            adam = Adam(lr=self.learning_rate)
            model.compile(loss='mse',optimizer=adam)
            print("We finish building the model")
            return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)
        
    def train(self, env):
        done = False
        batch_size = 32        
            
        for e in range(EPISODES):
            state = env.reset()
            xt = agent.preprocess(state)
            cv2.imshow('img', xt)
            st = np.stack((xt, xt, xt, xt), axis=2)
            st = st.reshape((1,80,80,4))
            for time in range(500):
                env.render()
                #t.sleep(0.1)
                action = self.act(st)
                next_state, reward, done, _ = env.step(action)
                xt1 = self.preprocess(next_state)
                xt1 = xt1.reshape((1,80,80,1))
                st1 = np.append(xt1, st[:, :,:, :3], axis=3)
                reward = reward if not done else -10
                self.remember(st, action, reward, st1, done)
                #print len(self.memory)
                state = next_state
                st=st1
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, time, agent.epsilon))
                    break
            if len(self.memory) > batch_size:
                self.replay(batch_size)
            
            if e%101 == 0:
                print "Saving model"
                self.save("./save/model.h5")
        
if __name__ == '__main__':
    
    env = gym.make('FlappyBird-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    agent.train(env)
