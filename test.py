import wrapped_flappy_bird as game
import numpy as np
import cv2
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from collections import deque
import random

ACTION_SIZE = 2
BATCH_SIZE = 32
OBSERVATION = 5000

class  FlappyBirdDQN:
    def __init__(self, image, action_size):
        self.img_rows = image.shape[0]
        self.img_cols = image.shape[1]
        self.channels = image.shape[2]
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.init_epsilon = 0.1
        self.fin_epsilon = 0.01
        self.epsilon = self.init_epsilon
        self.explore = 3000000
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(self.img_rows, self.img_cols, self.channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))
        adam = Adam(lr=1e-4)
        model.compile(loss='mse', optimizer=adam)
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        '''
        for state, action_index, reward, next_state, done in minibatch:
            q_value = reward
            if not done:
                q_value = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            q_tar = self.model.predict(state)
            q_tar[0][action_index] = q_value
            self.model.fit(state, q_tar, epochs=1, verbose=0)
        '''
        state_batch = np.asarray([d[0] for d in minibatch])
        actindex_batch = np.asarray([d[1] for d in minibatch])
        reward_batch = np.asarray([d[2] for d in minibatch])
        nstate_batch = np.asarray([d[3] for d in minibatch])
        done_batch = np.asarray([d[4] for d in minibatch])
        y_batch = self.model.predict(state_batch)
        q_batch = np.zeros(batch_size)
        qn_batch = self.model.predict(nstate_batch)
        for i in range(batch_size):
            if done_batch[i]:
                q_batch[i] = reward_batch[i]
            else:
                q_batch[i] = reward_batch[i] + self.gamma * np.amax(qn_batch[i])
            y_batch[i][actindex_batch[i]] = q_batch[i]
        self.model.fit(state_batch, y_batch, batch_size=batch_size, verbose=0)
        # loss = self.model.evaluate(state_batch, y_batch, verbose=0)
        if self.epsilon > self.fin_epsilon:
            self.epsilon -= (self.init_epsilon - self.fin_epsilon) / self.explore
        # return loss


def preprocess(image):
    image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image


if __name__ == '__main__':
    game_state = game.GameState()
    init_action = np.array([1, 0])
    image_org, reward, done = game_state.frame_step(init_action)
    image = preprocess(image_org)
    image = np.stack((image, image, image, image), axis=2)
    # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    state = image
    bird = FlappyBirdDQN(state, ACTION_SIZE)
    #bird.model = load_model('train_model_bird700000.h5')
    t = 0

    while True:
        t += 1
        loss = 0
        action_index = bird.act(state)
        action = np.zeros(ACTION_SIZE)
        action[action_index] = 1
        image_org, reward, done = game_state.frame_step(action)
        image = preprocess(image_org)
        # image = image.reshape(1, image.shape[0], image.shape[1], 1)
        # image = np.append(image, state[:, :, :, :3], axis=3)
        image = image.reshape(image.shape[0], image.shape[1], 1)
        image = np.append(image, state[:, :, :3], axis=2)
        next_state = image
        bird.remember(state, action_index, reward, next_state, done)
        state = next_state

        if t > OBSERVATION:
            bird.replay(BATCH_SIZE)
            # loss += bird.replay(BATCH_SIZE)
            if t % 5000 == 0:
                bird.model.save('train_model_bird' + str(t) + '.h5')
                print('Now save model. Time:', t)

        if reward == 1:
            print('T:', t, '| epsilon:', bird.epsilon, '| reward:', reward)
        if (reward != 1) and (t % 1000 == 0):
            print('T:', t, '| epsilon:', bird.epsilon)

