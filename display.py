import wrapped_flappy_bird as game
import numpy as np
from keras.models import load_model
from test import FlappyBirdDQN, preprocess

if __name__ == '__main__':
    ACTION_SIZE = 2
    BATCH_SIZE = 32
    game_state = game.GameState()
    init_action = np.array([1, 0])
    image_org, reward, done = game_state.frame_step(init_action)
    image = preprocess(image_org)
    image = np.stack((image, image, image, image), axis=2)
    # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    state = image
    bird = FlappyBirdDQN(state, ACTION_SIZE)
    bird.epsilon = 0
    bird.model = load_model('train_model_bird900000.h5')
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

        if reward == 1:
            print('T:', t, '| epsilon:', bird.epsilon, '| reward:', reward)
        if (reward != 1) and (t % 100 == 0):
            print('T:', t, '| epsilon:', bird.epsilon)
