from agent import Agent
from game import SnakeGameAI


def train():
    record = 0
    agent = Agent(name='cracked')
    
    game = SnakeGameAI()
    while True:
        # get old state 
        state_old = agent.get_state(game)
        # get move based on current state
        action = agent.get_action(state_old)
        # do the move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # experienced replay - train again on ALL previous moves to improve 
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            record = max(record, score)
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            if agent.n_games % 10 == 0:
                agent.save()


if __name__ == '__main__':
    train()
