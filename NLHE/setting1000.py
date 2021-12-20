from Player import Player
from Hand import NLHEHand
from Agents.Random_Agent import Random_Agent
import random

agent0 = Random_Agent()
player0 = Player(id=0, agent=agent0)
agent1 = Random_Agent()
player1 = Player(id=1, agent=agent0)
agent2 = Random_Agent()
player2 = Player(id=2, agent=agent0)
agent3 = Random_Agent()
player3 = Player(id=3, agent=agent0)

players_in = [player0, player1, player2, player3]

episode_counter = 0
while(True):
    NLHE = NLHEHand(0.5, players_in[:])
    NLHE.play_hand()
    print(f'\nEpisode done: {episode_counter}')
    for p in players_in:
        print(f'Player id={p.id}, total winnings={p.total_winnings}, average last 100 = {sum(p.last100)/len(p.last100)}')
    episode_counter += 1

    winnings_sum = 0
    for p in players_in:
        winnings_sum += p.total_winnings

    if episode_counter == 100:
        break
