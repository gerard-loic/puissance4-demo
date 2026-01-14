from tqdm import tqdm
from board import Board, PLAYER_ONE, PLAYER_TWO
from players_connect4 import RandomPlayerCF, HumanPlayerCF, HeuristicPlayer, LookAheadheuristicPlayer, CustomLookAheadheuristicPlayer, DRLPlayer, TsDRLPlayer
import time 
import numpy as np

_player_categories = {
    'human_user' : HumanPlayerCF,
    'random_player' : RandomPlayerCF,
    'heuristic_player' : HeuristicPlayer,
    'lookaheadheuristic_player' : LookAheadheuristicPlayer,
    'customlookaheadheuristic_player' : CustomLookAheadheuristicPlayer,
    'drl_player' : DRLPlayer,
    'ts_drl_player' : TsDRLPlayer
}

class Connect4:
    def __init__(self, player_one, player_two, display_board:bool = True):
        self.board = Board(rows=6, cols=7, connexions_to_win=4)
        self.display_board = display_board
        self.current_player = _player_categories[player_one](position=PLAYER_ONE)
        self.next_player = _player_categories[player_two](position=PLAYER_TWO)
        self.timesP1 = []
        self.timesP2 = []

    
    def playGame(self, returnWinner:bool=False):
        while True:
            if self.display_board:
                print(self.board)
                print(f"Player : {'X' if self.current_player.position == PLAYER_ONE else 'O'}\n\n")

            ts_before = int(time.time() * 1000)
            row, col = self.current_player.play(board=self.board)
            ts_after = int(time.time() * 1000)
      
            if self.current_player.position == PLAYER_ONE:
                self.timesP1.append(ts_after-ts_before)
            else:
                self.timesP2.append(ts_after-ts_before)

            self.board.play_move(row, col)
            game_ended, there_is_a_winner = self.board.game_has_ended()
            if game_ended:

                if self.display_board:
                    print(self.board)
                    if there_is_a_winner:
                        print(f"Player {'X' if self.current_player.position == PLAYER_ONE else 'O'} wins !")
                    else:
                        print("It's a draw !")
       
                    if self.current_player.position == PLAYER_ONE:
                        print(f"Stats player 1 {np.mean(self.timesP1)}ms")
                        print(f"Stats player 2 {np.mean(self.timesP2)}ms")
                    else:
                        print(f"Stats player 1 {np.mean(self.timesP2)}ms")
                        print(f"Stats player 2 {np.mean(self.timesP1)}ms")
                if returnWinner:
                    if there_is_a_winner:
                        return 1 if self.current_player.position == PLAYER_ONE else -1
                    else:
                        return 0
                break
            #Intervertir les joueurs
            self.current_player, self.next_player = self.next_player, self.current_player
            self.board.next_player()


class GameSessionCF:
    def __init__(self, player_one:str, player_two:str, number_of_games:int=1, display_board:bool=False):
        self.player_one = player_one
        self.player_two = player_two
        self.number_of_games = number_of_games
        self.display_board = display_board

        self.winP1 = 0
        self.winP2 = 0
        self.draw = 0
        self.timeP1 = []
        self.timeP2 = []

    def start(self):
        for _ in tqdm(range(self.number_of_games), f"{self.player_one} against {self.player_two} : "):
            game = Connect4(player_one=self.player_one, player_two=self.player_two, display_board=self.display_board)
            r = game.playGame(returnWinner=True)
            if r == 1:
                self.winP1 += 1
            elif r == -1:
                self.winP2 += 1
            else:
                self.draw += 1
            self.timeP1.append(np.mean(game.timesP1))
            self.timeP2.append(np.mean(game.timesP2))


    def getStats(self):
        return {
            'p1_win' : self.winP1,
            'p2_win' : self.winP2,
            'draw' : self.draw,
            'p1_avg_time' : np.mean(self.timeP1),
            'p2_avg_time' : np.mean(self.timeP2)
        }

if __name__ == "__main__":
    game = Connect4(player_one="human_user", player_two="drl_player", display_board=True)
    game.playGame()

    #gamesession = GameSessionCF(player_one="lookaheadheuristic_player", player_two="drl_player", number_of_games=100)
    #gamesession.start()
    #print(gamesession.getStats())

    #Training DRL_Player
    """
    num_training_iterations = 20000
    games_per_iteration = 6
    drl_player = DeepReenforcementLearningPlayer(position=PLAYER_ONE, training_mode=True)

    for i in tqdm(range(1, num_training_iterations + 1), desc="Training the network"):
        for _ in range(games_per_iteration):
            game = Connect4(player_one='drl_player', player_two='drl_player', display_board=False)
            game.playGame()
        
        #Train the network and save weights
        drl_player.update()
        drl_player.save_weights()
    print("Training completed !")
    """
  
    """
    num_training_iterations = 20000
    games_per_iterations = 6
    ts_drl_player = TsDRLPlayer(position=PLAYER_ONE)
    TsDRLPlayer.setTrainingMode(actif=True)
    
    TsDRLPlayer.enableExperienceCache()
    for i in tqdm(range(1, num_training_iterations+1), desc='Training iterations'):
        for i in range(games_per_iterations):
            game = Connect4(player_one='ts_drl_player', player_two='ts_drl_player', display_board=False)
            game.playGame()
    
        #Train the networks and save the weights
        ts_drl_player.update()
        ts_drl_player.save_weights()
        TsDRLPlayer.clearExperienceCache()
    print('Training is complete.')
    """