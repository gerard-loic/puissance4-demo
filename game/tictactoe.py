from board import Board, PLAYER_ONE, PLAYER_TWO
from players import RandomPlayer, HumanPlayer, MinimaxPlayer, DynamicprogrammingPlayer, MontecarloPlayer, TemporaldifferencePlayer, TDPlayer
from tqdm import tqdm
import time
import numpy as np

_player_categories = {
    'human_user' : HumanPlayer,
    'random_player' : RandomPlayer,
    'minimax_player' : MinimaxPlayer,
    'dynamicprogramming_player' : DynamicprogrammingPlayer,
    'montecarlo_player' : MontecarloPlayer,
    'temporaldifference_player' : TDPlayer
}

class TicTacToe:
    def __init__(self, player_one, player_two, display_board:bool = True):
        self.board = Board(rows=3, cols=3, connexions_to_win=3)
        self.display_board = display_board
        self.current_player = _player_categories[player_one](position=PLAYER_ONE)
        self.next_player = _player_categories[player_two](position=PLAYER_TWO)
        self.timesP1 = []
        self.timesP2 = []
        self.computationP1 = []
        self.computationP2 = []


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
                if self.current_player == PLAYER_ONE:
                    self.computationP1.append(self.current_player.stats)
                    self.computationP2.append(self.next_player.stats)
                else:
                    self.computationP1.append(self.next_player.stats)
                    self.computationP2.append(self.current_player.stats)

                if self.display_board:
                    print(self.board)
                    if there_is_a_winner:
                        print(f"Player {'X' if self.current_player.position == PLAYER_ONE else 'O'} wins !")
                    else:
                        print("It's a draw !")
       
                    if self.current_player.position == PLAYER_ONE:
                        print(f"Stats player 1 {self.current_player.stats} - {np.mean(self.timesP1)}")
                        print(f"Stats player 2 {self.next_player.stats} - {np.mean(self.timesP2)}")
                    else:
                        print(f"Stats player 1 {self.next_player.stats} - {np.mean(self.timesP2)}")
                        print(f"Stats player 2 {self.current_player.stats} - {np.mean(self.timesP1)}")
                if returnWinner:
                    if there_is_a_winner:
                        return 1 if self.current_player.position == PLAYER_ONE else -1
                    else:
                        return 0
                break
            #Intervertir les joueurs
            self.current_player, self.next_player = self.next_player, self.current_player
            self.board.next_player()

class GameSession:
    def __init__(self, player_one:str, player_two:str, number_of_games:int=1):
        self.player_one = player_one
        self.player_two = player_two
        self.number_of_games = number_of_games

        self.winP1 = 0
        self.winP2 = 0
        self.draw = 0
        self.computationsP1 = []
        self.computationsP2 = []
        self.timeP1 = []
        self.timeP2 = []

    def start(self):
        for _ in tqdm(range(self.number_of_games), f"{self.player_one} against {self.player_two} : "):
            game = TicTacToe(player_one=self.player_one, player_two=self.player_two, display_board=False)
            r = game.playGame(returnWinner=True)
            if r == 1:
                self.winP1 += 1
            elif r == -1:
                self.winP2 += 1
            else:
                self.draw += 1
            self.timeP1.append(np.mean(game.timesP1))
            self.timeP2.append(np.mean(game.timesP2))
            self.computationsP1.append(np.mean(game.computationP1))
            self.computationsP2.append(np.mean(game.computationP2))


    def getStats(self):
        return {
            'p1_win' : self.winP1,
            'p2_win' : self.winP2,
            'draw' : self.draw,
            'p1_avg_time' : np.mean(self.timeP1),
            'p2_avg_time' : np.mean(self.timeP2),
            'p1_avg_computations' : np.mean(self.computationsP1),
            'p2_avg_computations' : np.mean(self.computationsP2)
        }

if __name__ == "__main__":
    gameSession = GameSession(player_one='temporaldifference_player', player_two='dynamicprogramming_player', number_of_games=50)
    gameSession.start()

    print(gameSession.getStats())

  