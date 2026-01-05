import random
from board import Board, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL
import utils


class RandomPlayerCF:
    def __init__(self, position:int):
        self.position = position

    @staticmethod
    def play(board:Board):
        return random.choice(utils.get_available_moves_cf(board))
    


class HumanPlayerCF:
    def __init__(self, position:int):
        self.position = position

    @staticmethod
    def play(board:Board):
        while True:
            try:
                #Demande à l'utilisateur ou est ce qu'il veut jouer
                move = input("Enter the column that you would like to play (ex. A) :  ").strip().upper()
                col = ord(move[0]) - ord('A')

                if all(board.board[col][row] != EMPTY_CELL for row in range(board.rows)):
                    print("Column is full, try again")
                else:
                    for row in range(board.rows-1, -1, -1):
                        if board.board[col][row] == EMPTY_CELL:
                            return (row, col)
            except (IndexError, ValueError):
                print("This input is not correct !")
    

class HeuristicPlayer:
    def __init__(self, position:int):
        self.position = position

    def play(self, board:Board):
        best_move = None
        best_score = float('-inf')
        available_moves = utils.get_available_moves_cf(board=board)

        random.shuffle(available_moves) #Pour ne pas toujours sortir les memes coups au début
        for row, col in available_moves:
            board.play_move(row=row, col=col)   #On joue le coup
            score = utils.heuristic_evaluation(board=board, from_player_perspective=self.position)
            board.board[col][row] = EMPTY_CELL    #Annule le coup
            if score > best_score:
                best_score = score
                best_move = (row, col)

        return best_move