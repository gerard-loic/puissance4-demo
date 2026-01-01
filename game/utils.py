from board import EMPTY_CELL, Board
import numpy as np

def get_available_moves(board:Board):
    #Définir tous les coups possibles
    availableMoves = []
    for col in range(board.rows):
        for row in range(board.cols):
            if board.board[col][row] == EMPTY_CELL:
                availableMoves.append((row, col))
    return availableMoves

def equivalent_board_representation(board:Board):
    #Convertion en un array numpy (pour translation plus facile)
    #Le plateau est converti en array NumPy pour faciliter les transformations géométriques.
    board_array = np.array(board.board)

    #Utilisation de set pour ne pas avoir de doublons
    #Un ensemble (set) pour stocker toutes les positions équivalentes sans doublons.
    equivalent_positions = set([])

    #4 rotations
    for _ in range(4):
        #Convertit le plateau en tuple de tuples (car les listes ne sont pas "hashables" pour un set)
        board_tuples = tuple(map(tuple, board_array))
        equivalent_positions.add(board_tuples)

        #On effectue la rotation (clockwise)
        board_array = np.rot90(board_array, k=1)

    #Reflexion selon la diagonale
    reflected_board = np.transpose(board_array)

    #4 rotations
    for _ in range(4):
        board_tuples = tuple(map(tuple, reflected_board))
        equivalent_positions.add(board_tuples)

        #On effectue la rotation (clockwise)
        reflected_board = np.rot90(reflected_board, k=1)

    #Parmi toutes les positions équivalentes, on choisit le maximum lexicographique. 
    # Cela garantit que deux plateaux équivalents auront toujours la même représentation choisie.
    representative_position = max(equivalent_positions)
    representative_board = Board(
        rows=board.rows,
        cols=board.cols,
        connexions_to_win=board.connexions_to_win,
        player_to_move=board.current_player,
        board=[list(tup) for tup in representative_position]
    )

    return representative_board


if __name__ == "__main__":
    _board = Board(cols=3, rows=3, connexions_to_win=3)
    _board.play_move(row=0, col=2)
    _board.next_player()
    _board.play_move(row=0, col=1)
    _board.next_player()
    print(_board)

    _repr_board = equivalent_board_representation(board=_board)
    print(_repr_board)