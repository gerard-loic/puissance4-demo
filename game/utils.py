from board import EMPTY_CELL, Board, PLAYER_ONE, PLAYER_TWO
import numpy as np

def get_available_moves(board:Board):
    #Définir tous les coups possibles
    availableMoves = []
    for col in range(board.cols):
        for row in range(board.rows):
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


def get_available_moves_cf(board:Board):
    #Définir tous les coups possibles
    availableMoves = []
    for col in range(board.cols):
        for row in range(board.rows-1, -1, -1): #Parcourir dans le sens contraire
            if board.board[col][row] == EMPTY_CELL:
                availableMoves.append((row, col))
                break   #On a trouvé une position dans la colonne, donc on passe à la colonne suivante
    return availableMoves

def heuristic_evaluation(board:Board, from_player_perspective:int):
    score = 0

    #Score pour les positions verticales
    for col in range(board.cols):
        for row in range(board.rows - board.connexions_to_win + 1):
            connected_positions = [(row+i, col) for i in range(4)]
            loc_score = _heuristic_scoring(board=board, connected_positions=connected_positions, from_player_perspective=from_player_perspective)

            if abs(loc_score) == 1: #Indique que le jeu est terminé
                return loc_score
            score += loc_score

    #Score pour les positions horizontales
    for row in range(board.rows):
        for col in range(board.cols - board.connexions_to_win + 1):
            connected_positions = [(row, col+i) for i in range(4)]
            loc_score = _heuristic_scoring(board=board, connected_positions=connected_positions, from_player_perspective=from_player_perspective)

            if abs(loc_score) == 1: #Indique que le jeu est terminé
                return loc_score
            score += loc_score

    #Score pour les positions diagonales
    for col in range(board.cols - board.connexions_to_win + 1):
        for row in range(board.rows - board.connexions_to_win + 1):
            connected_positions = [(row+i, col+i) for i in range(4)]
            loc_score = _heuristic_scoring(board=board, connected_positions=connected_positions, from_player_perspective=from_player_perspective)

            if abs(loc_score) == 1: #Indique que le jeu est terminé
                return loc_score
            score += loc_score

    #Score pour les positions anti-diagonales
    for col in range(board.connexions_to_win -1, board.cols):
        for row in range(board.rows - board.connexions_to_win + 1):
            connected_positions = [(row+i, col-i) for i in range(4)]
            loc_score = _heuristic_scoring(board=board, connected_positions=connected_positions, from_player_perspective=from_player_perspective)

            if abs(loc_score) == 1: #Indique que le jeu est terminé
                return loc_score
            score += loc_score

    return score

def _disc_counter_map(board:Board, positions:list):
    disc_counter = {}
    for row, col in positions:
        disk = board.board[col][row]
        disc_counter[disk] = disc_counter.get(disk, 0) + 1
    return disc_counter    
    
def _heuristic_scoring(board:Board, connected_positions:list, from_player_perspective:int):
    other_player = PLAYER_ONE if from_player_perspective == PLAYER_TWO else PLAYER_TWO
    disk_counter_map = _disc_counter_map(board=board, positions=connected_positions)

    if disk_counter_map.get(from_player_perspective, 0) == 3 and disk_counter_map.get(EMPTY_CELL, 0) == 1:
        return 0.01
    elif disk_counter_map.get(other_player, 0) == 3 and disk_counter_map.get(EMPTY_CELL, 0) == 1:
        return -0.1
    elif disk_counter_map.get(from_player_perspective, 0) == 4:
        return 1
    elif disk_counter_map.get(other_player, 0) == 4:
        return -1
    return 0
        



if __name__ == "__main__":
    _board = Board(cols=3, rows=3, connexions_to_win=3)
    _board.play_move(row=0, col=2)
    _board.next_player()
    _board.play_move(row=0, col=1)
    _board.next_player()
    print(_board)

    _repr_board = equivalent_board_representation(board=_board)
    print(_repr_board)