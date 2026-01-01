#Constantes
PLAYER_ONE = 1
PLAYER_TWO = 2
EMPTY_CELL = 0
STR_MAPPING = {
    PLAYER_ONE : 'X',
    PLAYER_TWO : 'O',
    EMPTY_CELL : '-'
}

class Board:
    @staticmethod
    def cell_to_string(cell):
        return STR_MAPPING[cell]
    

    def __init__(self, rows:int, cols:int, connexions_to_win, board=None, player_to_move=PLAYER_ONE):
        self.cols = cols
        self.rows = rows
        self.connexions_to_win = connexions_to_win
        self.current_player = player_to_move
        self.board = [[EMPTY_CELL] * self.rows for _ in range(self.cols)] if board is None else board

    def next_player(self):
        self.current_player=  PLAYER_ONE if self.current_player == PLAYER_TWO else PLAYER_TWO

    def play_move(self, row, col):
        assert self.board[col][row] == EMPTY_CELL, f"The call {row, col} is not empty."
        self.board[col][row] = self.current_player

    def game_has_ended(self):
        if self._horizontal_check() or self._vertical_check() or self._diagonal_check() or self._diagonal_decreasing_check():
            return True, True
            #Le jeu est fini ET on a un gagnant
        elif all([self.board[col][row] != EMPTY_CELL for col in range(self.cols) for row in range(self.rows)]):
            return True, False
            #Le jeu est fini ET on a PAS de gagnant
        return False, False
        #Le jeu n'est pas fini
    
    def _horizontal_check(self):
        #Cette boucle parcourt toutes les lignes du plateau (de haut en bas).
        for row in range(self.rows):
            #Cette boucle parcourt les colonnes, mais pas toutes.
            #Imaginons un plateau avec cols = 4 et connexions_to_win = 3 :
            #  | A B C D
            #1 | - - - -
            #Pour vérifier une séquence horizontale de 3 pions, on peut commencer la vérification :
            #En colonne A (indices 0, 1, 2) ✅
            #En colonne B (indices 1, 2, 3) ✅
            #En colonne C ? NON, car il ne reste que 2 positions (2, 3) ❌
            #Donc on s'arrête à col = 4 - 3 + 1 = 2, c'est-à-dire les colonnes A et B (indices 0 et 1).
            for col in range(self.cols - self.connexions_to_win + 1):
                #Cette expression vérifie que toutes les cellules consécutives contiennent la même valeur.
                if all(self.board[col + i][row] == self.board[col][row] for i in range(self.connexions_to_win)) and self.board[col][row] != EMPTY_CELL:
                    return True
        return False        
        

    def _vertical_check(self):
        #On boucle sur les colonnes
        for col in range(self.cols):
            for row in range(self.rows - self.connexions_to_win + 1):
                #On véfifie que toutes les cellules consécutives contiennent la même valeur
                if all(self.board[col][row + i] == self.board[col][row] for i in range(self.connexions_to_win)) and self.board[col][row] != EMPTY_CELL:
                    return True
        return False


    def _diagonal_check(self):
        for col in range(self.cols - self.connexions_to_win + 1):
            for row in range(self.rows - self.connexions_to_win + 1):
                if all(self.board[col + i][row + i] == self.board[col][row] for i in range(self.connexions_to_win)) and self.board[col][row] != EMPTY_CELL:
                    return True
        return False

    def _diagonal_decreasing_check(self):
        for col in range(self.cols - self.connexions_to_win + 1):
            for row in range(self.rows - self.connexions_to_win + 1):
                start_row = row + self.connexions_to_win - 1
                if all(self.board[col + i][start_row - i] == self.board[col][start_row] for i in range(self.connexions_to_win)) and self.board[col][start_row] != EMPTY_CELL:
                    return True
        return False

    #Surcharge print(instance)
    def __str__(self):
        res_str = '\n'.join(
            f"{ row + 1} | " + ' '.join(self.cell_to_string(self.board[col][row]) for col in range(self.cols))
            for row in range(self.rows)
        )
        letters = [chr(i) for i in range(ord('A'), ord('A') + self.cols)]
        res_str += '\n'
        res_str += '  | ' + ' '.join(letters) + '\n'
        return res_str

    #Représentation (pour machine learning)
    def __repr__(self):
        board_repr = f"{self.current_player}"
        for col in range(self.cols):
            board_repr += '|' + ''.join([str(plr_type) for plr_type in self.board[col]])
        return board_repr


if __name__ == "__main__":
    board = Board(rows=3, cols=4, connexions_to_win=3)
    board.play_move(row=0, col=2)
    board.play_move(row=1, col=1)
    board.play_move(row=2, col=0)
   
    #board.play_move(2, 0)
    print(board._diagonal_decreasing_check())
    print(board)
    
    