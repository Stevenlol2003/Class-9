import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = True   # TODO: detect drop phase

        count_B = sum((i.count('b') for i in state))
        count_R = sum((i.count('r') for i in state))

        if count_B >= 4 and count_R >= 4:
            drop_phase = False

        if not drop_phase:
            # move is chosen by minimax algorithm
            move = self.minimax_decision(state)
            return move

        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        move = []
        (row, col) = (random.randint(0,4), random.randint(0,4))
        while not state[row][col] == ' ':
            (row, col) = (random.randint(0,4), random.randint(0,4))

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def succ(self, state):
        """ Successor function that takes in a board state and returns a list of the 
            legal successors. During the drop phase, this simply means adding a new 
            piece of the current player's type to the board; during continued gameplay, 
            this means moving any one of the current player's pieces to an unoccupied
            location on the board, adjacent to that piece.

        Args:
            current board state

        Returns:
            list of the legal successors     
        """
        successors = []

        # during the drop phase, add a new piece to the board
        if len([cell for row in state for cell in row if cell != ' ']) < 8:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        successor_state = [row.copy() for row in state]
                        successor_state[i][j] = self.my_piece
                        successors.append(successor_state)
        else:
            # during continued gameplay, move a piece to an adjacent unoccupied location
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        # Check adjacent positions
                        adjacent_positions = [
                            (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
                        ]
                        for new_i, new_j in adjacent_positions:
                            if 0 <= new_i < 5 and 0 <= new_j < 5 and state[new_i][new_j] == ' ':
                                successor_state = [row.copy() for row in state]
                                successor_state[i][j] = ' '
                                successor_state[new_i][new_j] = self.my_piece
                                successors.append(successor_state)

        return successors

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for i in range(2):
            if state[i][i] != ' ' and state[i][i] == state[i+1][i+1] == state[i+2][i+2] == state[i+3][i+3]:
                return 1 if state[i][i] == self.my_piece else -1

        # check / diagonal wins
        for i in range(2):
            if state[i][4-i] != ' ' and state[i][4-i] == state[i+1][3-i] == state[i+2][2-i] == state[i+3][1-i]:
                return 1 if state[i][4-i] == self.my_piece else -1

        # check box wins
        for i in range(4):
            for j in range(4):
                if state[i][j] != ' ' and state[i][j] == state[i][j+1] == state[i+1][j] == state[i+1][j+1]:
                    return 1 if state[i][j] == self.my_piece else -1

        return 0 # no winner yet
    
    def heuristic_game_value(self, state):
        """ Evaluates non-terminal states. For some hints, check out the 
            Games II Lecture Slides (you should call the game_value method from this 
            function to determine whether the state is a terminal state before you start 
            evaluating it heuristically.) This function should return some floating-point 
            value between 1 and -1.
        """ 
        terminal_value = self.game_value(state)

        # determine if the state is a terminal state
        if terminal_value != 0:
            return terminal_value
        
        
        # Calculate player's and opponent's progress towards forming 4 in a row or a box
        my_progress = self.calculate_progress(state, self.my_piece)
        opp_progress = self.calculate_progress(state, self.opp)

        # Normalize the progress values to be between 0 and 1
        my_normalized_progress = my_progress / 4.0
        opp_normalized_progress = opp_progress / 4.0

        # Calculate the net heuristic value based on the difference in progress
        heuristic_value = my_normalized_progress - opp_normalized_progress

        return heuristic_value
    
    def calculate_progress(self, state, player_piece):
        progress = 0

        # check horizontal progress
        for row in state:
            for i in range(2):
                if row[i] == row[i + 1] == row[i + 2] == player_piece:
                    progress += 1

        # check vertical progress
        for col in range(5):
            for i in range(2):
                if state[i][col] == state[i + 1][col] == state[i + 2][col] == player_piece:
                    progress += 1

        # check \ diagonal progress
        for i in range(2):
            if state[i][i] == state[i + 1][i + 1] == state[i + 2][i + 2] == player_piece:
                progress += 1

        # check / diagonal progress
        for i in range(2):
            if state[i][4 - i] == state[i + 1][3 - i] == state[i + 2][2 - i] == player_piece:
                progress += 1

        # check for box progress
        for i in range(4):
            for j in range(4):
                if state[i][j] == state[i][j + 1] == state[i + 1][j] == state[i + 1][j + 1] == player_piece:
                    progress += 1

        return progress

    def max_value(self, state, depth):
        terminal_value = self.game_value(state)

        if depth == 0 or terminal_value != 0:
            return self.heuristic_game_value(state)

        max_score = float('-inf')
        successors = self.succ(state)

        for successor in successors:
            max_score = max(max_score, self.min_value(successor, depth - 1))

        return max_score

    def min_value(self, state, depth):
        terminal_value = self.game_value(state)

        if depth == 0 or terminal_value != 0:
            return self.heuristic_game_value(state)

        min_score = float('inf')
        successors = self.succ(state)

        for successor in successors:
            min_score = min(min_score, self.max_value(successor, depth - 1))

        return min_score

    def minimax_decision(self, state):
        successors = self.succ(state)
        best_move = None
        best_score = float('-inf')

        for successor in successors:
            score = self.min_value(successor, depth=3)  # set depth to 3, checks next 3 moves
            if score > best_score:
                best_score = score
                best_move = successor[0]

        return [best_move]
############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
