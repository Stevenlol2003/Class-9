import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0

    starting_state = [
        [from_state[0], from_state[1], from_state[2]],
        [from_state[3], from_state[4], from_state[5]],
        [from_state[6], from_state[7], from_state[8]]
    ]

    # in the example provided top left position is (0, 0), 
    # so a coordinate map would look like:
    # (0, 0) (0, 1) (0, 2)
    # (1, 0) (1, 1) (1, 2)
    # (2, 0) (2, 1) (2, 2)

    final_state = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 0: (2, 1), 0: (2, 2)
    }
    
    # example where the number 1 tile starts top right:
    # (manhattan([0,2], [0,0]) = abs(0-0) + abs(2-0) = 2), 
    for i in range(3):
        for j in range(3):
            start = starting_state[i][j]
            if start != 0:
                final = final_state[start]
                distance += abs(i - final[0]) + abs(j - final[1])

    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))

def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle 
        (don't forget to sort the result as done below). 
    """
    succ_states = []

    # valid moves: up, down, left, right
    moves = [(-1, 0), (1,0), (0, -1), (0, 1)]

    for i in range(9):
        # skip if tile is 0
        if state[i] == 0:
            continue
        
        # calculate row and column number
        # row is quotient, col is remainder
        # this should provide a coordiante to all 9 points
        row = i // 3
        col = i % 3

        # (0, 0) (0, 1) (0, 2)
        # (1, 0) (1, 1) (1, 2)
        # (2, 0) (2, 1) (2, 2)

        # check all possible moves for the given tile, if it's valod
        for move in moves:
            newRow = row + move[0]
            newCol = col + move[1]
            
            # check bounds, if true then valid move
            if 0 <= newRow <= 2 and 0 <= newCol <= 2:
                # check if tile to move to is 0
                if state[i] != 0 and state[newRow * 3 + newCol] == 0:
                    successor = state.copy()

                    # swap current tile with empty tile
                    successor[row * 3 + col], successor[newRow * 3 + newCol] = successor[newRow * 3 + newCol], successor[row * 3 + col]

                    succ_states.append(successor)

    return sorted(succ_states)
    

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    # intialize pq and store starting state
    # cost, state, (g, h, parent_index)
    # parent_index is the number of moves to reach that state
    pq = [(0, state, (0, get_manhattan_distance(state), -1))]
    # store visited states
    visited = set() 

    # store information for each state: g, h, parent
    state_info = {tuple(state): (0, get_manhattan_distance(state), None)}
    # maximum queue length
    max_length = 0

    while pq:
        # Remove and get the state with the minimum f(n)
        #print(*pq, sep='\n')

        current_f, current_state, current_info = heapq.heappop(pq)
        #print(current_state)
        visited.add(tuple(current_state))

        # check if goal state is reached
        if current_state == goal_state:
            # back trace to get path
            state_info_list = []
            while current_state:
                state_info_list.append(current_state)
                current_state, current_info = state_info[tuple(current_state)][2], state_info[tuple(current_state)]
            state_info_list.reverse()

            # print path
            for state in state_info_list:
                h = get_manhattan_distance(state, goal_state)
                move = state_info[tuple(state)][0]
                print(state, "h={}".format(h), "moves: {}".format(move))
            #print(state_info_list)
            print("Max queue length: {}".format(max_length))
            return

        # find successors and update pq
        successors = get_succ(current_state)
        for successor in successors:
            successor_tuple = tuple(successor)

            # f(n) = cost, g(n) = moves, h(n) = manhattan distance
            g = current_info[0] + 1  # cost of 1 for each move
            h = get_manhattan_distance(successor, goal_state)
            cost = g + h

            if successor_tuple not in visited:
                if (cost, successor, (g, h, current_state)) not in pq:
                    # add successor to pq
                    heapq.heappush(pq, (cost, successor, (g, h, current_state)))
                    # update successor information
                    state_info[successor_tuple] = (g, h, current_state)
                elif cost < state_info[successor_tuple][0]:
                    # update successor information if a better path is found
                    state_info[successor_tuple] = (g, h, current_state)    

        # update maximum queue length
        max_length = max(max_length, len(pq))

    # if pq is empty and no solution is found, 
    # this shouldn't happen in a 7-tile puzzle,
    # but i know that it could happen in a 8-tile puzzle, so just incase
    print("No valid solution")


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    #print_succ([2,5,1,4,0,6,7,0,3])
    #print()

    #print(get_manhattan_distance([2, 5 ,1 ,4 ,3 ,6 ,7 ,0 ,0], [1, 2, 3, 4, 5, 6, 7, 0, 0])) 
    #print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    #print()

    #solve([2,5,1,4,0,6,7,0,3])
    #print()

    solve([4,3,0,5,1,6,7,2,0])
    print()

    #solve([3, 4, 6, 0, 0, 1, 7, 2, 5])
    #print()
    #solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    #print()
    #solve([0, 4, 7, 1, 3, 0, 6, 2, 5])