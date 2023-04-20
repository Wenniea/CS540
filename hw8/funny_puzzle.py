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
    for i in range(9):
        if from_state[i] != to_state[i]:
            if from_state[i] != 0:
                from_index = i
                to_index = to_state.index(from_state[i])
                from_row = from_index // 3
                from_col = from_index % 3
                to_row = to_index // 3
                to_col = to_index % 3
                distance += abs(from_row - to_row) + abs(from_col - to_col)
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
        INPUT:
            A state (list of length 9)
        RETURNS:
            A list of all the valid successors in the puzzle (don't forget to sort the result as done below).
    """
    def swap_positions(arr, pos1, pos2):
        arr_copy = arr.copy()
        arr_copy[pos1], arr_copy[pos2] = arr_copy[pos2], arr_copy[pos1]
        return arr_copy

    def is_adjacent(pos1, pos2):
        x1, y1 = pos1 % 3, pos1 // 3
        x2, y2 = pos2 % 3, pos2 // 3
        return abs(x1 - x2) + abs(y1 - y2) == 1

    succ_states = []
    zero_indices = [i for i in range(9) if state[i] == 0]

    for i in range(9):
        if state[i] != 0:
            for zero_index in zero_indices:
                if is_adjacent(i, zero_index):
                    new_state = swap_positions(state, i, zero_index)
                    succ_states.append(new_state)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    pq = []
    visited = set()
    g = 0
    h = get_manhattan_distance(state)

    heapq.heappush(pq, (g + h, state, (g, h, None)))

    while pq:
        # pop the state with the lowest cost value
        cost, curr_state, (g, h, parent) = heapq.heappop(pq)

        # if the state is the goal state, print the path
        if curr_state == goal_state:
            # create a list of the path
            path = []
            newPath = (cost, curr_state, (g, h, parent))
            while newPath[2][2] is not None:
                path.append(newPath)
                newPath = newPath[2][2]
            path.append(newPath)
            path.reverse()

            for i in range(len(path)):
                print(f"{path[i][1]} h={path[i][2][1]} moves: {path[i][2][0]}")
            print("Max queue length: {}".format(len(pq) + 1))
            return

        visited.add(tuple(curr_state))

        # get the successors of the state
        succ_states = get_succ(curr_state)

        # add the successors to the priority queue
        for succ_state in succ_states:
            if tuple(succ_state) not in visited:
                new_g = g + 1
                new_h = get_manhattan_distance(succ_state)
                new_cost = new_g + new_h
                heapq.heappush(pq, (
                    new_cost, succ_state, (new_g, new_h, (cost, curr_state, (g, h, parent)))))


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5])
    print()

    print(get_manhattan_distance([2, 5, 1, 4, 0, 6, 7, 0, 3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([4, 3, 0, 5, 1, 6, 7, 2, 0])
    print()
