import csv
import math
import time

import numpy as np
from functools import reduce
from copy import deepcopy


def read_input_file(input_file_name):
    initial_states = []

    with open(input_file_name) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            initial_states.append([int(x) for x in row])

    return initial_states


def convert_list_to_matrix(number_columns, list_to_convert):
    if len(list_to_convert) % number_columns != 0:
        raise Exception("Length of list should be multiple of number of columns")

    if len(list_to_convert) <= number_columns:
        return [list_to_convert]
    else:
        return [list_to_convert[:number_columns]] + convert_list_to_matrix(number_columns,
                                                                           list_to_convert[number_columns:])


def convert_matrix_to_list(matrix_to_convert):
    return reduce(lambda accumulator, current_val: [*accumulator, *current_val], matrix_to_convert, [])


def get_final_state(state):
    number_of_cells = len(convert_matrix_to_list(state))
    return [x + 1 for x in list(range(number_of_cells - 1))]  # Because we also have 0 in our matrix


def is_state_final(state):
    state_as_list = convert_matrix_to_list(state)
    state_as_list.remove(0)
    return state_as_list == get_final_state(state)


def get_empty_cell_coordinates(state):
    val = 0
    indexes_array = [(index, row.index(val)) for index, row in enumerate(state) if val in row]

    if len(indexes_array) != 1:
        raise Exception("One and only one empty cell should be present in the data-set")

    return indexes_array[0]


def is_transition_valid(state, transition):
    assert len(transition) == 2, "Transition array should contain two elements"
    empty_cell_coordinates = get_empty_cell_coordinates(state)
    empty_cell_after_transition_coordinates = np.add(list(empty_cell_coordinates), transition)

    state_rows = len(state)
    state_columns = len(state[0])

    empty_cell_row_after_transition = empty_cell_after_transition_coordinates[0]
    empty_cell_column_after_transition = empty_cell_after_transition_coordinates[1]

    if 0 <= empty_cell_row_after_transition < state_rows and 0 <= empty_cell_column_after_transition < state_columns:
        return True
    else:
        return False


def get_new_state(state, transition):
    assert len(transition) == 2, "Transition array should contain two elements"

    if not is_transition_valid(state, transition):
        return None

    empty_cell_coordinates = list(get_empty_cell_coordinates(state))
    empty_cell_row = empty_cell_coordinates[0]
    empty_cell_column = empty_cell_coordinates[1]

    transition_neighbour_value = state[empty_cell_row + transition[0]][empty_cell_column + transition[1]]
    new_state = deepcopy(state)
    new_state[empty_cell_row][empty_cell_column] = transition_neighbour_value
    new_state[empty_cell_row + transition[0]][empty_cell_column + transition[1]] = 0

    return new_state


def get_transitions():
    return [
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0]
    ]


def DFS(initial_state, max_depth):
    transitions_stack = [(initial_state, [])]
    visited = set()

    while len(transitions_stack) > 0:
        state, path = transitions_stack.pop()

        if is_state_final(state):
            return [state, path]

        visited.add(str(state))

        if len(path) < max_depth:
            for transition in get_transitions():
                # We don't go back in the last state
                if len(path) > 0 and transition == [x * -1 for x in path[-1]]:
                    continue
                new_state = get_new_state(state, transition)
                if new_state is not None:
                    if str(new_state) not in visited:
                        new_path = path + [transition]
                        transitions_stack.append((new_state, new_path))

    return None


def IDDFS(state, max_depth):
    for i in range(max_depth):
        result = DFS(state, i)

        if result is not None:
            return result

    return None


def get_manhattan_distance(coordinates1, coordinates2):
    if len(coordinates1) != 2 and len(coordinates2) != 2:
        raise Exception("Coordinates should be of type [row_number, column_number]")

    row_distance = abs(coordinates1[0] - coordinates2[0])
    column_distance = abs(coordinates1[1] - coordinates2[1])

    return row_distance + column_distance


def get_possible_positions_in_final_state_for_value(value, state_shape):
    if len(state_shape) != 2:
        raise Exception("State shape should be of type [number_of_rows, number_of_columns]")

    number_of_rows = state_shape[0]
    number_of_columns = state_shape[1]

    if value >= number_of_rows * number_of_columns or value <= 0:
        raise Exception(f"Value should be less than {number_of_rows * number_of_columns} and bigger than 0")

    #   Position of the slot with the value passed          #
    #       as an argument if the empty slot is after it.   #
    position1 = [(value - 1) // number_of_columns, (value - 1) % number_of_columns]
    #   Position of the slot with the value passed          #
    #       as an argument if the empty slot is before it.  #
    position2 = [value // number_of_columns, value % number_of_columns]

    return [position1, position2]


def get_state_total_manhattan_distance(state):
    total = 0
    state_shape = [len(state), len(state[0])]

    for row_number in range(len(state)):
        for column_number in range(len(state[row_number])):
            state_cell_value = state[row_number][column_number]
            if state_cell_value == 0:
                continue
            possible_positions = get_possible_positions_in_final_state_for_value(state_cell_value, state_shape)
            total += (get_manhattan_distance(possible_positions[0], [row_number, column_number]) +
                      get_manhattan_distance(possible_positions[1], [row_number, column_number])) / 2

    return total


def get_state_hamming_distance(state):
    state_as_list = convert_matrix_to_list(state)
    hamming_distance = 0
    counter = 0

    for value in state_as_list:
        if value == 0:
            continue

        counter += 1

        if counter != value:
            hamming_distance += 1

    return hamming_distance


def min_swap(state):
    state_as_list = convert_matrix_to_list(state)
    state_as_list.remove(0)

    n = len(state_as_list)

    # Create two arrays and use
    # as pairs where first array
    # is element and second array
    # is position of first element
    arrpos = [*enumerate(state_as_list)]

    # Sort the array by array element
    # values to get right position of
    # every element as the elements
    # of second array.
    arrpos.sort(key=lambda it: it[1])

    # To keep track of visited elements.
    # Initialize all elements as not
    # visited or false.
    vis = {k: False for k in range(n)}

    # Initialize result
    ans = 0
    for i in range(n):

        # already swapped or
        # already present at
        # correct position
        if vis[i] or arrpos[i][0] == i:
            continue

        # find number of nodes
        # in this cycle and
        # add it to ans
        cycle_size = 0
        j = i

        while not vis[j]:

            # mark node as visited
            vis[j] = True

            # move to next node
            j = arrpos[j][0]
            cycle_size += 1

        # update answer by adding
        # current cycle
        if cycle_size > 0:
            ans += (cycle_size - 1)

            # return answer
    return ans



def greedy(state, heuristic):
    number_of_moves = 0
    visited = []
    transitions = [[0, 0]]
    while True:
        if is_state_final(state):
            break

        visited.append(str(state))

        minimum_score = math.inf
        best_state = None
        best_transition = []

        for transition in get_transitions():
            if is_transition_valid(state, transition) and str(get_new_state(state, transition)) not in visited:
                new_state = get_new_state(state, transition)
                state_score = heuristic(new_state)
                if state_score < minimum_score:
                    minimum_score = state_score
                    best_state = new_state
                    best_transition = transition

        if best_state is None:
            for transition in get_transitions():
                if transition != [x * -1 for x in transitions[-1]] and is_transition_valid(state, transition):
                    new_state = get_new_state(state, transition)
                    state_score = heuristic(new_state)
                    if state_score < minimum_score:
                        minimum_score = state_score
                        best_state = new_state
                        best_transition = transition

        transitions.append(best_transition)
        state = best_state
        number_of_moves += 1

    return [state, transitions]


def write_to_file(file_name, content):
    f = open(file_name, "w")
    f.write(content)
    f.close()


def print_search_result(result, algorithm_time, text, file_name):
    final_state = result[0]
    transitions_array = result[1]
    transitions_number = len(transitions_array)

    write_to_file(file_name, str(transitions_array))
    print(text.upper())
    print(f"Algorithm took {algorithm_time} seconds to execute.")
    print(f"Final state found after {transitions_number} moves.")
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in final_state]))
    print()
    print()


def benchmark(initial_states):
    counter = 0
    for state in initial_states:
        counter += 1

        print("BENCHMARK FOR DATASET " + str(counter) + ":")

        start_IDDFS = time.time()
        IDDFS_output = IDDFS(state, 60)
        end_IDDFS = time.time()

        start_greedy_manhattan = time.time()
        greedy_manhattan_output = greedy(state, get_state_total_manhattan_distance)
        end_greedy_manhattan = time.time()

        start_greedy_hamming = time.time()
        greedy_hamming_output = greedy(state, get_state_hamming_distance)
        end_greedy_hamming = time.time()

        start_greedy_min_swaps = time.time()
        greedy_min_swaps_output = greedy(state, min_swap)
        end_greedy_min_swaps = time.time()

        print_search_result(IDDFS_output, end_IDDFS - start_IDDFS, "Benchmark for IDDFS:\n", "IDDFS_" + str(counter))
        print_search_result(greedy_manhattan_output, end_greedy_manhattan - start_greedy_manhattan, "Benchmark for Greedy Manhattan:\n", "greedy_manhattan_" + str(counter))
        print_search_result(greedy_hamming_output, end_greedy_hamming - start_greedy_hamming, "Benchmark for Greedy Hamming:\n", "greedy_hamming_" + str(counter))
        print_search_result(greedy_min_swaps_output, end_greedy_min_swaps - start_greedy_min_swaps, "Benchmark for Greedy Minimum Swaps:\n", "greedy_min_swap" + str(counter))

        print(".........................................................................")



def main():
    input_file_name = "input-matrix.csv"
    number_of_columns = 3

    initial_states_as_list = read_input_file(input_file_name)
    initial_states_as_matrix = [
        convert_list_to_matrix(number_of_columns, state_as_list)
        for state_as_list in initial_states_as_list
    ]

    benchmark(initial_states_as_matrix)


if __name__ == "__main__":
    main()
