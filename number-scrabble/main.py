import math
from copy import deepcopy
from functools import reduce


MAX_PLAYER_ID = 1
MIN_PLAYER_ID = 2


def get_game_from_input():
    player1_moves = []
    player2_moves = []

    with open("input.txt") as file:
        for line in file:
            split_line = line.replace('\n', '').split(':')
            if split_line[0] == 'A':
                player1_moves.append(split_line[1])
            elif split_line[0] == 'B':
                player2_moves.append(split_line[1])
            else:
                raise Exception("Only 2 players allowed that are called 'A' or 'B'!")

    return player1_moves, player2_moves


def get_test_game_board():
    return [
        [2, 7, 6],
        [9, 5, 1],
        [4, 3, 8]
    ]


def get_init_game_board():
    return [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]


def get_number_index_in_game_board(number):
    assert 1 <= number <= 9, "Number chosen should be bigger or equal to 1 and smaller or equal to 9"
    return [(ix,iy) for ix, row in enumerate(get_test_game_board()) for iy, i in enumerate(row) if i == number][0]


def get_available_numbers_to_choose(state):
    test_game_board = get_test_game_board()
    return [test_game_board[ix][iy] for ix, row in enumerate(state) for iy, i in enumerate(row) if i == 0]


def convert_matrix_to_list(matrix_to_convert):
    return reduce(lambda accumulator, current_val: [*accumulator, *current_val], matrix_to_convert, [])


def set_slot_in_game_board(state, number, player_id):
    number_coordinates = get_number_index_in_game_board(number)
    x = number_coordinates[0]
    y = number_coordinates[1]

    assert state[x][y] == 0, "Slot already selected"

    new_state = deepcopy(state)
    new_state[x][y] = player_id

    return new_state


def get_transpose(state):
    return list(map(list, zip(*state)))


def get_main_diagonal(state):
    return [state[x][y] for x in range(len(state)) for y in range(len(state[x])) if x == y]


def get_secondary_diagonal(state):
    return [state[x][y] for x in range(len(state)) for y in range(len(state[x])) if x + y == len(state) - 1]


def power_10(lst, number):
    number_of_appearances = lst.count(number)

    if number_of_appearances == 0:
        return 0

    score = 10 ** (number_of_appearances - 1)
    return score


def get_lines(state):
    return [*state, *get_transpose(state), get_main_diagonal(state), get_secondary_diagonal(state)]


def power_10_total(state):
    lines = get_lines(state)
    result = 0

    for line in lines:
        result += power_10(line, MAX_PLAYER_ID)
        #result -= power_10(line, MIN_PLAYER_ID)

    return result


def validate_player_is_winning(state, player_id):
    lines = get_lines(state)
    for line in lines:
        if line.count(player_id) == len(line):
            return True

    return False


def is_state_final(state):
    if validate_player_is_winning(state, MIN_PLAYER_ID) or validate_player_is_winning(state, MAX_PLAYER_ID):
        return True

    for row in state:
        if row.count(0) > 0:
            return False

    return True


def is_max_turn(state):
    max_moves = 0
    min_moves = 0

    for row in state:
        max_moves += row.count(MAX_PLAYER_ID)
        min_moves += row.count(MIN_PLAYER_ID)

    assert max_moves - min_moves == 0 or max_moves - min_moves == 1, "Invalid state"

    return max_moves == min_moves


def minimax(state, depth):
    if depth == 0 or is_state_final(state):
        return power_10_total(state), 0

    if is_max_turn(state):
        max_evaluation = -math.inf
        number = 0

        for number_to_choose in get_available_numbers_to_choose(state):
            new_state = set_slot_in_game_board(state, number_to_choose, MAX_PLAYER_ID)
            eva, _ = minimax(new_state, depth - 1)
            if eva > max_evaluation:
                max_evaluation = eva
                number = number_to_choose

        return max_evaluation, number

    else:
        min_evaluation = math.inf
        number = 0

        for number_to_choose in get_available_numbers_to_choose(state):
            new_state = set_slot_in_game_board(state, number_to_choose, MIN_PLAYER_ID)
            eva, _ = minimax(new_state, depth - 1)
            if eva < min_evaluation:
                min_evaluation = eva
                number = number_to_choose

        return min_evaluation, number


def get_number_from_console(allowed_numbers):
    while True:
        number = int(input("\nAllowed numbers are: " + str(sorted(allowed_numbers)) + "\n    Enter your number: "))
        if number in allowed_numbers:
            return number


def main():
    state = get_init_game_board()

    while True:
        max_player_number = get_number_from_console(get_available_numbers_to_choose(state))
        state = set_slot_in_game_board(state, max_player_number, MAX_PLAYER_ID)

        if is_state_final(state):
            break

        _, min_player_number = minimax(state, 4)
        state = set_slot_in_game_board(state, min_player_number, MIN_PLAYER_ID)

        print("\nComputer has chosen the number " + str(min_player_number))

        if is_state_final(state):
            break

    print("\nGame over\n")

    if validate_player_is_winning(state, MAX_PLAYER_ID):
        print("You won!!! Congrats")
    elif validate_player_is_winning(state, MIN_PLAYER_ID):
        print("Computer won :(")
    else:
        print("Tie")



if __name__ == "__main__":
    main()
