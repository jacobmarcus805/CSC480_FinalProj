
from environment import HIT, STAND, DOUBLE

# Dealer upcard value for each column index (Ace = 11).
DEALER_UPCARDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

CHAR_TO_ACTION = {"H": HIT, "S": STAND, "D": DOUBLE}
ACTION_TO_CHAR = {HIT: "H", STAND: "S", DOUBLE: "D"}

# Hard totals (no usable ace), player total -> per-upcard action string.
HARD_STRATEGY = {
    5:  "HHHHHHHHHH",
    6:  "HHHHHHHHHH",
    7:  "HHHHHHHHHH",
    8:  "HHHHHHHHHH",
    9:  "HDDDDHHHHH",  # double vs 3-6
    10: "DDDDDDDDHH",  # double vs 2-9
    11: "DDDDDDDDDH",  # double vs 2-10
    12: "HHSSSHHHHH",  # stand vs 4-6
    13: "SSSSSHHHHH",  # stand vs 2-6
    14: "SSSSSHHHHH",
    15: "SSSSSHHHHH",
    16: "SSSSSHHHHH",
    17: "SSSSSSSSSS",
    18: "SSSSSSSSSS",
    19: "SSSSSSSSSS",
    20: "SSSSSSSSSS",
}

# Soft totals (one ace counted as 11), player total -> per-upcard action string.
SOFT_STRATEGY = {
    13: "HHHDDHHHHH",  # A,2 : double vs 5-6
    14: "HHHDDHHHHH",  # A,3 : double vs 5-6
    15: "HHDDDHHHHH",  # A,4 : double vs 4-6
    16: "HHDDDHHHHH",  # A,5 : double vs 4-6
    17: "HDDDDHHHHH",  # A,6 : double vs 3-6
    18: "SDDDDSSHHH",  # A,7 : double vs 3-6, stand vs 2/7/8, hit vs 9/10/A
    19: "SSSSSSSSSS",  # A,8 : stand
    20: "SSSSSSSSSS",  # A,9 : stand
}


def _upcard_index(upcard_val: int) -> int:
    """Map a dealer upcard value (2-11, Ace=11) to its column index."""
    return DEALER_UPCARDS.index(upcard_val)


def optimal_action(player_total: int, is_soft: bool, upcard_val: int) -> int:
    """Return the optimal action ID for a two-card decision.

    player_total : hand value (soft totals use the ace-as-11 value, e.g. A,7 = 18)
    is_soft      : True if a usable ace is present
    upcard_val   : dealer upcard value, Ace = 11
    """
    table = SOFT_STRATEGY if is_soft else HARD_STRATEGY
    row = table[player_total]
    return CHAR_TO_ACTION[row[_upcard_index(upcard_val)]]


def _self_test() -> None:
    """Validate table shapes and a few well-known cells."""
    for name, table in (("hard", HARD_STRATEGY), ("soft", SOFT_STRATEGY)):
        for total, row in table.items():
            assert len(row) == 10, f"{name} {total} has {len(row)} cols, expected 10"
            assert set(row) <= set("HSD"), f"{name} {total} has bad chars: {row}"

    assert optimal_action(16, False, 10) == HIT        # hard 16 vs 10 -> hit
    assert optimal_action(16, False, 6) == STAND       # hard 16 vs 6  -> stand
    assert optimal_action(11, False, 6) == DOUBLE      # hard 11 vs 6  -> double
    assert optimal_action(11, False, 11) == HIT        # hard 11 vs A  -> hit (S17)
    assert optimal_action(20, False, 2) == STAND       # hard 20       -> stand
    assert optimal_action(18, True, 9) == HIT          # soft 18 vs 9  -> hit
    assert optimal_action(18, True, 6) == DOUBLE       # soft 18 vs 6  -> double
    assert optimal_action(18, True, 2) == STAND        # soft 18 vs 2  -> stand
    assert optimal_action(13, True, 5) == DOUBLE       # soft 13 vs 5  -> double
    print("basic_strategy self-test passed.")


if __name__ == "__main__":
    _self_test()
