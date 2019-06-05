import argparse
from pathlib import Path

from stratego import Game
from stratego.deep_q_agent import DeepQAgent, compare_deep_q_head_to_head, \
    compare_deep_q_versus_random
# noinspection PyUnresolvedReferences
from stratego.human_agent import HumanAgent
from stratego.printer import Printer
# noinspection PyUnresolvedReferences
from stratego.random_agent import RandomAgent
from stratego.utils import setup_logger, EXPORT_DIR


# ToDo Add argparse
def parse_args():
    r""" Parse the command line input arguments """
    args = argparse.ArgumentParser()
    args.add_argument("-human", help="Play against a deep Q agent", default=False,
                      action='store_true')
    args.add_argument("-train", help="Perform training", default=False, action='store_true')
    args.add_argument("-random", help="Play two random agents", default=False, action='store_true')
    args.add_argument("-base", help="Play a random agent against the Deep Q network",
                      default=False, action='store_true')
    # args.add_argument("-compare", help="", default=False, action='store_true')
    return args.parse_args()


def _main_random():
    r""" Player two random agents """
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    a1, a2 = RandomAgent(game.red), RandomAgent(game.blue)
    # a2 = RandomAgent(game.blue)
    # a1 = HumanAgent(game.red)
    game.two_agent_automated(a1, a2, display=True, wait_time=1)


def _main_deep_q_vs_human():
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    a1 = HumanAgent(game.blue)
    a2 = DeepQAgent(game.board, game.red, game.state)
    game.two_agent_automated(a1, a2, display=True, move_f_out="moves.txt")


def _main_deep_q_vs_rand(num_games: int = 501):
    def _default_name(i: int) -> Path:
        return EXPORT_DIR / ("_checkpoint_lr=1e-%d_wd=0.pth" % i)

    # Test against a random agent
    all_paths = [EXPORT_DIR / "_checkpoint_initial.pth"] + [_default_name(i) for i in range(2, 6)]
    for file in all_paths:
        compare_deep_q_versus_random("boards/small.txt", "states/test_debug.txt", num_games, file)

    # Test against a Deep-Q agent with random weights.
    for file in all_paths[1:]:
        compare_deep_q_head_to_head("boards/small.txt", "states/test_debug.txt", num_games, file,
                                    EXPORT_DIR / "_checkpoint_initial.pth" )


# def _compare_two_deep_q():
#     compare_deep_q_head_to_head("boards/small.txt", "states/test_debug.txt", 1001,
#                                 EXPORT_DIR / "_checkpoint_lr=1e-5_wd=0_epoch=260_no_checkpoint.pth",
#                                 EXPORT_DIR / "_checkpoint_lr=1e-5_wd=0_no_checkpoint.pth")


def _main_train():
    game = Game("boards/small.txt", "states/test_debug.txt", visibility=Printer.Visibility.ALL)
    agent = DeepQAgent(game.board, game.red, game.state)
    agent.train_network(game.state)


def _main():
    args = parse_args()

    if args.human:
        _main_deep_q_vs_human()
    elif args.train:
        _main_train()
    elif args.random:
        _main_random()
    elif args.base:
        _main_deep_q_vs_rand()
    # elif args.compare:
    #     _compare_two_deep_q()


if __name__ == "__main__":
    setup_logger()
    _main()
