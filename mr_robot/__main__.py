import argparse

from mr_robot.mr_robot import MrRobot, strategies


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file", type=str, required=True)
parser.add_argument(
    "-s", "--strategy", help="Annotation strategy to use", type=str, choices=strategies.keys(), required=True
)


def main():
    args = parser.parse_args()
    MrRobot(args.config, args.strategy)
    return 0


if __name__ == "__main__":
    exit(main())
