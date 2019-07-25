import argparse

from mr_robot.mr_robot import MrRobot, strategies


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to config file", type=str, required=True)
parser.add_argument(
    "-s", "--strategy", help="Annotation strategy to use", type=str, choices=strategies.keys(), required=True
)
parser.add_argument("-d", "--device", help="Which device to use", type=str, required=True, default="cpu")


def main():
    args = parser.parse_args()
    robo = MrRobot(args.config, args.strategy, args.device)
    robo._load_model()
    robo._run()
    return 0


if __name__ == "__main__":
    exit(main())
