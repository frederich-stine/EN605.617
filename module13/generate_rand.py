import argparse
import random
import time

def main():

    fh = open(args.file, 'w')

    fh.write(f"{args.width}\n")
    fh.write(f"{args.count}\n")

    random.seed(time.time())
    for i in range(args.count):
        for x in range(args.width):
            fh.write(f"{random.uniform(0.0, 10.0)},")
        fh.write("\n")

    fh.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="Rand Generator",
        description="Program creates random values for OpenCL assignment"
    )

    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-c", "--count", type=int, required=True)
    parser.add_argument("-w", "--width", type=int, required=True)

    args = parser.parse_args()

    main()