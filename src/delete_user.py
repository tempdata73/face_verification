import os
import sys
import argparse
import numpy as np
import pandas as pd


def delete_user(args):
    # Load data
    user_data = pd.read_csv(args.user_data)
    embeds = np.load(args.embeddings)
    # Get index of username and delete it from database
    idx = user_data[user_data.name == args.username].index
    if len(idx) == 0:
        print("{} is not in database".format(args.username))
        sys.exit(0)
    user_data = user_data.drop(idx)
    embeds = np.delete(embeds, idx, axis=0)
    # Save data
    user_data.to_csv(args.user_data, index=None)
    np.save(os.path.splitext(args.embeddings)[0], embeds)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("user_data", type=str,
                        help="file containing user metadata")
    parser.add_argument("embeddings", type=str,
                        help="file containing user embeddings")
    parser.add_argument("username", type=str,
                        help="user to delete from database")
    return parser.parse_args(argv)


if __name__ == "__main__":
    delete_user(parse_arguments(sys.argv[1:]))
