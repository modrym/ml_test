import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml


def load_data(cache=True, save=True):
    mnist_dir = os.path.join(os.getcwd(), 'data', 'mnist')
    os.makedirs(mnist_dir, exist_ok=True)
    mnist_file = os.path.join(mnist_dir, 'mnist.pkl')

    if cache and os.path.isfile(mnist_file):
        mnist = joblib.load(mnist_file)
    else:
        mnist = fetch_openml('mnist_784', version=1)

        if save:
            if os.path.exists(mnist_file):
                os.remove(mnist_file)
            joblib.dump(mnist, mnist_file)

    return mnist['data'], mnist['target'].astype(np.uint8)


def show_image(arr, reshape=(28, 28), ax=None):
    arr = np.array(arr)

    if reshape:
        arr = arr.reshape(reshape)

    if ax is None:
        fig, ax = plt.subplots()
        ax.imshow(arr, cmap='binary')
        ax.axis('off')
        fig.show()
    else:
        ax.imshow(arr, cmap='binary')
        ax.axis('off')


def show_grid(arrs, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols)

    if nrows * ncols == 1:
        axes = np.array([axes])

    for ax, arr in zip(axes.reshape(nrows * ncols), arrs[:(nrows * ncols)]):
        show_image(arr, ax=ax)

    fig.show()


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cache', '-c', action='store_true',
                        help='If specified, do not use the cache file and '
                             'download the data')
    parser.add_argument('--disable-save', '-s', action='store_true',
                        help='If specified, do not save fetched data')

    subp = parser.add_subparsers(dest='action', required=True)

    p = subp.add_parser('show-grid', help='Show grid of numbers')
    p.add_argument('nrows', type=int, help='Number of rows')
    p.add_argument('ncols', type=int, help='Number of columns')
    p.add_argument('start_from', type=int, default=0,
                   help='Number of image to start from (counting from 0)')

    p = subp.add_parser('show', help='Show a particular image')
    p.add_argument('number', help='Number of image (counting from 0)',
                   type=int)

    p = subp.add_parser('run', help='Train and test')
    p.add_argument('--classifier', '-c', type=str,
                   help='Number of image (counting from 0)')

    return parser


def main():
    parser = setup_argparser()
    parsed = parser.parse_args()

    X, y = load_data(cache=not parsed.disable_cache,
                     save=not parsed.disable_save)

    if parsed.action == 'show-grid':
        sf = parsed.start_from
        nr = parsed.nrows
        nc = parsed.ncols
        show_grid(np.array(X.loc[sf:sf + nr * nc]), nr, nc)
    elif parsed.action == 'show':
        show_image(X.loc[parsed.number])
    else:
        # TODO
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
