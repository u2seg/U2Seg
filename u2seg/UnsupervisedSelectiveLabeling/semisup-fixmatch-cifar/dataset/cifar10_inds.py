import os
import numpy as np

def get_labeled_inds_random(args, labels, get_labeled_inds_method):
    print("Get random labels")
    # selected_inds = np.random.choice(np.arange(len(labels)), size=args.num_labeled, replace=False)
    selected_inds = np.load(os.path.join("indices/cifar10", get_labeled_inds_method + ".npy"))

    selected_inds = np.array(selected_inds)

    print("Label distribution")
    print(np.unique(labels[selected_inds], return_counts=True))

    assert selected_inds.shape[0] == args.num_labeled

    return selected_inds

def get_labeled_inds_select(args, labels, get_labeled_inds_method):
    print("Use selected labels")
    selected_inds = np.load(os.path.join("indices/cifar10", get_labeled_inds_method + ".npy"))

    selected_inds = np.array(selected_inds)

    assert selected_inds.shape[0] == args.num_labeled

    print("Label distribution")
    print(np.unique(labels[selected_inds], return_counts=True))

    return selected_inds
