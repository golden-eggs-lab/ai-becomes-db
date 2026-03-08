import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# TensorFlow 2.x compatible imports
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD

import numpy as np
import time
import util_nearpy as util
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='MNIST CRAIG Training')
parser.add_argument('--paper_settings', action='store_true', 
                    help='Use CRAIG paper settings (batch=10, subset=0.5, lr=0.01, normalize=True). '
                         'Default: use original repo settings (batch=32, subset=0.4, lr=default, normalize=False)')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Batch size (default: 32 for repo, 10 for paper)')
parser.add_argument('--subset_size', type=float, default=None,
                    help='Subset size ratio (default: 0.4 for repo, 0.5 for paper)')
parser.add_argument('--lr', type=float, default=None,
                    help='Learning rate (default: keras default for repo, 0.01 for paper)')
parser.add_argument('--normalize', action='store_true', default=None,
                    help='Normalize data to [0,1] by dividing by 255')
parser.add_argument('--epochs', type=int, default=15,
                    help='Number of epochs (default: 15)')
parser.add_argument('--runs', type=int, default=5,
                    help='Number of runs (default: 5)')
parser.add_argument('--subset', action='store_true', default=True,
                    help='Use subset selection (default: True)')
parser.add_argument('--random', action='store_true', default=False,
                    help='Use random subset instead of CRAIG (default: False)')
args = parser.parse_args()

# Determine settings based on --paper_settings flag
if args.paper_settings:
    # CRAIG paper settings
    batch_size = args.batch_size if args.batch_size is not None else 10
    subset_size = args.subset_size if args.subset_size is not None else 0.5
    learning_rate = args.lr if args.lr is not None else 0.01
    normalize = True if args.normalize is None else args.normalize
    print("Using CRAIG paper settings")
else:
    # Original repo settings (default)
    batch_size = args.batch_size if args.batch_size is not None else 32
    subset_size = args.subset_size if args.subset_size is not None else 0.4
    learning_rate = args.lr if args.lr is not None else None  # Use Keras default
    normalize = args.normalize if args.normalize is not None else False
    print("Using original repo settings")

print(f"Settings: batch_size={batch_size}, subset_size={subset_size}, "
      f"lr={learning_rate if learning_rate else 'default'}, normalize={normalize}")

# Load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape and optionally normalize
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
if normalize:
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    print("Data normalized to [0,1]")

num_classes, smtk = 10, 0
Y_train_nocat = Y_train
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

subset = args.subset
random = args.random
subset_size = subset_size if subset else 1.0
epochs = args.epochs
reg = 1e-4
runs = args.runs
save_subset = False

folder = f'/tmp/mnist'

# Build model
model = Sequential()
model.add(Dense(100, input_dim=784, kernel_regularizer=l2(reg)))
model.add(Activation('sigmoid'))
model.add(Dense(10, kernel_regularizer=l2(reg)))
model.add(Activation('softmax'))

# Compile with appropriate optimizer
if learning_rate is not None:
    optimizer = SGD(learning_rate=learning_rate)
    print(f"Using SGD with lr={learning_rate}")
else:
    optimizer = 'sgd'
    print("Using SGD with default learning rate")
    
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)


train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_time = np.zeros((runs, epochs))
grd_time, sim_time, pred_time = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
not_selected = np.zeros((runs, epochs))
times_selected = np.zeros((runs, len(X_train)))
best_acc = 0
print(f'----------- smtk: {smtk} ------------')

if save_subset:
    B = int(subset_size * len(X_train))
    selected_ndx = np.zeros((runs, epochs, B))
    selected_wgt = np.zeros((runs, epochs, B))

for run in range(runs):
    X_subset = X_train
    Y_subset = Y_train
    W_subset = np.ones(len(X_subset))
    ordering_time,similarity_time, pre_time = 0, 0, 0
    loss_vec, acc_vec, time_vec = [], [], []
    for epoch in range(0, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        num_batches = int(np.ceil(X_subset.shape[0] / float(batch_size)))

        for index in range(num_batches):
            X_batch = X_subset[index * batch_size:(index + 1) * batch_size]
            Y_batch = Y_subset[index * batch_size:(index + 1) * batch_size]
            W_batch = W_subset[index * batch_size:(index + 1) * batch_size]

            start = time.time()
            history = model.train_on_batch(X_batch, Y_batch, sample_weight=W_batch)
            train_time[run][epoch] += time.time() - start

        if subset:
            if random:
                # indices = np.random.randint(0, len(X_train), int(subset_size * len(X_train)))
                indices = np.arange(0, len(X_train))
                np.random.shuffle(indices)
                indices = indices[:int(subset_size * len(X_train))]
                W_subset = np.ones(len(indices))
            else:
                start = time.time()
                _logits = model.predict_proba(X_train)
                pre_time = time.time() - start
                features = _logits - Y_train

                indices, W_subset, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                    int(subset_size * len(X_train)), features, 'euclidean', smtk, 0, False, Y_train_nocat)

                W_subset = W_subset / np.sum(W_subset) * len(W_subset)  # todo

            if save_subset:
                selected_ndx[run, epoch], selected_wgt[run, epoch] = indices, W_subset

            grd_time[run, epoch], sim_time[run, epoch], pred_time[run, epoch] = ordering_time, similarity_time, pre_time
            times_selected[run][indices] += 1
            not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
        else:
            pred_time = 0
            indices = np.arange(len(X_train))

        X_subset = X_train[indices, :]
        Y_subset = Y_train[indices]

        start = time.time()
        score = model.evaluate(X_test, Y_test, verbose=1)
        eval_time = time.time()-start

        start = time.time()
        score_loss = model.evaluate(X_train, Y_train, verbose=1)
        print(f'eval time on training: {time.time()-start}')

        test_loss[run][epoch], test_acc[run][epoch] = score[0], score[1]
        train_loss[run][epoch], train_acc[run][epoch] = score_loss[0], score_loss[1]
        best_acc = max(test_acc[run][epoch], best_acc)

        grd = 'random_wor' if random else 'grd_normw'
        print(f'run: {run}, {grd}, subset_size: {subset_size}, epoch: {epoch}, test_acc: {test_acc[run][epoch]}, '
              f'loss: {train_loss[run][epoch]}, best_prec1_gb: {best_acc}, not selected %:{not_selected[run][epoch]}')

    if save_subset:
        print(
            f'Saving the results to {folder}_{subset_size}_{grd}_{runs}')

        np.savez(f'{folder}_{subset_size}_{grd}_{runs}',
                 # f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}_subset',
                 train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                 train_time=train_time, grd_time=grd_time, sim_time=sim_time, pred_time=pred_time,
                 not_selected=not_selected, times_selected=times_selected,
                 subset=selected_ndx, weights=selected_wgt)
    else:
        print(
            f'Saving the results to {folder}_{subset_size}_{grd}_{runs}')

        np.savez(f'{folder}_{subset_size}_{grd}_{runs}',
                 # f'_{grd}_{args.lr_schedule}_start_{args.start_subset}_lag_{args.lag}',
                 train_loss=train_loss, test_acc=test_acc, train_acc=train_acc, test_loss=test_loss,
                 train_time=train_time, grd_time=grd_time, sim_time=sim_time, pred_time=pred_time,
                 not_selected=not_selected, times_selected=times_selected)

