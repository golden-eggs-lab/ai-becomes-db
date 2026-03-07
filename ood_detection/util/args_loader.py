import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="imagenet", type=str, help='CIFAR-10 imagenet')
    parser.add_argument('--out-datasets', default=['inat', 'sun50', 'places50', 'dtd', ], nargs="*", type=str, help="['SVHN', 'LSUN', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")
    parser.add_argument('--name', default="resnet18-supcon", type=str, help='neural network name and training set')
    parser.add_argument('--model-arch', default='resnet18-supcon', type=str, help='model architecture')
    parser.add_argument('--p', default=0, type=float, help='sparsity level')
    parser.add_argument('--imagenet-root', default='./datasets/imagenet/', type=str, help='imagenet root')
    parser.add_argument('--seed', default=0, type=int, help='seed')

    parser.add_argument('--method', default='', type=str, help='')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save-epoch', default=10, type=int,
                        help='save the model every save_epoch')
    parser.add_argument('--cal-metric', help='calculatse metric directly', action='store_true')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
    parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')
    parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
    parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
    parser.add_argument('--depth', default=40, type=int, help='depth of resnet')
    parser.add_argument('--width', default=4, type=int, help='width of resnet')
    parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
    parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
    
    # ANN parameters
    parser.add_argument('--use-ann', action='store_true', help='use approximate nearest neighbors instead of exact KNN')
    parser.add_argument('--ann-method', default='ivf', type=str, choices=['ivf', 'hnsw'], help='ANN method: ivf or hnsw')
    parser.add_argument('--nlist', default=100, type=int, help='number of clusters for IVF (default: 100)')
    parser.add_argument('--nprobe', default=10, type=int, help='number of clusters to search for IVF (default: 10)')
    parser.add_argument('--hnsw-M', default=32, type=int, help='number of connections for HNSW (default: 32)')
    parser.add_argument('--hnsw-efSearch', default=64, type=int, help='search parameter for HNSW (default: 64)')
    parser.add_argument('--use-faiss-gpu', action='store_true', help='use GPU for Faiss index (both KNN and ANN)')
    
    # Milvus parameters
    parser.add_argument('--use-milvus', action='store_true', help='use Milvus vector database instead of Faiss')
    parser.add_argument('--milvus-lite', action='store_true', help='use Milvus Lite (local embedded mode) instead of Milvus server')
    parser.add_argument('--milvus-host', default='localhost', type=str, help='Milvus server host (default: localhost, ignored if using Milvus Lite)')
    parser.add_argument('--milvus-port', default='19530', type=str, help='Milvus server port (default: 19530, ignored if using Milvus Lite)')
    parser.add_argument('--milvus-index-type', default='IVF_FLAT', type=str, choices=['FLAT', 'IVF_FLAT', 'IVF_SQ8', 'HNSW'], help='Milvus index type')
    parser.add_argument('--milvus-metric', default='L2', type=str, choices=['L2', 'IP'], help='Milvus distance metric')
    
    # Spark parameters
    parser.add_argument('--use-spark', action='store_true', help='use Spark for distributed KNN search')
    parser.add_argument('--spark-master', default='local[*]', type=str, help='Spark master URL (default: local[*])')
    parser.add_argument('--spark-partitions', default=100, type=int, help='number of Spark partitions (default: 100)')
    parser.add_argument('--spark-batch-size', default=1000, type=int, help='batch size for Spark processing (default: 1000)')
    parser.add_argument('--spark-subsample', default=0.1, type=float, help='subsample ratio of training data for Spark (default: 0.1 = 10%%)')
    parser.add_argument('--subsample-ratio', default=1.0, type=float, help='subsample ratio of training data for scaling experiments (default: 1.0 = 100%%)')
    parser.add_argument('--spark-use-lsh', action='store_true', help='use LSH for approximate search in Spark')
    parser.add_argument('--spark-lsh-tables', default=5, type=int, help='number of hash tables for LSH (default: 5)')
    parser.add_argument('--spark-lsh-bucket-length', default=2.0, type=float, help='bucket length for LSH (default: 2.0)')
    parser.add_argument('--spark-lsh-threshold', default=2.0, type=float, help='distance threshold for LSH similarity join (default: 2.0, max L2 dist for normalized vectors)')
    
    parser.set_defaults(argument=True)
    args = parser.parse_args()
    return args