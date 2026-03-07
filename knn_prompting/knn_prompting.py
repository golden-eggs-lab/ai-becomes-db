import json
import random
from datetime import datetime
from time import sleep, time
import logging
import argparse
from tqdm import tqdm
import csv
import os

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from utils.dataset import *
from utils.template import *
from utils.anchor import AnchorStore
from utils.anchor_reuse import AnchorStoreOptV1, AnchorStoreOptV2
try:
    from utils.anchor_milvus import AnchorStoreMilvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("Warning: pymilvus not installed. Milvus support disabled.")

try:
    from utils.anchor_spark import AnchorStoreSparkV3
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Warning: pyspark not installed. Spark support disabled.")


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="KNN Prompting.")
    parser.add_argument(
        "--llm_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_train_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_demo_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_anchor_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="baseline",
        choices=["baseline", "opt_v1", "opt_v2"],
        help="Optimization level: baseline (original KL), opt_v1 (pre-compute log), opt_v2 (pre-compute log + matmul)",
    )
    parser.add_argument(
        "--use_milvus",
        action="store_true",
        help="Use Milvus vector database for search",
    )
    parser.add_argument(
        "--milvus_use_ann",
        action="store_true",
        help="When using Milvus, use ANN (IVF) instead of exact search (FLAT)",
    )
    parser.add_argument(
        "--milvus_no_projection",
        action="store_true",
        help="When using Milvus, disable dimension reduction (requires Milvus with proxy.maxDimension >= vocab_size)",
    )
    parser.add_argument(
        "--use_spark",
        action="store_true",
        help="Use Spark for distributed k-NN search",
    )
    parser.add_argument(
        "--spark_master",
        type=str,
        default="local[*]",
        help="Spark master URL (e.g., local[*], spark://host:port)",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="results_knnprompting.csv",
        help="CSV filename for saving results",
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def llm_gen(model, prompt, tokenizer, max_context_len):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device=model.device)
    if inputs['input_ids'].shape[1] > max_context_len:
        inputs['input_ids'] = inputs['input_ids'][:, -max_context_len:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -max_context_len:]
    with torch.no_grad():
        logits = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],
                               return_dict=True).logits.detach().cpu()
    # the output prob is shifted by -1, so we should use the output at the last input token position
    # gen_logits.shape = [1, 50257]
    gen_logits = logits[:, -1, :]

    return gen_logits


def main():
    args = parse_args()

    args.n_anchor_shot = args.n_train_shot - args.n_demo_shot
    if args.n_anchor_shot <= 0:
        raise Exception("Num. of demonstration must be set smaller than num. of training.")

    args.knn = min(args.knn, args.n_anchor_shot)  # knn can not exceed num. of anchors

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(args.llm_dir)
    model = AutoModelForCausalLM.from_pretrained(args.llm_dir)
    model.to(device)
    model.eval()

    if 'gpt2' in args.llm_dir:
        max_context_len = 1024
    else:
        max_context_len = 2048

    # prepare dataset
    if args.dataset == 'sst2':
        AutoDataset = SST2Dataset
    elif args.dataset == 'subj':
        AutoDataset = SUBJDataset
    elif args.dataset == 'agnews':
        AutoDataset = AGNEWSDataset
    elif args.dataset == 'cb':
        AutoDataset = CBDataset
    elif args.dataset == 'cr':
        AutoDataset = CRDataset
    elif args.dataset == 'dbpedia':
        AutoDataset = DBPEDIADataset
    elif args.dataset == 'mpqa':
        AutoDataset = MPQADataset
    elif args.dataset == 'mr':
        AutoDataset = MRDataset
    elif args.dataset == 'rte':
        AutoDataset = RTEDataset
    elif args.dataset == 'trec':
        AutoDataset = TRECDataset

    datadir = os.path.join(args.data_dir, args.dataset)
    train_data = AutoDataset(datadir, mode='train')
    dev_data = AutoDataset(datadir, mode='dev')

    anchor_data = AutoDataset(datadir, mode='train')

    # Stage1: Meta Test
    train_data.subsamplebyshot(args.n_demo_shot, args.seed)
    prompt_prefix = make_prompt(train_data, args.dataset, mode='train')
    anchor_data.subsamplebyshot(args.n_anchor_shot, args.seed, exclude=train_data.data)
    label2id = dev_data.label2id
    id2verb = train_data.id2verb
    logger.info(f"===== build anchor store of {anchor_data.__len__()} anchor examples =====")
    
    # Choose anchor store based on setup: in-memory / Milvus / Spark
    if args.use_spark:
        # Spark distributed setup
        if not SPARK_AVAILABLE:
            raise ImportError("Spark requested but pyspark not installed. Install with: pip install pyspark")
        
        method_name = "Spark (Batch Processing)"
        logger.info(f"Using method: {method_name}")
        anchor_store = AnchorStoreSparkV3(
            K=anchor_data.__len__(),
            dim=model_config.vocab_size,
            knn=args.knn,
            n_class=len(label2id),
            spark_master=args.spark_master,
            app_name=f"KNN_Prompting_Spark_{args.dataset}"
        )
    
    elif args.use_milvus:
        # Milvus vector database setup
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus requested but pymilvus not installed. Install with: pip install pymilvus")
        
        method_name = f"Milvus ({'ANN' if args.milvus_use_ann else 'Exact KNN'})"
        logger.info(f"Using method: {method_name}")
        
        milvus_db_name = f"milvus_knn_{args.dataset}_{'ann' if args.milvus_use_ann else 'exact'}.db"
        anchor_store = AnchorStoreMilvus(
            K=anchor_data.__len__(),
            dim=model_config.vocab_size,
            knn=args.knn,
            n_class=len(label2id),
            use_ann=args.milvus_use_ann,
            collection_name=f"knn_anchors_{args.dataset}_{'ann' if args.milvus_use_ann else 'exact'}",
            use_projection=not args.milvus_no_projection,
            milvus_uri=f"./{milvus_db_name}"
        )
    else:
        # In-memory implementations
        opt_names = {
            "baseline": "Baseline (original KL with runtime log)",
            "opt_v1": "Opt-v1 (pre-computed log only)",
            "opt_v2": "Opt-v2 (pre-computed log + matmul)"
        }
        logger.info(f"Using method: {opt_names[args.optimization]}")
        
        if args.optimization == "baseline":
            anchor_store = AnchorStore(K=anchor_data.__len__(),
                                       dim=model_config.vocab_size,
                                       knn=args.knn,
                                       n_class=len(label2id))
        elif args.optimization == "opt_v1":
            anchor_store = AnchorStoreOptV1(K=anchor_data.__len__(),
                                            dim=model_config.vocab_size,
                                            knn=args.knn,
                                            n_class=len(label2id))
        else:  # opt_v2
            anchor_store = AnchorStoreOptV2(K=anchor_data.__len__(),
                                            dim=model_config.vocab_size,
                                            knn=args.knn,
                                            n_class=len(label2id))
    
    # Build anchor store - measure time
    build_start_time = time()
    for ins in tqdm(anchor_data.data, total=anchor_data.__len__(), desc="Building anchors"):
        labels = label2id[ins['label']]
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        anchor_store.enqueue(torch.softmax(gen_logits, dim=-1), torch.tensor(labels))
    
    # Build index if using Milvus or Spark
    if args.use_milvus or args.use_spark:
        anchor_store.build_index()
    
    build_time = time() - build_start_time
    logger.info(f"Anchor store build time: {build_time:.2f}s")

    # Stage2: Formal Test
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
    dev_labels = []
    dev_pred = []
    
    # Measure inference time
    infer_start_time = time()
    for ins in tqdm(dev_data.data, total=dev_data.__len__(), desc="Inference"):
        dev_labels.append(label2id[ins['label']])
        prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
        gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
        dev_pred.extend(anchor_store.knn_infer(torch.softmax(gen_logits, dim=-1)))
    infer_time = time() - infer_start_time
    total_time = build_time + infer_time

    dev_correct = [1 if dev_labels[i] == dev_pred[i] else 0 for i in range(len(dev_labels))]
    acc = sum(dev_correct) / len(dev_labels)
    logger.info(f"Acc: {acc}")
    logger.info(f"Build time: {build_time:.2f}s, Inference time: {infer_time:.2f}s, Total time: {total_time:.2f}s")
    
    # Log bottleneck stats for baseline in-memory
    bottleneck_stats = None
    if hasattr(anchor_store, 'get_bottleneck_stats'):
        bottleneck_stats = anchor_store.get_bottleneck_stats()
        logger.info(f"=== Bottleneck Analysis ===")
        logger.info(f"  log(k) computation time: {bottleneck_stats['log_k_time']:.4f}s ({bottleneck_stats['log_k_percentage']:.2f}%)")
        logger.info(f"  Other computation time:  {bottleneck_stats['other_time']:.4f}s ({bottleneck_stats['other_percentage']:.2f}%)")

    # Cleanup resources if used
    if args.use_milvus:
        try:
            anchor_store.cleanup()
        except:
            pass
    elif args.use_spark:
        # Spark session cleanup is handled in __del__
        pass
    
    # logging
    if args.use_spark:
        method_tag = "spark"
    elif args.use_milvus:
        method_tag = f"milvus_{'ann' if args.milvus_use_ann else 'exact'}"
    else:
        method_tag = args.optimization
    
    # Use custom csv_name if provided, otherwise use default naming
    save_results_file = os.path.join(args.output_dir, args.csv_name)
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'n_demo_shot', 'n_anchor_shot', 'seed', 'knn', 'method', 'setup', 'acc', 'build_time', 'infer_time', 'total_time', 'log_k_pct'])
        
        # Determine setup string
        if args.use_spark:
            setup = f"spark_{args.spark_master.replace(':', '_')}"
        elif args.use_milvus:
            setup = f"milvus_{'ann' if args.milvus_use_ann else 'exact'}"
        else:
            setup = f"inmem_{args.optimization}"
        
        # Get bottleneck percentage if available
        log_k_pct = ""
        if bottleneck_stats:
            log_k_pct = f"{bottleneck_stats['log_k_percentage']:.2f}"
        
        csvwriter.writerow([args.dataset,
                            args.llm_dir,
                            args.n_train_shot,
                            args.n_demo_shot,
                            args.n_anchor_shot,
                            args.seed,
                            args.knn,
                            method_tag,
                            setup,
                            acc,
                            f"{build_time:.2f}",
                            f"{infer_time:.2f}",
                            f"{total_time:.2f}",
                            log_k_pct])


if __name__ == "__main__":
    main()
