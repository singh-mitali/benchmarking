import tensorflow as tf
import os, sys
import numpy as np
import pandas
import threading
import multiprocessing as mp
import time
import argparse

from transformers import LlamaTokenizer
import huggingface_hub
HUGGINGFACE_TOKEN="XXXXXX"
huggingface_hub.login(token=HUGGINGFACE_TOKEN)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf") 

# sys.path.append('/saxml/bazel-bin/saxml/client/python/')
sys.path.append("/mnt/disks/persist/llama2_work/saxml/bazel-bin/saxml/client/python/")

import sax


def register_sax_model(model_id):
  model = sax.Model(model_id)
  global lm_model
  lm_model = model.LM()


def process_data(batch):
  option = sax.ModelOptions()
  option.SetExtraInput("per_example_max_decode_steps", 128)
  num_prompt_tokens = 0
  num_output_tokens = 0
  for prompt in batch:
    num_prompt_tokens += len(tokenizer.encode(prompt))
    predicted = lm_model.Generate(prompt, option)
    num_output_tokens += len(tokenizer.encode(predicted[0][0]))
  return num_prompt_tokens, num_output_tokens, 


def create_prompt_data(filename):
  df = pandas.read_pickle(filename)
  return df["input"].to_list()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, default="/sax/test/llama")
  parser.add_argument('-d', '--data', type=str, default="open_orca_gpt4_50k_filtered_tokenized_llama_prompt.pkl")
  parser.add_argument('-n', '--num_batches', type=int, default=32)
  parser.add_argument('-b', '--batch_size', type=int, default=1)
  parser.add_argument('-t', '--num_threads', type=int, default=1)
  args = parser.parse_args()

  register_sax_model(args.model)
  prompts = create_prompt_data(args.data)
  num_prompts = args.num_batches * args.batch_size
  prompts = prompts[:num_prompts]

  start = time.time()
  batched_data = []
  for i in range(0, args.num_batches):
    batched_data.append(prompts[i:i+args.batch_size])

  total_input_tokens = 0
  total_output_tokens = 0
  with mp.pool.ThreadPool(processes=args.num_threads) as pool:
    for result in pool.map(process_data, batched_data):
      total_input_tokens += result[0]
      total_output_tokens += result[1]

  total_time = time.time() - start
  # print(f"batch_size: {args.batch_size}")
  # print(f"threads: {args.num_threads}")
  # print(f"prompts: {len(prompts)}")
  # print(f"batches: {len(batched_data)}")
  # print(f"input tokens: {total_input_tokens}")
  # print(f"output tokens: {total_output_tokens}")
  # print(f"time: {total_time}")
  # print(f"time per batch: {total_time / len(batched_data)}")
  # print(f"time per input: {total_time / len(prompts)}")
  # print(f"time per output token: {total_time / total_output_tokens}")
  # print(f"output tokens per second: {total_output_tokens / total_time}")
  # print(f"input tokens per prompt: {total_input_tokens / len(prompts)}")
  # print(f"output tokens per prompt: {total_output_tokens / len(prompts)}")
  # BatchSize	Batches	Threads	Time	QPS	OutTokenPerSec	Batch Latency (s)	Query Latency (s)	AvgInputLen	AvgOutputLen
  
  print("BatchSize, Batches, Threads, Time, QPS, OutTokenPerSec, Batch Latency(s), Query Latency (s), AvgInputLen, AvgOutputLen")
  print(f"{args.batch_size}, {len(batched_data)}, {args.num_threads}, {total_time}, {total_time / len(prompts)}, {total_output_tokens / total_time}, {total_time / len(batched_data)}, {total_time / len(prompts)}, {total_input_tokens / len(prompts)}, {total_output_tokens / len(prompts)}")


if __name__ == '__main__':
  main()
