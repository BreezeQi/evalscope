# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.perf.main import run_perf_benchmark


def run_perf():
    task_cfg = {
        'url': 'http://1.15.50.14:4043/api/aidong/kbqa/adrag_chat',
        'parallel': 1,
        'model': 'linde',
        'number': 1,
        'api': 'openai',
        'dataset': 'custom',
        'dataset_path': 'E:/workspace/evalscope/examples/test.jsonl',
        'apply_chat_template': True,
        'stream': True,
        'debug': True,
        'api_key': 'sk-z-UEJeKLUkUvN8sYfWl9gw',
        'max_prompt_length': 320000,
        'min_prompt_length': 0,
        'max_tokens': 320000,
        'min_tokens': 0,
    }
    run_perf_benchmark(task_cfg)
    
# def run_perf():
#     task_cfg = {
#         'url': 'http://llm-api.forklift-ai.com/v1/chat/completions',
#         'parallel': 1,
#         'model': 'qwen-max',
#         'number': 1,
#         'api': 'openai',
#         'dataset': 'custom',
#         'dataset_path': 'E:/workspace/evalscope/examples/test.jsonl',
#         'stream': True,
#         'debug': True,
#         'api_key': 'sk-z-UEJeKLUkUvN8sYfWl9gw',
#     }
#     run_perf_benchmark(task_cfg)


def run_perf_stream():
    task_cfg = {
        'url': 'http://127.0.0.1:8000/v1/chat/completions',
        'parallel': 1,
        'model': 'qwen2.5',
        'number': 15,
        'api': 'openai',
        'dataset': 'openqa',
        'stream': True,
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


def run_perf_speed_benchmark():
    task_cfg = {
        'url': 'http://127.0.0.1:8000/v1/completions',
        'parallel': 1,
        'model': 'qwen2.5',
        'api': 'openai',
        'dataset': 'speed_benchmark',
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


def run_perf_local():
    task_cfg = {
        'parallel': 1,
        'model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'number': 5,
        'api': 'local',
        'dataset': 'openqa',
        'debug': True,
        'dataset_path': 'E:/workspace/evalscope/examples/test.jsonl',
    }
    run_perf_benchmark(task_cfg)


def run_perf_local_stream():
    task_cfg = {
        'parallel': 1,
        'model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'number': 5,
        'api': 'local',
        'dataset': 'openqa',
        'stream': True,
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


def run_perf_local_speed_benchmark():
    task_cfg = {
        'parallel': 1,
        'model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'api': 'local_vllm',
        'dataset': 'speed_benchmark',
        'min_tokens': 2048,
        'max_tokens': 2048,
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


def run_perf_local_custom_prompt():
    task_cfg = {
        'parallel': 1,
        'model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'api': 'local',
        'number': 10,
        'prompt': '写一个诗歌',
        'min_tokens': 100,
        'max_tokens': 1024,
        'debug': True,
    }
    run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    run_perf()
    pass
    run_perf_local_custom_prompt()
