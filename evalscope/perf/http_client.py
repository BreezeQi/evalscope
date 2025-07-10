import aiohttp
import asyncio
import json
import time
import re
import os
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Tuple, Optional

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.local_server import ServerSentEvent
from evalscope.utils.logger import get_logger

logger = get_logger()


def load_custom_config() -> Optional[Dict]:
    """
    从环境变量config_path加载配置文件
    
    返回示例:
    {
        "enabled": true,
        "user_id": "user66bc707e62bdb86871ef345b",
        "kb_ids": ["kb2bf6ce06c6d645ab9c33478b88178c92"],
        "websearch": false,
        "imageRecognition": true,
        "iterations": 1
    }
    """
    os.environ['CONFIG_PATH'] = 'E:/workspace/evalscope/evalscope/linde_config.json'
    config_path = os.environ.get('CONFIG_PATH')
    if not config_path or not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load custom config from {config_path}: {e}")
        return None


def custom_format_data(body: dict, config: Dict) -> dict:
    """
    将标准聊天格式转换为自定义API格式
    
    输入示例:
    body = {
        "messages": [{"role": "user", "content": "MT15SC车型标配的电池是多大的？"}],
        "model": "qwen-max",
        "max_tokens": 10,
        "stream": false
    }
    config = {
        "user_id": "user66bc707e62bdb86871ef345b",
        "kb_ids": ["kb2bf6ce06c6d645ab9c33478b88178c92"],
        "websearch": false,
        "imageRecognition": true,
        "iterations": 1
    }
    
    输出示例:
    {
        "user_id": "user66bc707e62bdb86871ef345b",
        "kb_ids": ["kb2bf6ce06c6d645ab9c33478b88178c92"],
        "question": "MT15SC车型标配的电池是多大的？",
        "history": [],
        "images": [],
        "streaming": false,
        "websearch": false,
        "imageRecognition": true,
        "iterations": 1
    }
    """
    # 提取用户消息内容作为问题
    question = ""
    if "messages" in body:
        messages = body["messages"]
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                if i == len(messages) - 1:  # 最后一条用户消息作为当前问题
                    question = msg.get("content", "")
    # 构建自定义格式
    custom_data = {
        "user_id": config.get("user_id", "user66bc707e62bdb86871ef345b"),
        "kb_ids": config.get("kb_ids", ["kb2bf6ce06c6d645ab9c33478b88178c92"]),
        "question": question,
        "history": config.get("history", []),
        "images": [],
        "streaming": body.get("stream", False),
        "websearch": config.get("websearch", False),
        "imageRecognition": config.get("imageRecognition", True),
        "iterations": config.get("iterations", 1)
    }
    
    return custom_data


class AioHttpClient:

    def __init__(
        self,
        args: Arguments,
    ):
        self.url = args.url
        self.headers = {'user-agent': 'modelscope_bench', **(args.headers or {})}
        self.read_timeout = args.read_timeout
        self.connect_timeout = args.connect_timeout
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(connect=self.connect_timeout, sock_read=self.read_timeout),
            trace_configs=[self._create_trace_config()] if args.debug else [])

    def _create_trace_config(self):
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_chunk_sent.append(self.on_request_chunk_sent)
        trace_config.on_response_chunk_received.append(self.on_response_chunk_received)
        return trace_config

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.close()

    async def _custom_handle_stream(self, response: aiohttp.ClientResponse):
        """
        将自定义API的流式响应转换为OpenAI Chat Completion流式格式
        
        输入格式：
        data: {"status": "success", "response": "MT"}
        
        输出格式：
        data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"MT"}}]}
        """
        is_error = False
        collected_responses = []  # 收集所有response内容用于计算tokens
        first_question = ""  # 保存第一次获取到的问题用于计算prompt tokens
        chat_id = f"chatcmpl-custom-{int(time.time())}"  # 生成唯一ID
        created_timestamp = int(time.time())
        
        async for line in response.content:
            line = line.decode('utf8').rstrip('\n\r')
            
            # 处理多种格式的流式数据
            if line.startswith('data: '):
                data_content = line[6:]  # 去掉 "data: " 前缀
            elif line.strip() and not line.startswith(':'):  # 忽略注释行
                data_content = line.strip()
            else:
                continue
                
            try:
                data_json = json.loads(data_content)
                logger.debug(f'Custom response received: {line}')
                
                # 检查错误状态
                if data_json.get("status") != "success":
                    is_error = True
                    # 返回错误格式
                    error_response = {
                        "error": {
                            "message": data_json.get("message", "API returned error status"),
                            "type": "api_error",
                            "code": "custom_api_error"
                        }
                    }
                    yield True, response.status, json.dumps(error_response, ensure_ascii=False)
                    continue
                
                # 跳过包含"running"字段的消息
                if "running" in data_json:
                    continue
                
                # 处理包含response内容的消息
                if "response" in data_json:
                    response_content = data_json["response"]
                    
                    # 收集非空内容
                    if response_content:  
                        collected_responses.append(response_content)
                    
                    # 保存第一次看到的问题信息
                    if not first_question and "improved_question" in data_json:
                        first_question = data_json["improved_question"]
                    
                    # 检查是否为最后一条消息
                    is_final = ("source_documents" in data_json or 
                              "chat_history" in data_json or 
                              (response_content == "" and len(collected_responses) > 0))
                    
                    if is_final:
                        # 最后一条消息，计算usage并添加
                        full_response = ''.join(collected_responses)
                        
                        # 计算tokens
                        question_for_prompt = (
                            data_json.get("improved_question", "") or
                            first_question or
                            ""
                        )
                        prompt_tokens = len(question_for_prompt.encode('utf-8'))*0.6
                        
                        # 使用正则提取thinking内容
                        thinking_pattern = r'<think>(.*?)</think>'
                        thinking_matches = re.findall(thinking_pattern, full_response, re.DOTALL)
                        reasoning_tokens = sum(len(match.encode('utf-8')) for match in thinking_matches)*0.6
                        
                        # 移除thinking标签后的内容长度
                        content_without_thinking = re.sub(thinking_pattern, '', full_response, flags=re.DOTALL)
                        completion_tokens = len(content_without_thinking.encode('utf-8'))*0.6
                        total_tokens = prompt_tokens + completion_tokens + reasoning_tokens
                        
                        # 构建最终的OpenAI格式响应（带usage）
                        openai_response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": "custom-api",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "completion_tokens": completion_tokens,
                                "prompt_tokens": prompt_tokens,
                                "total_tokens": total_tokens,
                                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
                                "prompt_tokens_details": {"cached_tokens": 0}
                            }
                        }
                        yield False, response.status, json.dumps(openai_response, ensure_ascii=False)
                        
                        # 结束流式处理（与原始格式一致）
                        break
                    else:
                        # 非最后一条消息，转换为OpenAI流式格式
                        openai_response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk", 
                            "created": created_timestamp,
                            "model": "custom-api",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": response_content,
                                    "role": "assistant"
                                }
                            }]
                        }
                        yield False, response.status, json.dumps(openai_response, ensure_ascii=False)
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {data_content}")
                # 可能是非JSON的流式数据，直接返回
                yield True, response.status, data_content

    async def _handle_stream(self, response: aiohttp.ClientResponse):
        is_error = False
        async for line in response.content:
            line = line.decode('utf8').rstrip('\n\r')
            sse_msg = ServerSentEvent.decode(line)
            if sse_msg:
                logger.debug(f'Response recevied: {line}')
                if sse_msg.event == 'error':
                    is_error = True
                if sse_msg.data:
                    if sse_msg.data.startswith('[DONE]'):
                        break
                    yield is_error, response.status, sse_msg.data

    async def _custom_handle_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Tuple[bool, int, str], None]:
        """
        处理自定义API的响应格式
        
        支持的响应格式：
        1. 流式: data: {"status": "success", "response": "内容"}
        2. 非流式: {"status": "success", "response": "内容", "source_documents": [...]}
        
        输入: aiohttp.ClientResponse对象
        输出: AsyncGenerator[Tuple[bool, int, str], None] - (is_error, status_code, content)
        """
        response_status = response.status
        response_content_type = response.content_type
        content_type_json = 'application/json'
        content_type_event_stream = 'text/event-stream'
        is_success = response_status == HTTPStatus.OK

        if is_success:
            # 处理流式响应 - 支持 text/event-stream 和 text/plain
            if (content_type_event_stream in response_content_type or 
                'text/plain' in response_content_type or
                response_content_type.startswith('text/')):
                async for is_error, response_status, content in self._custom_handle_stream(response):
                    yield (is_error, response_status, content)
            # 处理JSON响应
            elif content_type_json in response_content_type:
                content = await response.json()
                
                # 检查是否为自定义API格式 (包含status字段)
                if "status" in content:
                    if content.get('status') != 'success':
                        # 错误响应
                        error_response = {
                            "error": {
                                "message": content.get("message", "API returned error status"),
                                "type": "api_error", 
                                "code": "custom_api_error"
                            }
                        }
                        yield (True, response_status, json.dumps(error_response, ensure_ascii=False))
                    else:
                        # 成功响应，转换为OpenAI格式
                        response_text = content.get("response", "")
                        
                        # 计算tokens
                        improved_question = ""
                        step_data = content.get("step", {})
                        if isinstance(step_data, dict):
                            cls_data = step_data.get("cls", {})
                            if isinstance(cls_data, dict):
                                improved_question = cls_data.get("improved_question", "")
                        
                        prompt_tokens = len(improved_question.encode('utf-8'))*0.6
                        
                        # 使用正则提取thinking内容
                        thinking_pattern = r'<think>(.*?)</think>'
                        thinking_matches = re.findall(thinking_pattern, response_text, re.DOTALL)
                        reasoning_tokens = sum(len(match.encode('utf-8')) for match in thinking_matches)*0.6
                        
                        # 移除thinking标签后的内容长度
                        content_without_thinking = re.sub(thinking_pattern, '', response_text, flags=re.DOTALL)
                        completion_tokens = len(content_without_thinking.encode('utf-8'))*0.6
                        total_tokens = prompt_tokens + completion_tokens + reasoning_tokens
                        
                        # 转换为OpenAI Chat Completion格式
                        openai_response = {
                            "id": f"chatcmpl-custom-{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": "custom-api",
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response_text
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "completion_tokens": completion_tokens,
                                "prompt_tokens": prompt_tokens,
                                "total_tokens": total_tokens,
                                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
                                "prompt_tokens_details": {"cached_tokens": 0}
                            }
                        }
                        yield (False, response_status, json.dumps(openai_response, ensure_ascii=False))
                
                # 检查是否为标准OpenAI Chat Completion格式
                elif "choices" in content and content.get("object") == "chat.completion":
                    # 标准OpenAI格式，已经包含usage信息，直接返回
                    logger.debug(f'OpenAI format response received with usage: {content.get("usage", {})}')
                    yield (False, response_status, json.dumps(content, ensure_ascii=False))
                
                # 检查是否为错误响应
                elif content.get('object') == 'error' or 'error' in content:
                    yield (True, response_status, json.dumps(content, ensure_ascii=False))
                
                else:
                    # 其他格式的JSON响应，直接返回
                    yield (False, response_status, json.dumps(content, ensure_ascii=False))
            else:
                # 处理其他类型响应，可能包含data:格式的流式数据
                content_bytes = await response.read()
                content_text = content_bytes.decode('utf-8')
                
                # 检查是否是data:格式的流式响应
                if 'data:' in content_text:
                    # 手动解析data:格式
                    lines = content_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('data: '):
                            data_content = line[6:]  # 去掉 "data: " 前缀
                            try:
                                data_json = json.loads(data_content)
                                is_error = data_json.get("status") != "success"
                                yield (is_error, response_status, json.dumps(data_json, ensure_ascii=False))
                            except json.JSONDecodeError:
                                yield (True, response_status, data_content)
                else:
                    yield (False, response_status, content_text)
        else:
            # 处理错误响应
            if content_type_json in response_content_type:
                error = await response.json()
                yield (True, response_status, json.dumps(error, ensure_ascii=False))
            else:
                msg = await response.read()
                yield (True, response_status, msg.decode('utf-8'))

    async def _handle_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Tuple[bool, int, str], None]:
        response_status = response.status
        response_content_type = response.content_type
        content_type_json = 'application/json'
        content_type_event_stream = 'text/event-stream'
        is_success = response_status == HTTPStatus.OK

        if is_success:
            # Handle successful response with 'text/event-stream' content type
            if content_type_event_stream in response_content_type:
                async for is_error, response_status, content in self._handle_stream(response):
                    yield (is_error, response_status, content)
            # Handle successful response with 'application/json' content type
            elif content_type_json in response_content_type:
                content = await response.json()
                if content.get('object') == 'error':
                    yield (True, content.get('code'), content.get('message'))  # DashScope
                else:
                    yield (False, response_status, json.dumps(content, ensure_ascii=False))
            # Handle other successful responses
            else:
                content = await response.read()
                yield (False, response_status, content.decode('utf-8'))
        else:
            # Handle error response with 'application/json' content type
            if content_type_json in response_content_type:
                error = await response.json()
                yield (True, response_status, json.dumps(error, ensure_ascii=False))
            # Handle error response with 'text/event-stream' content type
            elif content_type_event_stream in response_content_type:
                async for _, _, data in self._handle_stream(response):
                    error = json.loads(data)
                    yield (True, response_status, json.dumps(error, ensure_ascii=False))
            # Handle other error responses
            else:
                msg = await response.read()
                yield (True, response_status, msg.decode('utf-8'))

    async def post(self, body):
        headers = {'Content-Type': 'application/json', **self.headers}
        try:
            # 加载自定义配置
            custom_config = load_custom_config()
            use_custom_format = custom_config and custom_config.get('enabled', False)
            
            # 根据配置选择数据格式化方式
            if use_custom_format and custom_config:
                formatted_data = custom_format_data(body, custom_config)
            else:
                formatted_data = body
            
            data = json.dumps(formatted_data, ensure_ascii=False)  # 原有的简单dumps
            async with self.client.request('POST', url=self.url, data=data, headers=headers) as response:
                # 根据配置选择响应处理方式
                if use_custom_format:
                    async for rsp in self._custom_handle_response(response):
                        yield rsp
                else:
                    async for rsp in self._handle_response(response):
                        yield rsp
        except asyncio.TimeoutError:
            logger.error(
                f'TimeoutError: connect_timeout: {self.connect_timeout}, read_timeout: {self.read_timeout}. Please set longger timeout.'  # noqa: E501
            )
            yield (True, None, 'Timeout')
        except (aiohttp.ClientConnectorError, Exception) as e:
            logger.error(e)
            yield (True, None, e)

    @staticmethod
    async def on_request_start(session, context, params: aiohttp.TraceRequestStartParams):
        logger.debug(f'Starting request: <{params}>')

    @staticmethod
    async def on_request_chunk_sent(session, context, params: aiohttp.TraceRequestChunkSentParams):
        method = params.method
        url = params.url
        chunk = params.chunk.decode('utf-8')
        max_length = 100
        if len(chunk) > 2 * max_length:
            truncated_chunk = f'{chunk[:max_length]}...{chunk[-max_length:]}'
        else:
            truncated_chunk = chunk
        logger.debug(f'Request sent: <{method=},  {url=}, {truncated_chunk=}>')

    @staticmethod
    async def on_response_chunk_received(session, context, params: aiohttp.TraceResponseChunkReceivedParams):
        method = params.method
        url = params.url
        chunk = params.chunk.decode('utf-8')
        max_length = 200
        if len(chunk) > 2 * max_length:
            truncated_chunk = f'{chunk[:max_length]}...{chunk[-max_length:]}'
        else:
            truncated_chunk = chunk
        logger.debug(f'Request received: <{method=},  {url=}, {truncated_chunk=}>')


async def test_connection(args: Arguments) -> bool:
    is_error = True
    start_time = time.perf_counter()

    async def attempt_connection():
        client = AioHttpClient(args)
        async with client:
            if args.apply_chat_template:
                request = {
                    'messages': [{
                        'role': 'user',
                        'content': 'hello'
                    }],
                    'model': args.model,
                    'max_tokens': 10,
                    'stream': args.stream
                }
            else:
                request = {'prompt': 'hello', 'model': args.model, 'max_tokens': 10}
            async for is_error, state_code, response_data in client.post(request):
                return is_error, state_code, response_data
        return True, None, "No response received"

    while True:
        try:
            is_error, state_code, response_data = await asyncio.wait_for(
                attempt_connection(), timeout=args.connect_timeout)
            if not is_error:
                logger.info('Test connection successful.')
                return True
            logger.warning(f'Retrying...  <{state_code}> {response_data}')
        except Exception as e:
            logger.warning(f'Retrying... <{e}>')

        if time.perf_counter() - start_time >= args.connect_timeout:
            logger.error('Overall connection attempt timed out.')
            return False

        await asyncio.sleep(10)
