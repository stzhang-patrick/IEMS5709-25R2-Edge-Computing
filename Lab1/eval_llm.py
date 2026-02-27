import requests
import time
import json

# vLLM server address
API_URL = "http://localhost:8000/v1/chat/completions"

# Request headers
headers = {"Content-Type": "application/json"}

# Request data
data = {
    "model": "/root/.cache/huggingface/Qwen3-4B-quantized.w4a16",
    "messages": [
        {"role": "user", "content": "Please provide a detailed introduction to the main features of Jetson Orin NX."}
    ],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": True  # Stream must be enabled to measure TTFT
}

# Record request start time
start_time = time.time()
first_token_time = None
response_tokens = 0

print("Sending request...")
with requests.post(API_URL, headers=headers, json=data, stream=True) as response:
    if response.status_code == 200:
        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith('data: '):
                    content = chunk_str[6:]
                    if content.strip() == '[DONE]':
                        break
                    try:
                        payload = json.loads(content)
                        # Record first token time
                        if first_token_time is None:
                            first_token_time = time.time()
                        
                        # Count tokens
                        if 'choices' in payload and payload['choices'][0]['delta'].get('content'):
                            response_tokens += 1
                            print(payload['choices'][0]['delta']['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        print(f"\nUnable to parse JSON: {content}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

# Record request end time
end_time = time.time()
print("\n\n--- Performance Test Results ---")

if first_token_time:
    ttft = (first_token_time - start_time) * 1000
    print(f"Time to First Token (TTFT): {ttft:.2f} ms")

if response_tokens > 1 and first_token_time:
    total_generation_time = end_time - first_token_time
    throughput = (response_tokens - 1) / total_generation_time
    print(f"Subsequent Token Throughput: {throughput:.2f} tokens/sec")

print(f"Total Response Time: {(end_time - start_time)*1000:.2f} ms")
print(f"Total Generated Tokens: {response_tokens}")