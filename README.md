# Sentient-AGI Alpinex Benchmark
Locust Testing Framework for Sentient AGI for Testing AlpineX Inference

### Get AlpineX API Key
From here: https://app.alpinex.ai/apiKeys

### Instructions to Run
1. Make sure you have poetry installed: https://python-poetry.org/docs/#installation
2. Install packes with `poetry install`
3. run in poetry shell `poetry shell`
4. Change `SENTIENT_TEST = "reasoning-ttft"` in the code (we support two ttft tests for now)
5. Fix `.env` 
    a. `cp .example-env .env` 
    b. Change `ALPINEX_API_KEY` with your API key
5. Run Locust `locust -f alpinex-load-test-ttft.py`
    a. You can open UI at http://0.0.0.0:8089/
    b. test with user = 100, ramp up = 5 (wait 1 min for it to warm up)