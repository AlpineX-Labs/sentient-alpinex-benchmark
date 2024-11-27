# sentient-alpinex-benchmark
Locust Testing Framework for Sentient AGI for Testing AlpineX Inference

### Instructions to Run
1. Make sure you have poetry installed: https://python-poetry.org/docs/#installation
2. Install packes with `poetry install`
3. run in poetry shell `poetry shell`
4. Change `SENTIENT_TEST = "reasoning-ttft"` in the code (supported two ttft tests for now)
5. Run Locust `locust -f alpinex-load-test-ttft.py`
6. You can open UI at http://0.0.0.0:8089/ and test with user = 100, ramp up = 5 (wait 1 min for it to warm up)