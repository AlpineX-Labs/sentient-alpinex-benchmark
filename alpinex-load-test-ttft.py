from locust import HttpUser, task, between, stats, events
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
import random

load_dotenv()
stats.PERCENTILES_TO_CHART = [0.5, 0.8, 0.9]

# SENTIENT_TEST = "short-ttft"
SENTIENT_TEST = "reasoning-ttft"
ALPINEX_HOST = "https://api.alpinex.ai/v1"


def get_random_message(test_type="short-ttft"):
    categories = [
            "ETF",
            "mutual fund",
            "investment strategy",
            "stock",
            "cryptocurrency",
            "savings account",
            "credit card",
            "interest rate",
            "budget app",
            "financial advisor",
            "real estate investment",
            "retirement plan",
            "dividend stock",
            "growth stock",
            "REIT",
            "bond",
            "index fund",
            "trading platform",
            "portfolio allocation",
            "hedge fund",
            "asset allocation",
            "insurance policy",
            "annuity",
            "personal loan",
            "business loan",
            "small-cap stock",
            "large-cap stock",
            "mid-cap stock",
            "startup idea",
            "venture capital fund",
            "angel investment",
            "high-yield savings account",
            "checking account",
            "online broker",
            "robo-advisor",
            "financial tool",
            "accounting software",
            "budgeting software",
            "retirement community",
            "401(k) plan",
            "IRA",
            "Roth IRA",
            "health insurance",
            "life insurance",
            "car insurance",
            "home insurance",
            "travel insurance",
            "term life insurance",
            "whole life insurance",
            "mortgage rate",
            "refinance rate",
            "property investment",
            "short-term investment",
            "long-term investment",
            "ESG fund",
            "socially responsible investment",
            "hedge against inflation",
            "emerging market ETF",
            "developed market ETF",
            "international ETF",
            "sector ETF",
            "technology ETF",
            "healthcare ETF",
            "energy ETF",
            "materials ETF",
            "consumer ETF",
            "financial ETF",
            "industrial ETF",
            "utility ETF",
            "real estate ETF",
            "gold ETF",
            "commodity ETF",
            "agriculture ETF",
            "alternative investment",
            "private equity fund",
            "pre-IPO stock",
            "dividend-paying ETF",
            "high-yield bond",
            "municipal bond",
            "TIPS bond",
            "corporate bond",
            "global bond",
            "junk bond",
            "blue-chip stock",
            "value stock",
            "penny stock",
            "cryptocurrency exchange",
            "stablecoin",
            "DeFi platform",
            "staking platform",
            "yield farming platform",
            "NFT marketplace",
            "blockchain investment",
            "AI stock",
            "tech startup",
            "green energy investment",
            "innovation fund",
            "metaverse stock",
            "biotech stock",
            "IPO stock",
        ]
    reasoning_prompts = ["Explain the architectural differences between transformer-based language models and traditional RNNs, including specific advantages and disadvantages of each approach. Include examples of where each architecture might be more suitable", "Explain the concept of attention mechanism in deep learning and its application in natural language processing.", "Compare and contrast the performance of CNNs and RNNs on image classification tasks.", "What are the advantages and disadvantages of using transfer learning in machine learning?", "Describe the architecture of a generative adversarial network (GAN) and its applications.", "How does the choice of optimizer affect the training of a neural network?", "Explain the concept of batch normalization and its impact on neural network training.", "What is the difference between a perceptron and a multilayer perceptron?", "Discuss the role of regularization in preventing overfitting in neural networks.", "Can you explain the concept of gradient clipping and its importance in training neural networks?", "How does the learning rate affect the convergence of a neural network?", "What is the difference between a convolutional neural network (CNN) and a recurrent neural network (RNN)?", "Explain the concept of word embeddings and their application in natural language processing.", "Describe the process of training a neural network using backpropagation.", "What are the advantages and disadvantages of using a pre-trained language model?", "Compare and contrast the performance of different activation functions in neural networks.", "Explain the concept of pooling in convolutional neural networks.", "How does the choice of kernel size affect the performance of a CNN?", "What is the difference between a fully connected neural network and a convolutional neural network?", "Discuss the role of data augmentation in improving the robustness of neural networks.", "Can you explain the concept of dropout and its importance in preventing overfitting?", "How does the batch size affect the training of a neural network?", "What is the difference between a stochastic gradient descent (SGD) optimizer and an Adam optimizer?", "Explain the concept of skip connections in neural networks.", "Describe the architecture of a U-Net and its applications in image segmentation.", "What are the advantages and disadvantages of using a graph neural network?", "Compare and contrast the performance of different architectures for image classification tasks.", "Explain the concept of sequence-to-sequence models and their application in machine translation.", "How does the choice of hyperparameters affect the performance of a neural network?", "What is the difference between a variational autoencoder (VAE) and a generative adversarial network (GAN)?", "Discuss the role of reinforcement learning in training neural networks.", "Can you explain the concept of transfer learning and its application in computer vision?", "How does the choice of loss function affect the training of a neural network?", "What is the difference between a Siamese network and a triplet network?", "Explain the concept of adversarial attacks and their impact on neural network security.", "Describe the process of neural network interpretability and its importance.", "What are the advantages and disadvantages of using a neural Turing machine?", "Compare and contrast the performance of different neural network architectures for natural language processing tasks.", "Explain the concept of attention-based neural networks and their application in machine translation.", "How does the choice of evaluation metric affect the performance of a neural network?", "What is the difference between a policy gradient method and a value-based method in reinforcement learning?", "Discuss the role of multi-task learning in improving the performance of neural networks.", "Can you explain the concept of knowledge graph embedding and its application in natural language processing?", "How does the choice of optimizer affect the convergence of a neural network?", "What is the difference between a temporal convolutional network (TCN) and a long short-term memory (LSTM) network?", "Explain the concept of graph attention networks and their application in graph classification.", "Describe the architecture of a neural network for object detection tasks.", "What are the advantages and disadvantages of using a pre-trained language model for text classification tasks?", "Compare and contrast the performance of different neural network architectures for speech recognition tasks."]
    
    if test_type == "short-ttft":
        return [
            {
                "role": "system",
                "content": "you're a helpful ai assistant",
            },
            {
                "role": "user",
                "content": "What is the best {} right now?".format(
                    categories[random.randint(0, len(categories) - 1)]
                ),
                # "content": "Explain the architectural differences between transformer-based language models and traditional RNNs, including specific advantages and disadvantages of each approach. Include examples of where each architecture might be more suitable",
            },
        ]
    elif test_type == "reasoning-ttft":
        return [
            {
                "role": "system",
                "content": "you're a helpful ai assistant",
            },
            {
                "role": "user",
                "content": reasoning_prompts[random.randint(0, len(reasoning_prompts) - 1)]
            },
        ]
    else:
        raise ValueError("Invalid test type")
class AlpineAPIUser(HttpUser):
    # Random wait between requests for each user
    wait_time = between(0.1, 0.2)
    host = ALPINEX_HOST
    print("HOST: ", host)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openai_client = OpenAI(
            api_key=os.environ.get("ALPINEX_API_KEY"),
            base_url=ALPINEX_HOST,
        )

    @task
    def chat_completion(self):
        start_time = time.time()

        try:
            # Create streaming completion
            stream = self.openai_client.chat.completions.create(
                model="Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=get_random_message(test_type=SENTIENT_TEST),
                max_tokens=400 if SENTIENT_TEST == "short-ttft" else 800,
                temperature=0.7,
                stream=True,
            )

            # Get just the first chunk
            request_time = None
            for chunk in stream:
                if request_time is None:
                    request_time = int((time.time() - start_time) * 1000)
                events.request.fire(
                    request_type="POST",
                    name="chat_completion_ttft",
                    response_time=request_time,
                    response_length=0,
                    exception=None,
                    context=None,
                )
            stream.close()  # Clean up the stream

        except Exception as e:
            request_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="POST",
                name="chat_completion_ttft",
                response_time=request_time,
                response_length=0,
                exception=e,
                context=None,
            )
