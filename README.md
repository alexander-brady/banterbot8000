# Meet Banter Bot 8000
<img src="./banterbot8000.png" align="left" width="180" style="margin-right: 25px;"/> 
Banter Bot 8000 isn't just your usual chatbot. Not only is he extremely funny and goofy, but he capable of explaining any joke. Don't believe me? Ask him to explain a joke, and he'll blow your mind. And no, Banter Bot 8000 is (probably) not just regurgitating a LLM response, but utilizes NLP technologies to classify the type of pun, be it homographic or homophonic, and finds the corresponding definition or similar sounding word to clarify any joke.

## How to use
1. Create and activate your virtual environment `python -m venv venv`.
2. Download all dependencies `pip install -r requirements.txt`.
3. Download your LLM (must be compatible with Llama.cpp). I utilized Llama 3.2-3B as a GGUF from [here](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF). Make sure to update the path in `config.yaml/llm/model`!
4. Run `banterbot8000.ipynb`. Run the initialization cell, then happy chatting!