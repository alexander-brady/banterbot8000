# Meet Banter Bot 8000
<img src="assets/banterbot8000.png" align="left" width="130" style="margin-right: 25px;"/> 
Banter Bot 8000 isn't just your usual chatbot. Not only is he extremely funny and goofy, but he capable of explaining any joke. Don't believe me? Ask him to explain a joke, and he'll blow your mind. And no, Banter Bot 8000 is (probably) not just regurgitating a LLM response, but utilizes NLP technologies to classify the type of pun, be it homographic or homophonic, and finds the corresponding definition or similar sounding word to clarify any joke.

## How to use
1. Create and activate your virtual environment.
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Download all dependencies.
```sh
pip install -r requirements.txt
```
3. Download a compatible LLM (must work with Llama.cpp). I used Llama 3.2-3B (GGUF) from this [repo](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF). Don’t forget to update the model path in `config.yaml` under `llm.model`.
4. Open `banterbot8000.ipynb`, run the initialization cell, then happy chatting!
