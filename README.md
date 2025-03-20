# ThinkRAG

This repository contains the implementation of the RAG (Retrieval-Augmented Generation) approach along with Reasoning, as described in the paper "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions" by ([authors](https://arxiv.org/pdf/2212.10509)).

The steps to follow for setting up and running the project are outlined below:

1. Create a virtual environment:
   - Run the command `python -m venv my_venv` to create a virtual environment.
   - Activate the virtual environment using `source my_venv/bin/activate`.

2. Install required packages:
   - Run the command `pip install -r requirement.txt` to install the required packages.

3. Create an .env file:
   - Create a new file named `.env` in the root directory of the project.
   - Add your GROQ key to the `.env` file in the format `GROQ_KEY=your_groq_key`.

4. Run the main script:
   - Run the command `python main.py` to start the project.

Note: You can connect any external source of data by modifying the relevant code.

Future direction:
- Can add query classifiers that categorizes incoming questions as either "explicit fact queries" or "complex reasoning queries.
- The project can be extended to incorporate external tools for additional functionality.
