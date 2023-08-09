# University Chat Bot

The University Chat Bot is an AI-powered chatbot designed to provide information and answer questions related to university-related topics. It uses natural language processing and retrieval-based QA to offer accurate responses based on a user's queries and historical conversation.

## Getting Started

To use the University Chat Bot, follow these steps:

### Prerequisites

1. Python 3.6 or later installed.
2. Pip package manager installed.

### Installation

1. Clone this repository to your local machine:



2. Navigate to the project directory:


3. Install the required Python packages:




### Usage

1. Start the application by running the following command:
<pre>
    git clone "https://github.com/theuppercaseguy/Univ-Chat-Bot"
    cd "Univ-Chat-Bot"
    pip install -r requirements.txt
    cd Project
    python -m uvicorn main2:app
</pre>



This will launch the chatbot application and make it accessible at `http://localhost:8000`.

##### use the following link to chat with the api using UI
<pre>
    http://localhost:8000/chat_me
</pre>

##### use the following link to chat with the api using potman or any other software

###### importan: it accepts a List of dictionary with the following variables:

<pre>
[
  {
    "content": "nice name",
    "role":"user"
  }
]
</pre>

<pre>
    http://localhost:8000/chat
    
</pre>


2. Open your web browser and navigate to `http://localhost:8000/chat_me`.

3. You can interact with the chatbot by typing your questions or queries in the input box and clicking "Send".

4. The chatbot will provide responses based on the provided input and historical conversation.

### Customization

- You can customize the chatbot's behavior by adjusting the conversation history and chat models in the code.

- To modify the conversation history, navigate to the `get_user_history` function in the `main.py` file.

- To adjust the chat models or retrieval mechanisms, refer to the imported modules in the code and update their configurations accordingly.

### Limitations

- The chatbot's response may vary based on the availability of relevant data and the quality of the conversation history.

- The maximum token limit of the language model is a consideration when managing conversation history.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).















