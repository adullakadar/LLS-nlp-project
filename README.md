LeftLaneSunny
Comparison AI agent

The program scrapes data from youtube comments and youtube transcripts, storing them in a csv (mongo integration will be done later). The data is cleaned and extracted of it's useful componenets such as sentiments, keywords, entities, topic modelling and embeddings. The metadata is then fed to the agent pipeline and allow the user to ask questions through a streamlit interface. The user can also review comments by searching and filtering through an additional interface and also view metrics in a general perspective.

architecture:
.LLS-NLP-PROJECt/
  -faiss_store -> vector storage
  -mlruns -> model testing data
  -model_data -> any relevant data created in the pipeline (raw comments, transcripts, meta_data, etc.)
  -testing_nbs -> evolution of testing notebooks, deprecated
  -app.py -> the main program used to run streamlit
  -data_extraction_main.ipynb -> the main notebook used to extract data from youtube videos, process them and store them in model_data
  -requirements.txt -> modules used for the program

to use the pipeline:
1) follow setup instructions
2) get youtube video ids from youtube and store as variables in data_extraction_main.ipynb
3) run textraction for all video_ids and process all.
4) merge all cleaned dataframes into one dataframe at the end.
Note) Steps will be explained in the notebook

Setup: 
1) Install python 3.11.9
note: make sure you are in the project folder in cmd or terminal (very very very common mistake that i absolutely did not make at least 3 times).
2) Create a python environment (3.11.9) and activate it
note2: in vsc make sure your python interpretor is running the venv python, and not global or any other environment, also a common mistake.
3) Install packages from requirements.txt (pip install -r requirements.txt)
4) also do this: python -m spacy download en_core_web_sm
5) If you want to run the code yourself, obtain a youtube API connection key and insert into .env (YT_API_KEY='_')
6) important on vscode if u want to run the code yourself: downgrade jupyter extension to 2025.8 if running code through notebooks
7) have ollama installed and running, pull embeddinggemma and qwen3:8b
8) the code currently does have mongodb integration but it does not use it right now, so no mongo connection string is required but this may change in the future.

To run the program:
in console with ollama running, run streamlit run app.py
you can also view metrics with running mlflow ui but its a big buggy
