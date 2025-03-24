{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mlflow\n",
    "import pandas as pd\n",
    "import mlflow.sklearn\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "import pandas as pd\n",
    "import re\n",
    "import string\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "import numpy as np\n",
    "import dagshub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Accessing as sudarshansahane1044\n",
       "</pre>\n"
      ],
      "text/plain": [
       "Accessing as sudarshansahane1044\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Initialized MLflow to track repo <span style=\"color: #008000; text-decoration-color: #008000\">\"sudarshansahane1044/mlopsminiproject\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "Initialized MLflow to track repo \u001b[32m\"sudarshansahane1044/mlopsminiproject\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">Repository sudarshansahane1044/mlopsminiproject initialized!\n",
       "</pre>\n"
      ],
      "text/plain": [
       "Repository sudarshansahane1044/mlopsminiproject initialized!\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "dagshub.init(repo_owner='sudarshansahane1044', repo_name='mlopsminiproject', mlflow=True)\n",
    "\n",
    "mlflow.set_tracking_uri('https://dagshub.com/sudarshansahane1044/mlopsminiproject.mlflow')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sentiment</th>\n",
       "      <th>content</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>empty</td>\n",
       "      <td>@tiffanylue i know  i was listenin to bad habi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>sadness</td>\n",
       "      <td>Layin n bed with a headache  ughhhh...waitin o...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>sadness</td>\n",
       "      <td>Funeral ceremony...gloomy friday...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>enthusiasm</td>\n",
       "      <td>wants to hang out with friends SOON!</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>neutral</td>\n",
       "      <td>@dannycastillo We want to trade with someone w...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    sentiment                                            content\n",
       "0       empty  @tiffanylue i know  i was listenin to bad habi...\n",
       "1     sadness  Layin n bed with a headache  ughhhh...waitin o...\n",
       "2     sadness                Funeral ceremony...gloomy friday...\n",
       "3  enthusiasm               wants to hang out with friends SOON!\n",
       "4     neutral  @dannycastillo We want to trade with someone w..."
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# data preprocessing\n",
    "\n",
    "# Define text preprocessing functions\n",
    "def lemmatization(text):\n",
    "    \"\"\"Lemmatize the text.\"\"\"\n",
    "    lemmatizer = WordNetLemmatizer()\n",
    "    text = text.split()\n",
    "    text = [lemmatizer.lemmatize(word) for word in text]\n",
    "    return \" \".join(text)\n",
    "\n",
    "def remove_stop_words(text):\n",
    "    \"\"\"Remove stop words from the text.\"\"\"\n",
    "    stop_words = set(stopwords.words(\"english\"))\n",
    "    text = [word for word in str(text).split() if word not in stop_words]\n",
    "    return \" \".join(text)\n",
    "\n",
    "def removing_numbers(text):\n",
    "    \"\"\"Remove numbers from the text.\"\"\"\n",
    "    text = ''.join([char for char in text if not char.isdigit()])\n",
    "    return text\n",
    "\n",
    "def lower_case(text):\n",
    "    \"\"\"Convert text to lower case.\"\"\"\n",
    "    text = text.split()\n",
    "    text = [word.lower() for word in text]\n",
    "    return \" \".join(text)\n",
    "\n",
    "def removing_punctuations(text):\n",
    "    \"\"\"Remove punctuations from the text.\"\"\"\n",
    "    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)\n",
    "    text = text.replace('؛', \"\")\n",
    "    text = re.sub('\\s+', ' ', text).strip()\n",
    "    return text\n",
    "\n",
    "def removing_urls(text):\n",
    "    \"\"\"Remove URLs from the text.\"\"\"\n",
    "    url_pattern = re.compile(r'https?://\\S+|www\\.\\S+')\n",
    "    return url_pattern.sub(r'', text)\n",
    "\n",
    "def normalize_text(df):\n",
    "    \"\"\"Normalize the text data.\"\"\"\n",
    "    try:\n",
    "        df['content'] = df['content'].apply(lower_case)\n",
    "        df['content'] = df['content'].apply(remove_stop_words)\n",
    "        df['content'] = df['content'].apply(removing_numbers)\n",
    "        df['content'] = df['content'].apply(removing_punctuations)\n",
    "        df['content'] = df['content'].apply(removing_urls)\n",
    "        df['content'] = df['content'].apply(lemmatization)\n",
    "        return df\n",
    "    except Exception as e:\n",
    "        print(f'Error during text normalization: {e}')\n",
    "        raise    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sentiment</th>\n",
       "      <th>content</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>empty</td>\n",
       "      <td>tiffanylue know listenin bad habit earlier sta...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>sadness</td>\n",
       "      <td>layin n bed headache ughhhh waitin call</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>sadness</td>\n",
       "      <td>funeral ceremony gloomy friday</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>enthusiasm</td>\n",
       "      <td>want hang friend soon</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>neutral</td>\n",
       "      <td>dannycastillo want trade someone houston ticke...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    sentiment                                            content\n",
       "0       empty  tiffanylue know listenin bad habit earlier sta...\n",
       "1     sadness            layin n bed headache ughhhh waitin call\n",
       "2     sadness                     funeral ceremony gloomy friday\n",
       "3  enthusiasm                              want hang friend soon\n",
       "4     neutral  dannycastillo want trade someone houston ticke..."
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = normalize_text(df)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['sentiment'] = df['sentiment'].replace({'sadness':0, 'happiness':1})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025/03/24 08:06:44 INFO mlflow.tracking.fluent: Experiment with name 'Bow vs tfidf' does not exist. Creating a new experiment.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Experiment: artifact_location='mlflow-artifacts:/372a7ce5d7f04cc899ee9a93bebdb07e', creation_time=1742783804062, experiment_id='3', last_update_time=1742783804062, lifecycle_stage='active', name='Bow vs tfidf', tags={}>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mlflow.set_experiment(\"Bow vs tfidf\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "vectorizer = {\n",
    "    'Bow':CountVectorizer(),\n",
    "    'TF-IDF':TfidfTransformer()\n",
    "\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "algorithms = {\n",
    "\n",
    "    'LR' : LogisticRegression(),\n",
    "    'Multinomialind': MultinomialNB(),\n",
    "    'Rf':RandomForestClassifier(),\n",
    "    'GB':GradientBoostingClassifier()\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🏃 View run All Experiments at: https://dagshub.com/sudarshansahane1044/mlopsminiproject.mlflow/#/experiments/3/runs/55ac28143b2a44fdbaeb51c2c5cc3302\n",
      "🧪 View experiment at: https://dagshub.com/sudarshansahane1044/mlopsminiproject.mlflow/#/experiments/3\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'vectorizers' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[15], line 4\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m mlflow\u001b[38;5;241m.\u001b[39mstart_run(run_name\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mAll Experiments\u001b[39m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;28;01mas\u001b[39;00m parent_run:\n\u001b[0;32m      2\u001b[0m     \u001b[38;5;66;03m# Loop through algorithms and feature extraction methods (Child Runs)\u001b[39;00m\n\u001b[0;32m      3\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m algo_name, algorithm \u001b[38;5;129;01min\u001b[39;00m algorithms\u001b[38;5;241m.\u001b[39mitems():\n\u001b[1;32m----> 4\u001b[0m         \u001b[38;5;28;01mfor\u001b[39;00m vec_name, vectorizer \u001b[38;5;129;01min\u001b[39;00m \u001b[43mvectorizers\u001b[49m\u001b[38;5;241m.\u001b[39mitems():\n\u001b[0;32m      5\u001b[0m             \u001b[38;5;28;01mwith\u001b[39;00m mlflow\u001b[38;5;241m.\u001b[39mstart_run(run_name\u001b[38;5;241m=\u001b[39m\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00malgo_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m with \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mvec_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m, nested\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m) \u001b[38;5;28;01mas\u001b[39;00m child_run:\n\u001b[0;32m      6\u001b[0m                 X \u001b[38;5;241m=\u001b[39m vectorizer\u001b[38;5;241m.\u001b[39mfit_transform(df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcontent\u001b[39m\u001b[38;5;124m'\u001b[39m])\n",
      "\u001b[1;31mNameError\u001b[0m: name 'vectorizers' is not defined"
     ]
    }
   ],
   "source": [
    "with mlflow.start_run(run_name=\"All Experiments\") as parent_run:\n",
    "    # Loop through algorithms and feature extraction methods (Child Runs)\n",
    "    for algo_name, algorithm in algorithms.items():\n",
    "        for vec_name, vectorizer in vectorizers.items():\n",
    "            with mlflow.start_run(run_name=f\"{algo_name} with {vec_name}\", nested=True) as child_run:\n",
    "                X = vectorizer.fit_transform(df['content'])\n",
    "                y = df['sentiment']\n",
    "                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "                \n",
    "\n",
    "                # Log preprocessing parameters\n",
    "                mlflow.log_param(\"vectorizer\", vec_name)\n",
    "                mlflow.log_param(\"algorithm\", algo_name)\n",
    "                mlflow.log_param(\"test_size\", 0.2)\n",
    "                \n",
    "                # Model training\n",
    "                model = algorithm\n",
    "                model.fit(X_train, y_train)\n",
    "                \n",
    "                # Log model parameters\n",
    "                if algo_name == 'LogisticRegression':\n",
    "                    mlflow.log_param(\"C\", model.C)\n",
    "                elif algo_name == 'MultinomialNB':\n",
    "                    mlflow.log_param(\"alpha\", model.alpha)\n",
    "                elif algo_name == 'XGBoost':\n",
    "                    mlflow.log_param(\"n_estimators\", model.n_estimators)\n",
    "                    mlflow.log_param(\"learning_rate\", model.learning_rate)\n",
    "                elif algo_name == 'RandomForest':\n",
    "                    mlflow.log_param(\"n_estimators\", model.n_estimators)\n",
    "                    mlflow.log_param(\"max_depth\", model.max_depth)\n",
    "                elif algo_name == 'GradientBoosting':\n",
    "                    mlflow.log_param(\"n_estimators\", model.n_estimators)\n",
    "                    mlflow.log_param(\"learning_rate\", model.learning_rate)\n",
    "                    mlflow.log_param(\"max_depth\", model.max_depth)\n",
    "                \n",
    "                # Model evaluation\n",
    "                y_pred = model.predict(X_test)\n",
    "                accuracy = accuracy_score(y_test, y_pred)\n",
    "                precision = precision_score(y_test, y_pred)\n",
    "                recall = recall_score(y_test, y_pred)\n",
    "                f1 = f1_score(y_test, y_pred)\n",
    "                \n",
    "                # Log evaluation metrics\n",
    "                mlflow.log_metric(\"accuracy\", accuracy)\n",
    "                mlflow.log_metric(\"precision\", precision)\n",
    "                mlflow.log_metric(\"recall\", recall)\n",
    "                mlflow.log_metric(\"f1_score\", f1)\n",
    "                \n",
    "                # Log model\n",
    "                mlflow.sklearn.log_model(model, \"model\")\n",
    "                \n",
    "                # Save and log the notebook\n",
    "                mlflow.log_artifact(__file__)\n",
    "                \n",
    "                # Print the results for verification\n",
    "                print(f\"Algorithm: {algo_name}, Feature Engineering: {vec_name}\")\n",
    "                print(f\"Accuracy: {accuracy}\")\n",
    "                print(f\"Precision: {precision}\")\n",
    "                print(f\"Recall: {recall}\")\n",
    "                print(f\"F1 Score: {f1}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
