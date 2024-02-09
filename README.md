# End-to-end Machine Learning Project - Disaster Tweet Detection

![mlops-disaster-tweet-detection](https://github.com/Juwono136/disaster-tweet-detection-mlops/assets/70443393/7581cf3b-4f5c-43a7-b3e6-b42f7f75cfba)


*You can see and run `testing.ipynb` file for testing and make prediction request to model serving.* 😎🙃👍


|                         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                 | For this project, the dataset used is from the [Natural Language Processing with Disaster Tweets Dataset]( https://www.kaggle.com/competitions/nlp-getting-started/data). The dataset used consists of over 7,000 manually classified tweets that cover the categories of disaster and non-disaster tweets. Each tweet is associated with a label of 0 or 1, where 1 indicates that the tweet is a disaster, and 0 indicates that the tweet is not a disaster.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Problem                 | Twitter, now more commonly known as the X app, is one of the most widely used social media apps today. It has become an important communication channel for people all over the world. Almost all information that occurs in real life is now discussed on the app, including emergency situations. The availability of smartphones allows people to announce emergency situations they are experiencing in real-time. Therefore, more and more institutions or organizations are interested in monitoring emergency situations through Twitter in a programmed manner and need the program to respond to disaster situations more quickly, especially in emergencies. This project aims to address this issue by creating a machine learning system that can improve emergency response by automatically identifying tweet sentences that contain disaster information and non-disaster information.                                                                                                                                                                                                                                                                                                                                                                      |
| Machine learning solution | The machine learning solution used in this project involves [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) as a framework for building and managing machine learning pipelines, and [Apache Beam](https://beam.apache.org/) as a pipeline orchestrator. TFX provides components for data preprocessing, model training, evaluation, and model serving. In this project, TFX components are utilized in several stages of the machine learning pipeline. The data processing stage employs the ExampleGen component for data ingestion, the StatisticsGen, SchemaGen, and ExampleValidator components for data validation, and the Transform component for data preprocessing. The model development and validation stage employs the Trainer, Tuner, Resolver, and Evaluator components. For deployment and model serving, the Pusher component is employed to move the model created using a docker image containing the TF-Serving (TensorFlow Serving) model to Railway, a cloud platform for accessing the model via HTTP Request. The entire process is executed using an Apache Beam pipeline orchestrator. This model is subsequently utilized in the production environment to classify tweets as either disaster or non-disaster. |
| Data processing methods       | Data processing involves several steps, including cleaning email text, removing special characters, converting letters to lowercase using regular expressions, vectorizing text with the TextVectorization layer from TensorFlow, and utilizing word embeddings. Additionally, feature selection and handling imbalanced data are addressed by partitioning the data into training and validation sets with a ratio of 8:2, and also by identifying labels and features from the dataset. The data processing process encompasses data ingestion to convert data into a standardized format, data splitting into training and validation sets, data validation to detect anomalies, and data preprocessing to ready the data for training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Model architecture        | The model architecture utilized in this project encompasses deep learning models. It involves the Text Vectorization process utilizing the Embedding layer to convert word token representations into numeric vectors with reduced dimensions. Additionally, the Global Average Pooling Layer is employed to further reduce data dimensions, enhancing model efficiency and preventing overfitting. The Hidden layer comprises Dense layers aimed at extracting complex features and modeling nonlinear relationships within the data. Given that the project addresses binary classification, specifically categorizing tweet sentences into two labels (disaster and not disaster), the sigmoid activation function is employed to generate output, representing the probability of the positive class (disaster) or the negative class (not disaster). This model architecture is seamlessly integrated into the TFX pipeline by storing it as a function within the trainer module file, facilitating structured training and evaluation using the Trainer and Tuner components.                                                                                                                                                                                                                                                              |
| Evaluation metrics         | The evaluation metrics utilized involve assessing the model's performance post-training within the Trainer component. These metrics encompass precision and recall values to gauge the classification model's effectiveness, as well as the AUC (Area Under Curve) value to compare its ability to differentiate between positive and negative classes. Given the project's focus on tweet sentence detection, particular attention is paid to the false positive (tweets inaccurately classified as disasters) and false negative (disaster tweets missed by the model) values. Furthermore, TruePositives represent the instances correctly classified as positive (disaster tweets), while TrueNegatives signify correctly classified negative instances (non-disaster tweets). These metrics' configuration is established using the TFMA (TensorFlow Model Analysis) library as part of the input for the Evaluator component within the TFX pipeline, facilitating the model evaluation process.                                                                                                                                                                                                                                                                                                   |
| Model performance          | The model's performance was assessed using the previously mentioned evaluation metrics. From a total of 1542 tweet examples in the evaluation set, the False Negative value was 241, indicating that 241 actual positive cases were inaccurately classified as negative by the model. The False Positive value was 121, indicating that 121 actual negative cases were wrongly classified as positive by the model. On the other hand, the True Negative value was 736, indicating that the model correctly identified 736 instances that did not belong to the positive category, and the True Positive value was 444, indicating that the model accurately identified 444 cases that did belong to the positive category. Additionally, the model training yielded a binary accuracy value of 0.765 or approximately 76.5%, indicating that the model accurately predicted the target for most of the data with a reasonably high level of accuracy.                                                                                                                                                                                                                                                                                              |
| Deployment options         | In this project, the chosen deployment option involves utilizing a cloud platform known as [Railway](https://railway.app/). Railway is a Platform-as-a-Service (PaaS) offering that facilitates application deployment and supports multiple programming languages, including Python. The trained model is stored within a folder named serving_model through the pipeline orchestration process. Initially, a docker image is created to encapsulate the TF-Serving model, following which the deployment procedure is executed using the Railway platform. Subsequently, the model becomes accessible via HTTP Requests, enabling testing and prediction requests to be performed seamlessly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Web app                 | Here is the web app link used to access the serving model: [mlops-disaster-tweets](https://mlops-disaster-tweets-production.up.railway.app/v1/models/tweets-model/metadata)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Monitoring              | For model serving monitoring, we utilize a platform called [Prometheus](https://prometheus.io/) to view metrics from the running model. Grafana serves as an open-source platform to visualize model metrics captured by Prometheus, displaying them in an attractive interactive dashboard format. Various metrics can be displayed by inputting available PromQL commands. In this project, the metrics showcased on the Grafana dashboard include Runtime Latency Sum, Request Count, and Request Latency Bucket. Runtime Latency Sum provides an overview of how fast or slow the model responds to requests, Request Count indicates the number of requests received by the model or offers an overview of the workload level or model popularity, while Request Latency Bucket offers detailed insights into the distribution of the model request response time.                                                                                                                                                                                                                                                                                                                                                                                                          |

-----------------------

### 🧑‍💻 Technologies:
- ➡️ Python 3.9.x
- ➡️ TFX (TensorFlow Extended)
- ➡️ TensorFlow
- ➡️ Apache Beam
- ➡️ Railway (https://railway.app/)
- ➡️ Prometheus
- ➡️ Grafana


### 🖥️ Requirements:
- ➡️ Python 3.9.15
- ➡️ Conda
- ➡️ [Docker](https://www.docker.com/products/docker-desktop/)
- ➡️ tfx 1.11.0


### 🛠️ Installation and setup:
- Create virtual environment using Anaconda prompt:
```
conda update conda
conda create --name <env_name> python==3.9.15
```

- Install package using `pip`:
```
pip --default-timeout=5000 --use-deprecated=legacy-resolver install -r requirements.txt
```

- Create TF-Serving in docker image:
```
docker build -t disaster-tweet-tf-serving .
```

- Run docker image locally:
```
docker run -p 8080:8501 disaster-tweet-tf-serving
```

- Install grafana:
  - **Windows**: https://grafana.com/docs/grafana/latest/setup-grafana/installation/windows/
  - **Debian or Ubuntu**: https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/

### 🤝 Project Members
- Juwono

### 📸 Screenshot application
- Model plot:

![model_plot](https://github.com/Juwono136/disaster-tweet-detection-mlops/assets/70443393/53234203-6995-4f76-bf84-74ef66173c0a)


- Metadata deployment file:

![tweet-deployment](https://github.com/Juwono136/disaster-tweet-detection-mlops/assets/70443393/9c17db9f-1bda-491d-83be-67c8ff5cbfab)


- Prometheus dashboard:

![tweet-monitoring](https://github.com/Juwono136/disaster-tweet-detection-mlops/assets/70443393/a6a2c0f3-c859-4520-a0c8-f9efb1507349)


- Grafana dashboard:

![tweet-grafana-dashboard](https://github.com/Juwono136/disaster-tweet-detection-mlops/assets/70443393/cb0bb33b-4b0d-4fd4-abea-202da133012e)

