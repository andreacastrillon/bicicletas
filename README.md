houseprice
==============================

A short description of the project.

Project Organization
------------

    ├── README.md               <- The top-level README for developers using this project.
    │
    ├── docs                    <- A place to store documentation for the project
    │
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                              the creator's initials, and a short `-` delimited description, e.g.
    │                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── modelling               <- Source code to develop the models and generate reports
    │   ├── requirements.txt    <- Hold requirements for the project
    │   ├── __init__.py         <- Makes modelling a Python module
    │   │
    │   ├── data.py             <- Objects to download or generate data
    │   │
    │   ├── models.py           <- Objects to define, train models and then use trained models to make
    │   │                          predictions
    │   │
    │   ├── visualize.py        <- Objects to create exploratory and results oriented visualizations
    │   │    
    │   └── app.py              <- Application to execute everything from the command line
    │
    ├── service                 <- Source code to expose your model as a service
    │   ├── requirements.txt    <- Makes src a Python module
    │   ├── __init__.py         <- Makes service a Python module
    │   │
    │   └── app.py              <- Objects to download or generate data
    │
    ├── tests                   <- Where you check that your code actually works as expected
    │
    └── setup.cfg               <- File holding settings for your project


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## RUN APP ON WINDOWS
''''
#Instalar dependencias
pip install -r ./service/requirements.txt

#Entrenar el modelo
python .\modelling\app.py train .\config.yml

#Evaluar el modelo
python .\modelling\app.py eval .\config.yml "2021-04-01 04_36_00+00_00"


#Registrar variables (Editor de registro)
SERIALIZED_MODEL_PATH=C:\Users\Asus\Documents\Development\WorkSpaces\ecad\machine-learning-ii\demo\models\2021-03-22 02_26_00+00_00\model.joblib
MODEL_LIB_DIR=C:\Users\Asus\Documents\Development\WorkSpaces\ecad\machine-learning-ii\demo\modellling

#Run server
uvicorn service.app:app
''''
