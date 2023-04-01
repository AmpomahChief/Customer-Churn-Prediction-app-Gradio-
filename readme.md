# Customer Churn Prediction App with Gradio
## DESCRIPTION
This REPO contains Gradio web app built to deploy a machine learning model. This app predicts the possibility of whether a customer will churn or not. The app collects verious required inputs and returns the prediction. 

The app has a user friendly interface tha makes it easier for users to interact with it, regardless of their level of knowledge in machine learning.

## SETUP
To setup and run this project you need to have Python3 installed on your system. Then you can clone this repo. At the repo's root, use the code from below which applies:

Windows:

    python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

Linux & MacOs:

    python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
NB: For MacOs users, please install Xcode if you have an issue.

You can then run the app (still at the repository root):

RUN APP.
    
    gradio src/Gradio_app.py
    

With inbrowser = True defined, it should open a browser tab automatically. If it doesn't, type this address in your browser: http://127.0.0.1:7860/

Screenshots
![](/screenshoots/gradioapp.png)

![](/screenshoots/gradioapp1.png)


## AUTHER

- BERNARD AMPOMAH []()