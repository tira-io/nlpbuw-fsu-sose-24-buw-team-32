# # docker build -t fschlatt/authorship-verification-naive-bayes:0.0.1 .
# FROM fschlatt/natural-language-processing-exercises:0.0.1

# ADD run.py /code/run.py
# ADD model.joblib /code/model.joblib

# ENTRYPOINT [ "python3", "/code/run.py" ]


# Base image
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Set the working directory
WORKDIR /code

# Add necessary files
ADD run.py /code/run.py
ADD model.joblib /code/model.joblib

# If requirements are not included in the base image, uncomment the following lines:
# ADD requirements.txt /code/requirements.txt
# RUN pip install -r /code/requirements.txt

# Entry point for the container
ENTRYPOINT ["python3", "/code/run.py"]
