# # docker build -t fschlatt/authorship-verification-naive-bayes:0.0.1 .
# FROM fschlatt/natural-language-processing-exercises:0.0.1

# ADD run.py /code/run.py
# ADD model.joblib /code/model.joblib

# ENTRYPOINT [ "python3", "/code/run.py" ]
# Use an image with the necessary Python environment
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /code

# Copy the Python scripts and model joblib file into the container
COPY run.py /code/run.py
COPY model.joblib /code/model.joblib

# Install necessary Python packages
RUN pip install numpy pandas scikit-learn joblib

# Command to run the script
ENTRYPOINT ["python3", "/code/run.py"]