# docker build -t fschlatt/authorship-verification-naive-bayes:0.0.1 .
#FROM fschlatt/natural-language-processing-exercises:0.0.1

#ADD run.py /code/run.py
#ADD model.joblib /code/model.joblib
# Define a version or timestamp label
#ARG VERSION=latest
#LABEL version=$VERSION
#ENTRYPOINT [ "python3", "/code/run.py" ]
# Use an existing custom image that presumably has all necessary Python and ML dependencies installed
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Work directory setup for better organization and clarity
WORKDIR /code

# Adding files to the container
ADD run.py /code/
ADD model.joblib /code/

# Define a version or timestamp label
ARG VERSION=latest
LABEL version=$VERSION

# Set the ENTRYPOINT to run your application
ENTRYPOINT ["python3", "/code/run.py"]
