FROM fschlatt/natural-language-processing-exercises:0.0.1

COPY train.py /code/train.py
COPY run.py /code/run.py
COPY levenshtein.py /code/levenshtein.py

WORKDIR /code

# Run train.py to generate best_threshold.txt during the build process
RUN python3 train.py

ENTRYPOINT ["python3", "run.py"]
