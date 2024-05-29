FROM fschlatt/natural-language-processing-exercises:0.0.1

COPY train.py /code/train.py
COPY run.py /code/run.py
COPY levenshtein.py /code/levenshtein.py

WORKDIR /code

ENTRYPOINT ["python3", "run.py"]
