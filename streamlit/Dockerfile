# FROM python:3.8-slim-buster
FROM python:3.8.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]