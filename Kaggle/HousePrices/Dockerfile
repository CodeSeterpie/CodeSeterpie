# 基礎となるイメージを設定
FROM jupyter/tensorflow-notebook 

# apt-get update後にインストールするとエラーするため、先にXGBoostをインストール
RUN pip install xgboost 

USER root
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    graphviz \
    cmake

# 不足しているライブラリを追加インストール
USER $NB_UID
RUN pip install pandas_profiling && \ 
    pip install lime && \
    pip install lightgbm && \
    pip install graphviz

# リント機能とコード自動整形機能の追加
RUN pip install flake8 && \
    jupyter labextension install jupyterlab-flake8

RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter && \
    pip install jupyterlab_code_formatter && \
    jupyter serverextension enable --py jupyterlab_code_formatter && \
    pip install autopep8

