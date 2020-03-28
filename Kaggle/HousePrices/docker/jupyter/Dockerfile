# 基礎となるイメージを設定
FROM jupyter/tensorflow-notebook 

USER root
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    graphviz \
    cmake

# 不足しているライブラリを追加インストール
USER $NB_UID
RUN conda update -n base conda && \
    conda install xgboost && \
    conda install lightgbm && \
    conda install pandas-profiling && \
    conda install lime && \
    conda install python-graphviz 

# リント機能とコード自動整形機能の追加
RUN conda install flake8 && \
    jupyter labextension install jupyterlab-flake8

RUN conda install autopep8
RUN conda install jupyterlab_code_formatter=1.1.0 && \
    jupyter labextension install @ryantam626/jupyterlab_code_formatter && \
    jupyter serverextension enable --py jupyterlab_code_formatter

