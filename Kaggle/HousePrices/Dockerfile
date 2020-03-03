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
    conda install -c conda-forge xgboost && \
    conda install -c conda-forge lightgbm && \
    conda install -c conda-forge pandas-profiling && \
    conda install -c conda-forge lime && \
    conda install -c conda-forge python-graphviz 

# リント機能とコード自動整形機能の追加
RUN conda install -c conda-forge flake8 && \
    jupyter labextension install jupyterlab-flake8

RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter && \
    conda install -c conda-forge jupyterlab_code_formatter && \
    jupyter serverextension enable --py jupyterlab_code_formatter && \
    conda install -c conda-forge autopep8
