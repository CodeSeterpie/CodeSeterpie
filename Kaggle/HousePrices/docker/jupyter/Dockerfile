# 基礎となるイメージを設定
FROM jupyter/tensorflow-notebook 

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    graphviz \
    cmake \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 不足しているライブラリを追加インストール
USER $NB_UID
RUN conda install --quiet --yes \
    'xgboost' \
    'lightgbm' \
    'pandas-profiling' \
    'lime' \
    'python-graphviz' \
    'PyMySQL' \
    'sqlalchemy' \
    'tqdm' \
    && \
    conda clean --all -f -y 

# Jupyterの拡張機能をインストール
#   ・リント機能
#   ・コード自動整形機能
RUN conda install --quiet --yes \
    'flake8' \
    'autopep8' \
    'jupyterlab_code_formatter=1.1.0' \
    && \
    conda clean --all -f -y && \
    jupyter labextension install jupyterlab-flake8 --no-build && \
    jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build && \
    jupyter serverextension enable --py jupyterlab_code_formatter && \
    jupyter lab build && \
    jupyter lab clean
