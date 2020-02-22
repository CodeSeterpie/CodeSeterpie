# 基礎となるイメージを設定
FROM jupyter/tensorflow-notebook 

# 不足しているライブラリを追加インストール
RUN pip install xgboost
RUN pip install pandas_profiling
RUN pip install lime
RUN conda install -c conda-forge lightgbm

# リント機能の追加
RUN pip install flake8
RUN jupyter labextension install jupyterlab-flake8

# コード自動整形機能の追加
#RUN pip install black
#RUN pip install yapf
RUN pip install autopep8
RUN pip install jupyterlab_code_formatter
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter
RUN jupyter serverextension enable --py jupyterlab_code_formatter
