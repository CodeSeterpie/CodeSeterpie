# 基礎となるイメージを設定
FROM jupyter/tensorflow-notebook 

# 不足しているライブラリを追加インストール
RUN pip install xgboost
RUN pip install pandas_profiling
RUN pip install lime
RUN conda install -c conda-forge lightgbm
