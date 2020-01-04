# 基礎となるイメージを設定
FROM jupyter/tensorflow-notebook 

# 不足しているライブラリを追加インストール
RUN pip install xgboost

# Jupyterの拡張機能を有効化する
USER jovyan
RUN pip install jupyter-contrib-nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextensions_configurator enable

# Hinterlandの拡張機能はデフォルトで有効にする
RUN jupyter nbextension enable hinterland/hinterland