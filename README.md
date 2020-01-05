# Kaggle
## 環境構築
このグループでKaggleを行うにあたり、必要な環境を構築します。実行環境としてはDockerを用いたコンテナにjupyter notebookが動く環境を構築します。

### 事前に準備する物
__【注意】__ インストーラなどの容量が大きいファイル（数GB程度）をダウンロードするため、家などのネットワークがリッチな環境で行ってください。
* Kaggleアカウントの取得(Googleアカウントでログイン可能)
  * https://www.kaggle.com/
* Githubアカウントの取得
  * https://github.com/
* Dockerアカウントの取得
  * https://hub.docker.com/
* Dockerのインストール
  * Windows 10 Pro, Macの場合はDocker Desktopをインストールする。
    * https://www.docker.com/products/docker-desktop 
  * Windows 10 Homeの場合はDocker Toolboxをインストールする。
    * インストール手順
      1. https://github.com/docker/toolbox/releases
      2. 上のリンクから最新のVersionのexeファイルをダウンロードして、ローカルでダブルクリックする。
      3. インストールが完了したら、Docker QuickStart Terminalを起動する。自動で初期設定が行われるため完了するまで待つ。
    * トラブルシューティング
      * 手順の詳細は[ここ](https://docs.docker.com/toolbox/toolbox_install_windows/)を参照。
* jupyter notebookのコンテナイメージの取得
  * Dockerインストール後に、下のコマンドを実行してコンテナイメージを取得する。
  * ``` docker pull jupyter/tensorflow-notebook ```
* SourceTreeのインストール
  * https://www.sourcetreeapp.com/