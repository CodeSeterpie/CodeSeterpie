# 環境構築
このグループでKaggleを行うにあたり、必要な環境を構築します。実行環境としてはDockerを用いたコンテナにjupyter notebookが動く環境を構築します。

## 事前準備
__【注意】__ インストーラなどの容量が大きいファイル（数GB程度）をダウンロードするため、家などのネットワークがリッチな環境で行ってください。
* Kaggleアカウント取得([ここ](https://www.kaggle.com/)から)
* Githubアカウント取得([ここ](https://github.com/)から)
* Dockerアカウント取得([ここ](https://hub.docker.com/)から)
* Dockerのインストール(下のDockerのインストール方法を参照)
* SourceTreeのインストール(ダウンロードは[ここ](https://www.sourcetreeapp.com/)から)
* Jupyter notebookのコンテナイメージの取得
  * Dockerインストール後に、下のコマンドを実行してコンテナイメージを取得する。
  
    ```
    docker pull jupyter/tensorflow-notebook
    ```
### Dockerのインストール方法
#### Windows 10 Pro, Macの場合
Docker Desktopをインストールする。 インストーラのダウンロードは[ここ](https://www.docker.com/products/docker-desktop)から。 
#### Windows 10 Homeの場合
Docker Toolboxをインストールする。単純なインストールではJupyter notebookが動作しないため、下の手順でインストールする。

インストール手順
1. [ここ](https://github.com/docker/toolbox/releases)から最新のVersionのexeファイルをダウンロードして、ローカルでダブルクリックする。
1. インストールが完了したら、Docker QuickStart Terminalを起動する。自動で初期設定が行われるため完了するまで待つ。
1. Oracle VM VirtualBoxマネージャーを開き、 今回作成した仮想環境(名前はdefault)を選択後、設定をクリックして設定画面を開く。
1. ネットワークを選択し、高度をクリックする。ポートフォワーディングボタンが表示されるので、クリックしてポートフォワーディングルール画面を開く。
1. プラスボタンをクリックしてルールを追加する。設定は下の通り。

    |名前|プロトコル|ポストIP|ポストポート|ゲストIP|ゲストポート|
    |:---|:---|:---|:---|:---|:---|
    |任意|TCP|127.0.0.1|8888|空白|8888|

トラブルシューティング
* 手順の詳細は[本家HP](https://docs.docker.com/toolbox/toolbox_install_windows/)を参照。
* Docker QuickStart Terminalの初回起動時に下のエラーが発生した場合

  ```
  Running pre-create checks... 
  Error with pre-create check: "This computer doesn't have VT-X/AMD-v enabled. Enabling it in the BIOS is mandatory" 
  ``` 
  * コマンドプロンプトで(Docker QuickStart Terminalではない)下のコマンドを実行する。

    ```
    docker-machine create default --virtualbox-no-vtx-check 
    ```

## 環境を動かす方法
### Jupyter notebookの起動
このGithubのレポジトリをpullした後、コマンドライン(Docker Toolboxの場合はDocker QuickStart Terminal)で下のコマンドを実行してカレントディレクトリを変更する。
```
cd [レポジトリをpullしたフォルダ]/HousePrices
```
Dockerのコンテナを起動するため、下のコマンドを実行する。
```
docker-compose up --build
```
しばらくすると下のようなURLが表示されるので、`http://127.0.0.1:8888`の方をコピーする。
```
To access the notebook, open this file in a browser:
    file:///home/jovyan/.local/share/jupyter/runtime/nbserver-7-open.html
Or copy and paste one of these URLs:
   http://dd918a6f7127:8888/?token=b059d85a3002c54cd20d4ff7292077145b19667f80198e1e
or http://127.0.0.1:8888/?token=b059d85a3002c54cd20d4ff7292077145b19667f80198e1e
```
ブラウザのURL入力欄にコピーしたURLを貼り付けてJupyter Notebookを開く。(開けなかったら127.0.0.1をlocalhostに変更して再度開く。)

### Docker Toolboxを停止する方法
コンテナを全て停止させた後、下のコマンドを実行する。
```
docker-machine.exe stop
```
仮想マシンが停止するので、`exit`コマンドを実行して、Docker Toolbox自体を停止させる。
