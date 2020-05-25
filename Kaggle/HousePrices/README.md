### データベース(MySQL)へのアクセス方法
任意のMySQLクライアントツールを使ってアクセスする。接続設定は下の通り。

| 設定項目       | 値        |
| :--- | :--- |
| Hostname       | 127.0.0.1 |
| Port           | 3306      |
| Username       | docker    |
| Password       | docker    |
| Default Schema | kaggle    |

上の設定をしても接続できない場合は、ファイアウォールで除外されている可能性ある。その場合はファイアウォールの設定を変更する。


### コード自動整形
メニューから`Edit`>`Apply Autopep8 Formatter`を選択するとコードが自動整形されます。
#### ショートカットに設定する方法
メニューから`Settings`>`Advanced Settings Editor`を選択し、`Settings`タブを表示させます。  
`Keyboard Shortcuts`を選択し、下の文字列を`User Preferences`に貼り付けます。

```
{  
  "shortcuts": [
      {  
          "command": "jupyterlab_code_formatter:autopep8",  
          "keys": [  
              "Ctrl B"  
          ],  
          "selector": ".jp-Notebook.jp-mod-editMode"  
      }
  ]  
}
```

`Ctrl+B`を押すとコードが自動整形されます。
