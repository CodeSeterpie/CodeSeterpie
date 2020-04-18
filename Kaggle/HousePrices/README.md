### データベース(MySQL)へのアクセス方法
下のURLを開き、phpMyAdminを起動してデータベースを操作します。  
http://localhost:8080/

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
