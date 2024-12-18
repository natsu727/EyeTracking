# Eye Tracking 試験用ファイル

## 使用するモデル
- [dlib/data/shape_predictor_68_landmarks.dat](https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat)

## 参考にさせていただいたサイト,コード

demo_ver1
- [[OpenCV+dlib] 顔認識の実験](https://qiita.com/kotai2003/items/fb1f35da5437eefbc5da)

demo_ver2
- [dlibとopencvを使って画像から瞳の位置を取得【python】](https://cppx.hatenablog.com/entry/2017/12/25/231121)
- [ctare/face_expr](https://github.com/ctare/face_expr/blob/master/main.py)

- [PythonとOpenCV+dlibを用いた頭部方向推定](https://qiita.com/oozzZZZZ/items/1e68a7572bc5736d474e)

## 使用しているパッケージ・ライブラリ
- OpenCV 
	- Webカメラの取得、及び画像処理用
- dlib
	- 顔のランドマーク取得用
- cmake
	- dlibの補助用パッケージ
- boost
	- dlibの補助用パッケージ
- numpy
	- 取得した座標の再計算などに使用
- matplotlib
	- 取得した座標等の可視化
- pyautogui
	- PC画面サイズの取得
- imutils
	- OpenCVの補助
## 環境構築手順

### ローカルに構築する場合 ( 非推奨 )
*※実行環境: Ubuntu22.04.4LTS*
1. モデルのダウンロード<br/>
	上記の[使用するモデル](#使用するモデル)のリンクよりモデルをダウンロードし,`model`ファイルに置く
2. Anaconda でのパッケージインストール<br/>
	ターミナルで以下のコマンドを実行しパッケージをインストールする（目安:約30分くらいかかる）
	```{iscopy=true}
	conda env create -n 新しい環境名 -f demo-env.yml
	```
	2.が終了したら`python demo_**.py`で動かすことができる

### Dockerで実行する場合（ 推奨 ）
Dockerの環境構築はできているものとします
1. 	`docker build -t eyetrack .` を実行しイメージをビルドします<br>	 　→ imageファイルを作成しました.　以下を実行してください <br>
```{iscopy=true}
sudo docker load < ./.DockerFiles/image.tar
```
1. 次に`xhost +local:docker` を実行し Dockerから本体（ハードウェア管理サーバ）へのアクセスを行う権限を付与します
1. 最後に
	```{iscopy=true}
	sudo docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    --device /dev/video0:/dev/video0 \
    eyetrack
	```
	を実行しコンテナを実行します。<br>
	ここでは、コンテナ実行時にPC本体のカメラ・ディスプレイに関する情報をマウントしています。

1. 作業終了後必要に応じて`xhost -local:docker`を行いセキュリティ設定をもとに戻してください


---

https://onedrive.live.com/edit?id=A0E19CFE7FCA0F8F!133535&resid=A0E19CFE7FCA0F8F!133535&ithint=file%2cpptx&authkey=!APZXO08MspuvSwc&wdo=2&cid=a0e19cfe7fca0f8f

#### 追記 11月25日 

---
- ローカルのコードを編集したあとは、必ず `sudo docker build -t eyetrack .`を実行してdocker環境内に変更したファイルをCOPYするようにしてください。
- docker環境から出るには`exit`で出ることができます。
