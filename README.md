# Eye Tracking 試験用ファイル

## 使用するモデル
- [dlib/data/shape_predictor_68_landmarks.dat](https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat)

## 参考にさせていただいたサイト,コード

demo_ver1
- [[OpenCV+dlib] 顔認識の実験](https://qiita.com/kotai2003/items/fb1f35da5437eefbc5da)

demo_ver2
- [dlibとopencvを使って画像から瞳の位置を取得【python】](https://cppx.hatenablog.com/entry/2017/12/25/231121)
- [ctare/face_expr](https://github.com/ctare/face_expr/blob/master/main.py)

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